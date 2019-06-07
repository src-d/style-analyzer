import logging
import lzma
from pathlib import Path
from typing import NamedTuple
import unittest

import bblfsh
from lookout.core.analyzer import ReferencePointer
from lookout.core.api.service_data_pb2 import File
import pandas

from lookout.style.format.benchmarks.general_report import FakeDataService
from lookout.style.typos.analyzer import IDENTIFIER_INDEX_COLUMN, IdTyposAnalyzer, IdTyposModel
from lookout.style.typos.utils import Candidate, Columns

Change = NamedTuple("Change", [("base", File), ("head", File)])
MODEL_PATH = str(Path(__file__).parent / "test_corrector.asdf")


class FakeFile:
    def __init__(self, path, content, uast, language):
        self.path = path
        self.content = content
        self.uast = uast
        self.language = language


class AnalyzerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logging.basicConfig(level=logging.INFO)
        logging.getLogger("IdTyposAnalyzer").setLevel(logging.DEBUG)
        base = Path(__file__).parent
        # str() is needed for Python 3.5
        cls.bblfsh_client = bblfsh.BblfshClient("0.0.0.0:9432")
        with lzma.open(str(base / "test_base_file.js.xz")) as fin:
            contents = fin.read()
            uast = cls.bblfsh_client.parse("test_base_file.js", contents=contents).uast
            cls.base_files = [FakeFile(path="test_base_file.js", content=contents, uast=uast,
                                       language="Javascript")]
        with lzma.open(str(base / "test_head_file.js.xz")) as fin:
            contents = fin.read()
            uast = cls.bblfsh_client.parse("test_head_file.js", contents=contents).uast
            cls.head_files = [FakeFile(path="test_head_file.js", content=contents, uast=uast,
                                       language="Javascript")]
        cls.ptr = ReferencePointer("someurl", "someref", "somecommit")

    @classmethod
    def tearDownClass(cls):
        cls.bblfsh_client._channel.close()

    def test_train(self):
        dataservice = FakeDataService(
            self.bblfsh_client, files=self.base_files, changes=[])
        model = IdTyposAnalyzer.train(ptr=self.ptr, config={}, data_service=dataservice)
        self.assertSetEqual(model.identifiers, {"name", "print_type", "get_length",
                                                "customidentifiertostore"})

    def test_analyze(self):
        dataservice = FakeDataService(
            self.bblfsh_client, files=self.base_files,
            changes=[Change(base=self.base_files[0], head=self.head_files[0])])
        model = IdTyposAnalyzer.train(ptr=self.ptr, config={}, data_service=dataservice)
        analyzer = IdTyposAnalyzer(model=model, url=self.ptr.url, config=dict(
            model=MODEL_PATH, confidence_threshold=0.0, n_candidates=3,
            check_all_identifiers=False))
        comments = analyzer.analyze(ptr_from=self.ptr, ptr_to=self.ptr, data_service=dataservice)
        self.assertGreater(len(comments), 0)
        bad_names = ["nam", "print_tipe", "gett_lenght"]
        good_names = ["name", "print_type", "get_length", "customidentifiertostore"]
        for c in comments:
            self.assertFalse(any(name in c.text.split(", fixes:")[0] for name in good_names))
            self.assertTrue(any(name in c.text.split(", fixes:")[0] for name in bad_names))

        analyzer = IdTyposAnalyzer(model=model, url=self.ptr.url, config=dict(
            model=MODEL_PATH, confidence_threshold=0.0, n_candidates=3,
            check_all_identifiers=True))
        comments = analyzer.analyze(ptr_from=self.ptr, ptr_to=self.ptr, data_service=dataservice)
        self.assertGreater(len(comments), 0)
        bad_names = ["nam", "print_tipe", "gett_lenght", "customidentifiertostore"]
        good_names = ["name", "print_type", "get_length"]
        for c in comments:
            self.assertFalse(any(name in c.text.split(", fixes:")[0] for name in good_names))
            self.assertTrue(any(name in c.text.split(", fixes:")[0] for name in bad_names))

    def test_reconstruct_identifier(self):
        tokens = [
            ("UpperCamelCase", "UpperComelCase", ["upper", "camel", "case"]),
            ("camelCase", "comelCase", ["camel", "case"]),
            ("FRAPScase", "FRAPScase", ["frap", "scase"]),
            ("SQLThing", "SQLThing", ["sql", "thing"]),
            ("_Astra", "_Ostra", ["astra"]),
            ("CAPS_CONST", "COPS_CONST", ["caps", "const"]),
            ("_something_SILLY_", "_something_SIILLY_", ["something", "silly"]),
            ("blink182", "blunk182", ["blink"]),
            ("FooBar100500Bingo", "FuBar100500Bingo", ["foo", "bar", "bingo"]),
            ("Man45var", "Men45var", ["man", "var"]),
            ("method_name", "metod_name", ["method", "name"]),
            ("Method_Name", "Metod_Name", ["method", "name"]),
            ("101dalms", "101dolms", ["dalms"]),
            ("101_dalms", "101_dolms", ["dalms"]),
            ("101_DalmsBug", "101_DolmsBug", ["dalms", "bug"]),
            ("101_Dalms45Bug7", "101_Dolms45Bug7", ["dalms", "bug"]),
            ("wdSize", "pwdSize", ["wd", "size"]),
            ("Glint", "Glunt", ["glint"]),
            ("foo_BAR", "fu_BAR", ["foo", "bar"]),
            ("sourced.ml.algorithms.uast_ids_to_bag", "source.ml.algorithmos.uast_ids_to_bags",
             ["sourced", "ml", "algorithms",
              "uast", "ids", "to", "bag"]),
            ("WORSTnameYOUcanIMAGINE", "WORSTnomeYOUcanIMGINE",
             ["wors", "tname", "yo", "ucan", "imagine"]),
            ("SmallIdsToFoOo", "SmallestIdsToFoOo", ["small", "ids", "to", "fo", "oo"]),
            ("SmallIdFooo", "SmallestIdFooo", ["small", "id", "fooo"]),
            ("ONE_M0re_.__badId.example", "ONE_M0ree_.__badId.exomple",
             ["one", "m", "re", "bad", "id", "example"]),
            ("never_use_Such__varsableNames", "never_use_Such__varsablezzNameszz",
             ["never", "use", "such", "varsable", "names"]),
            ("a.b.c.d", "a.b.ce.de", ["a", "b", "c", "d"]),
            ("A.b.Cd.E", "A.be.Cde.Ee", ["a", "b", "cd", "e"]),
            ("looong_sh_loooong_sh", "looongzz_shzz_loooongzz_shzz",
             ["looong", "sh", "loooong", "sh"]),
            ("sh_sh_sh_sh", "ch_ch_ch_ch", ["sh", "sh", "sh", "sh"]),
            ("loooong_loooong_loooong", "laoong_loaong_looang", ["loooong", "loooong", "loooong"]),
        ]

        parser = IdTyposAnalyzer.create_token_parser()

        for correct, corrupted, correct_tokens in tokens:
            self.assertEqual(correct,
                             IdTyposAnalyzer.reconstruct_identifier(parser,
                                                                    pred_tokens=correct_tokens,
                                                                    identifier=corrupted))

    def test_reconstruct_identifier_fail(self):
        tokens = [
            ("UpperCamelCase", ["upper", "camel", "case", "fail"]),
        ]

        parser = IdTyposAnalyzer.create_token_parser()

        for identifier, splitted_tokens in tokens:
            with self.assertRaises(AssertionError):
                IdTyposAnalyzer.reconstruct_identifier(parser, pred_tokens=splitted_tokens,
                                                       identifier=identifier)


class AnalyzerPayloadTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.checker = IdTyposAnalyzer(
            model=IdTyposModel(), url="", config=dict(
                model=MODEL_PATH, confidence_threshold=0.2, n_candidates=3))
        cls.identifiers = ["get", "gpt_tokeb"]
        cls.test_df = pandas.DataFrame(
            [[0, "get", "get"], [1, "gpt tokeb", "gpt"], [1, "gpt tokeb", "tokeb"]],
            columns=[IDENTIFIER_INDEX_COLUMN, Columns.Split, Columns.Token])
        cls.suggestions = {1: [Candidate("get", 0.9),
                               Candidate("gpt", 0.3)],
                           2: [Candidate("token", 0.98),
                               Candidate("taken", 0.3),
                               Candidate("tokem", 0.01)]}
        cls.filtered_suggestions = {1: [Candidate("get", 0.9),
                                        Candidate("gpt", 0.3)],
                                    2: [Candidate("token", 0.98),
                                        Candidate("taken", 0.3)]}

    def test_filter_suggestions(self):
        self.assertDictEqual(self.checker.filter_suggestions(self.test_df, self.suggestions),
                             self.filtered_suggestions)

    def test_check_identifiers(self):
        suggestions = self.checker.check_identifiers(self.identifiers)
        self.assertTrue(set(suggestions.keys()).issubset(set(range(len(self.identifiers)))))


if __name__ == "__main__":
    unittest.main()
