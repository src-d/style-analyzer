import logging
import lzma
from pathlib import Path
from typing import Iterable, NamedTuple, Optional
import unittest

import bblfsh
from lookout.core.analyzer import DummyAnalyzerModel, ReferencePointer
from lookout.core.api.service_data_pb2 import File
import pandas

from lookout.style.typos.analyzer import IdTyposAnalyzer
from lookout.style.typos.utils import Columns

Change = NamedTuple("Change", [("base", File), ("head", File)])


class FakeDataStub:
    def __init__(self, files: Optional[Iterable[File]], changes: Optional[Iterable[Change]]):
        self.files = files
        self.changes = changes

    def GetFiles(self, _):
        return self.files

    def GetChanges(self, _):
        return self.changes


class AnalyzerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logging.basicConfig(level=logging.INFO)
        logging.getLogger("IdTyposAnalyzer").setLevel(logging.DEBUG)
        base = Path(__file__).parent
        # str() is needed for Python 3.5
        client = bblfsh.BblfshClient("0.0.0.0:9432")
        with lzma.open(str(base / "test_base_file.py.xz")) as fin:
            uast = client.parse("test_base_file.py", contents=fin.read()).uast
            cls.base_files = [File(path="test_base_file.py", content=fin.read(), uast=uast,
                                   language="Python")]
        with lzma.open(str(base / "test_head_file.py.xz")) as fin:
            uast = client.parse("test_head_file.py", contents=fin.read()).uast
            cls.head_files = [File(path="test_head_file.py", content=fin.read(), uast=uast,
                                   language="Python")]
        cls.ptr = ReferencePointer("someurl", "someref", "somecommit")

    @unittest.skip
    def test_train(self):
        datastub = FakeDataStub(files=self.base_files, changes=None)
        model = IdTyposAnalyzer.train(self.ptr, {}, datastub)
        self.assertIsInstance(model, DummyAnalyzerModel)

    @unittest.skip
    def test_analyze(self):
        datastub = FakeDataStub(files=self.base_files,
                                changes=[Change(base=self.base_files[0], head=self.head_files[0])])
        model = IdTyposAnalyzer.train(self.ptr, {}, datastub)
        analyzer = IdTyposAnalyzer(model, self.ptr.url, {})
        comments = analyzer.analyze(self.ptr, self.ptr, datastub)
        self.assertGreater(len(comments), 0)

    def test_reconstruct_identifier(self):
        tokens = [
            ("UpperCamelCase", "UpperComelCase", ["upper", "camel", "case"]),
            ("camelCase", "comelCase", ["camel", "case"]),
            ("FRAPScase", "FRAPScase", ["fraps", "case"]),
            ("SQLThing", "SQLThing", ["sqlt", "hing"]),
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
             ["worst", "name", "you", "can", "imagine"]),
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


@unittest.skip("sample_corrector.asdf needs to be generated")
class AnalyzerPayloadTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.checker = IdTyposAnalyzer(
            DummyAnalyzerModel(), "", config=dict(
                model=str(Path(__file__).parent / "sample_corrector.asdf"),
                confidence_threshold=0.2, n_candidates=3))
        cls.identifiers = ["get", "gpt_tokeb"]
        cls.test_df = pandas.DataFrame(
            [[0, "get", "get"], [1, "gpt tokeb", "gpt"], [1, "gpt tokeb", "tokeb"]],
            columns=[IdTyposAnalyzer.INDEX_COLUMN, Columns.Split, Columns.Token])
        cls.suggestions = {1: [("get", 0.9),
                               ("gpt", 0.3)],
                           2: [("token", 0.98),
                               ("taken", 0.3),
                               ("tokem", 0.01)]}
        cls.filtered_suggestions = {1: [("get", 0.9)],
                                    2: [("token", 0.98),
                                        ("taken", 0.3)]}

    def test_filter_suggestions(self):
        self.assertDictEqual(self.checker.filter_suggestions(self.test_df, self.suggestions),
                             self.filtered_suggestions)

    def test_check_identifiers(self):
        suggestions = self.checker.check_identifiers(self.identifiers)
        self.assertTrue(set(suggestions.keys()).issubset(set(range(len(self.identifiers)))))


if __name__ == "__main__":
    unittest.main()
