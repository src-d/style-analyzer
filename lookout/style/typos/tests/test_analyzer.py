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
from lookout.style.typos.utils import SPLIT_COLUMN, TYPO_COLUMN

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
            columns=[IdTyposAnalyzer.INDEX_COLUMN, SPLIT_COLUMN, TYPO_COLUMN])
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
