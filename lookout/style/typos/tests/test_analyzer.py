import logging
import lzma
from pathlib import Path
from typing import Iterable, NamedTuple, Optional
import unittest

import bblfsh

from lookout.core.analyzer import ReferencePointer
from lookout.core.api.service_data_pb2 import File
from lookout.style.typos.analyzer import IdTyposAnalyzer
from lookout.style.typos.model import IdTyposModel

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
        self.assertIsInstance(model, IdTyposModel)

    @unittest.skip
    def test_analyze(self):
        datastub = FakeDataStub(files=self.base_files,
                                changes=[Change(base=self.base_files[0], head=self.head_files[0])])
        model = IdTyposAnalyzer.train(self.ptr, {}, datastub)
        analyzer = IdTyposAnalyzer(model, self.ptr.url, {})
        comments = analyzer.analyze(self.ptr, self.ptr, datastub)
        self.assertGreater(len(comments), 0)


if __name__ == "__main__":
    unittest.main()
