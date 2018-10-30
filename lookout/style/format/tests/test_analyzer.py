from collections import defaultdict
import logging
import lzma
from pathlib import Path
import tarfile
from tempfile import TemporaryFile
from typing import Dict, Iterable, NamedTuple, Optional
import unittest

import bblfsh
from bblfsh.client import BblfshClient

from lookout.core.analyzer import ReferencePointer
from lookout.core.api.service_data_pb2 import File
from lookout.style.format.analyzer import FormatAnalyzer
from lookout.style.format.model import FormatModel
from lookout.style.format.tests.test_model import compare_models

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
    @staticmethod
    def get_files_from_tar(tar_path: str) -> Dict[str, File]:
        files = defaultdict(lambda: [None, None])
        with tarfile.open(tar_path) as tar:
            for member in tar:
                name = member.name
                if name == ".":
                    continue
                file = tar.extractfile(member)
                uast = True if name.endswith(".uast") else False
                content = file.read()
                if uast:
                    name = name[:-5]
                    content = bblfsh.Node.FromString(content)
                files[name][uast] = content
        for key, (content, uast) in files.items():
            files[key] = File(path=key, content=content, uast=uast, language="JavaScript")
        return files

    @classmethod
    def setUpClass(cls):
        logging.basicConfig(level=logging.INFO)
        logging.getLogger("FormatAnalyzer").setLevel(logging.DEBUG)
        base = Path(__file__).parent
        # str() is needed for Python 3.5
        with lzma.open(str(base / "benchmark.uast.xz")) as fin:
            cls.uast = bblfsh.Node.FromString(fin.read())
        cls.base_files = cls.get_files_from_tar(str(base / "freecodecamp-base.tar.xz"))
        cls.head_files = cls.get_files_from_tar(str(base / "freecodecamp-head.tar.xz"))
        cls.ptr = ReferencePointer("someurl", "someref", "somecommit")

    def test_train(self):
        datastub = FakeDataStub(files=self.base_files.values(), changes=None)
        config = {"global": {"n_iter": 1}}
        model1 = FormatAnalyzer.train(self.ptr, config, datastub)
        self.assertIsInstance(model1, FormatModel)
        self.assertIn("javascript", model1, str(model1))
        datastub = FakeDataStub(files=self.base_files.values(), changes=None)
        config = {"global": {"n_iter": 1}}
        model2 = FormatAnalyzer.train(self.ptr, config, datastub)
        self.assertEqual(model1["javascript"].rules, model2["javascript"].rules)
        self.assertGreater(len(model1["javascript"]), 10)
        # Check that model can be saved without problems and then load back
        with TemporaryFile(prefix="analyzer_model-", suffix=".asdf") as f:
            model2.save(f)
            f.seek(0)
            model3 = FormatModel().load(f)
            compare_models(self, model2, model3)

    def test_analyze(self):
        common = self.base_files.keys() & self.head_files.keys()
        datastub = FakeDataStub(files=self.base_files.values(),
                                changes=[Change(base=self.base_files[k], head=self.head_files[k])
                                         for k in common])
        config = {"global": {"n_iter": 1}}
        model = FormatAnalyzer.train(self.ptr, config, datastub)
        analyzer = FormatAnalyzer(model, self.ptr.url, {})
        client = BblfshClient("0.0.0.0:9432")
        comments = analyzer.analyze(self.ptr, self.ptr, datastub, client)
        self.assertGreater(len(comments), 0)

    def test_file_filtering(self):
        datastub = FakeDataStub(files=self.base_files.values(), changes=None)
        config = {"global": {"n_iter": 1, "line_length_limit": 0}}
        model_trained = FormatAnalyzer.train(self.ptr, config, datastub)
        self.assertEqual(len(model_trained._rules_by_lang), 0)
        config = {"global": {"n_iter": 1, "line_length_limit": 500}}
        model_trained = FormatAnalyzer.train(self.ptr, config, datastub)
        self.assertGreater(len(model_trained._rules_by_lang), 0)


if __name__ == "__main__":
    unittest.main()
