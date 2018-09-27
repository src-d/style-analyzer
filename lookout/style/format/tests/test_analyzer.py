from collections import defaultdict
import logging
import lzma
from pathlib import Path
import tarfile
from typing import NamedTuple, Dict, Optional, Iterable
import unittest

import bblfsh

from lookout.core.analyzer import ReferencePointer
from lookout.core.api.service_data_pb2 import File
from lookout.style.format.analyzer import FormatAnalyzer
from lookout.style.format.model import FormatModel

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
            for i, member in enumerate(tar):
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

    def test_files_by_language(self):
        file_stats = {"js": 2, "Python": 5, "ruby": 7}
        files = []
        for language, n_files in file_stats.items():
            for i in range(n_files):
                files.append(File(language=language, uast=self.uast, path=str(i)))
        result = FormatAnalyzer._files_by_language(files)
        self.assertEqual({"js": 2, "python": 5, "ruby": 7}, {k: len(v) for k, v in result.items()})
        return result

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

    def test_analyze(self):
        common = self.base_files.keys() & self.head_files.keys()
        datastub = FakeDataStub(files=self.base_files.values(),
                                changes=[Change(base=self.base_files[k], head=self.head_files[k])
                                         for k in common])
        config = {"global": {"n_iter": 1}}
        model = FormatAnalyzer.train(self.ptr, config, datastub)
        analyzer = FormatAnalyzer(model, self.ptr.url, {})
        comments = analyzer.analyze(self.ptr, self.ptr, datastub)
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
