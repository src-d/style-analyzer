import lzma
from pathlib import Path
import tarfile
from typing import NamedTuple, Sequence
import unittest

import bblfsh

from lookout.core.analyzer import ReferencePointer
from lookout.core.api.service_data_pb2 import File
from lookout.style.format.analyzer import FormatAnalyzer


Change = NamedTuple("Change", [("base", File), ("head", File)])


class FakeDataStub:
    def __init__(self, files: Sequence[File] = [], changes: Sequence[Change] = []):
        self.files = files
        self.changes = changes

    def GetFiles(self, request):
        return self.files

    def GetChanges(self, request):
        return self.changes


class AnalyzerTests(unittest.TestCase):

    @staticmethod
    def get_files_from_tar(tar_path, bblfsh_client):
        files = {}
        with tarfile.open(tar_path) as tar:
            for i, member in enumerate(tar):
                if i > 10:
                    break
                name = member.name
                file = tar.extractfile(member)
                if file is None:
                    continue
                bytes_content = file.read()
                res = bblfsh_client.parse('', language="javascript", contents=bytes_content)
                if res.status != 0:
                    continue
                files[name] = File(path=name, content=bytes_content, uast=res.uast,
                                   language="JavaScript")
        return files

    @classmethod
    def setUpClass(cls):
        base = Path(__file__).parent
        # str() is needed for Python 3.5
        with lzma.open(str(base / "benchmark.uast.xz")) as fin:
            cls.uast = bblfsh.Node.FromString(fin.read())
        cls.base_files = []
        client = bblfsh.BblfshClient("0.0.0.0:9432")
        cls.base_files = cls.get_files_from_tar(str(base / "freecodecamp-base.tar.xz"), client)
        cls.head_files = cls.get_files_from_tar(str(base / "freecodecamp-head.tar.xz"), client)

    def test_files_by_language(self):
        file_stats = {"js": 2, "python": 5, "ruby": 7}
        files = []
        for language, n_files in file_stats.items():
            for i in range(n_files):
                files.append(File(language=language, uast=self.uast, path=str(i)))
        result = FormatAnalyzer.files_by_language(files)
        self.assertEqual(file_stats, {k: len(v) for k, v in result.items()})
        return result

    def test_train(self):
        datastub = FakeDataStub(files=self.base_files.values())
        ptr = ReferencePointer("https://youtu.be/dQw4w9WgXcQ", "refs/heads/master", "somecommit")
        config = {"n_iter": 1}
        FormatAnalyzer.train(ptr, config, datastub)

    def test_analyze(self):
        common = self.base_files.keys() & self.head_files.keys()
        datastub = FakeDataStub(files=self.base_files.values(),
                                changes=[Change(base=self.base_files[k], head=self.head_files[k])
                                         for k in common])
        ptr = ReferencePointer("https://youtu.be/dQw4w9WgXcQ", "refs/heads/master", "somecommit")
        config = {"n_iter": 1}
        model = FormatAnalyzer.train(ptr, config, datastub)
        analyzer = FormatAnalyzer(model, ptr.url, {})
        analyzer.analyze(ptr, ptr, datastub)


if __name__ == "__main__":
    unittest.main()
