from collections import defaultdict
import logging
import lzma
from pathlib import Path
import tarfile
from tempfile import TemporaryFile
from typing import Dict, Iterable, NamedTuple, Optional
import unittest

import bblfsh
from lookout.core.analyzer import ReferencePointer
from lookout.core.api.service_data_pb2 import File

from lookout.style.format.analyzer import FormatAnalyzer
from lookout.style.format.feature_extractor import FeatureExtractor
from lookout.style.format.model import FormatModel
from lookout.style.format.tests.test_model import compare_models

Change = NamedTuple("Change", [("base", File), ("head", File)])


def get_train_config():
    return {
        "global": {
            "feature_extractor": {
                "left_siblings_window": 2,
                "right_siblings_window": 2,
                "parents_depth": 2,
                "select_features_number": 250,
                "return_sibling_indices": False,
                "cutoff_label_support": 5,
            },
            "trainable_rules": {
                "prune_branches_algorithms": ["reduced-error"],
                "top_down_greedy_budget": [False, .5],
                "prune_attributes": False,
                "uncertain_attributes": False,
                "n_estimators": 3,
                "random_state": 42,
            },
            "n_iter": 1,
        },
    }


def get_analyze_config():
    return {
        "confidence_threshold": 0.9,
        "support_threshold": 10,
        "uast_break_check": False,
    }


class FakeDataService:
    def __init__(self, files: Optional[Iterable[File]], changes: Optional[Iterable[Change]]):
        self.data_stub = FakeDataStub(files, changes)
        self.bblfsh_client = bblfsh.BblfshClient("0.0.0.0:9432")

    def get_data(self):
        return self.data_stub

    def get_bblfsh(self):
        return self.bblfsh_client._stub


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
        FeatureExtractor._log.level = logging.DEBUG

    def tearDown(self):
        if hasattr(self, "data_service"):
            self.data_service.bblfsh_client._channel.close()

    def test_train(self):
        self.data_service = FakeDataService(files=self.base_files.values(), changes=None)
        model1 = FormatAnalyzer.train(self.ptr, get_train_config(), self.data_service)
        self.assertIsInstance(model1, FormatModel)
        self.assertIn("javascript", model1, str(model1))
        model2 = FormatAnalyzer.train(self.ptr, get_train_config(), self.data_service)
        self.assertEqual(model1["javascript"].rules, model2["javascript"].rules)
        self.assertGreater(len(model1["javascript"]), 10)
        # Check that model can be saved without problems and then load back
        with TemporaryFile(prefix="analyzer_model-", suffix=".asdf") as f:
            model2.save(f)
            f.seek(0)
            model3 = FormatModel().load(f)
            compare_models(self, model2, model3)

    def test_train_cutoff_labels(self):
        self.data_service = FakeDataService(files=self.base_files.values(), changes=None)
        model1 = FormatAnalyzer.train(self.ptr, get_train_config(), self.data_service)
        self.assertIsInstance(model1, FormatModel)
        self.assertIn("javascript", model1, str(model1))
        model2 = FormatAnalyzer.train(self.ptr, get_train_config(), self.data_service)
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
        self.data_service = FakeDataService(
            files=self.base_files.values(),
            changes=[Change(base=self.base_files[k], head=self.head_files[k])
                     for k in common])
        config = get_train_config()
        # Make uast_break_check only here
        config["uast_break_check"] = True
        model = FormatAnalyzer.train(self.ptr, get_train_config(), self.data_service)
        analyzer = FormatAnalyzer(model, self.ptr.url, get_analyze_config())
        comments = analyzer.analyze(self.ptr, self.ptr, self.data_service)
        self.assertGreater(len(comments), 0)

    def test_file_filtering(self):
        self.data_service = FakeDataService(files=self.base_files.values(), changes=None)
        config = get_train_config()
        config["global"]["line_length_limit"] = 0
        model_trained = FormatAnalyzer.train(self.ptr, config, self.data_service)
        self.assertEqual(len(model_trained._rules_by_lang), 0)
        config["global"]["line_length_limit"] = 500
        model_trained = FormatAnalyzer.train(self.ptr, config, self.data_service)
        self.assertGreater(len(model_trained._rules_by_lang), 0)


if __name__ == "__main__":
    unittest.main()
