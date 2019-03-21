from collections import defaultdict
import logging
import lzma
from pathlib import Path
import tarfile
from tempfile import TemporaryFile
from typing import Dict, NamedTuple
import unittest

import bblfsh
from lookout.core.analyzer import ReferencePointer
from lookout.core.api.service_data_pb2 import File

from lookout.style.format.analyzer import FormatAnalyzer
from lookout.style.format.benchmarks.general_report import FakeDataService
from lookout.style.format.feature_extractor import FeatureExtractor
from lookout.style.format.model import FormatModel
from lookout.style.format.tests.test_model import compare_models

Change = NamedTuple("Change", [("base", File), ("head", File)])


def get_config():
    return {
        "train": {
            "language_defaults": {
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
                    "n_estimators": 3,
                },
                "optimizer": {
                    "n_iter": 6,
                },
                "random_state": 42,
                "lines_ratio_train_trigger": 0.8,
                "overall_size_limit": 2 * 10 ** 4,
            },
        },
        "analyze": {
            "language_defaults": {
                "confidence_threshold": 0.9,
                "support_threshold": 10,
                "uast_break_check": False,
            },
        },
    }


class FakeUAST:
    def __init__(self):
        self.children = []

    def SerializeToString(self):
        return bblfsh.Node().SerializeToString()


class FakeFile:
    def __init__(self, path, content, uast, language):
        self.path = path
        self.content = content
        self.uast = uast
        self.language = language


def remove_uast(file):
    return FakeFile(path=file.path, content=file.content, uast=FakeUAST(),
                    language="JavaScript")


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
        cls.bblfsh_client = bblfsh.BblfshClient("0.0.0.0:9432")

    @classmethod
    def tearDownClass(cls):
        cls.bblfsh_client._channel.close()

    def test_train(self):
        self.data_service = FakeDataService(
            self.bblfsh_client, files=self.base_files.values(), changes=[])
        model1 = FormatAnalyzer.train(self.ptr, get_config(), self.data_service)
        self.assertIsInstance(model1, FormatModel)
        self.assertIn("javascript", model1, str(model1))
        model2 = FormatAnalyzer.train(self.ptr, get_config(), self.data_service)
        self.assertEqual(model1["javascript"].rules, model2["javascript"].rules)
        self.assertGreater(len(model1["javascript"]), 5)
        # Check that model can be saved without problems and then load back
        with TemporaryFile(prefix="analyzer_model-", suffix=".asdf") as f:
            model2.save(f)
            f.seek(0)
            model3 = FormatModel().load(f)
            compare_models(self, model2, model3)

    def test_train_check(self):
        common = self.base_files.keys() & self.head_files.keys()
        self.data_service = FakeDataService(
            self.bblfsh_client,
            files=self.base_files.values(),
            changes=[Change(base=self.base_files[k], head=self.head_files[k])
                     for k in common])
        model = FormatAnalyzer.train(self.ptr, get_config(), self.data_service)
        required = FormatAnalyzer.check_training_required(
            model, self.ptr, get_config(), self.data_service)
        self.assertFalse(required)
        self.data_service = FakeDataService(
            self.bblfsh_client,
            files=self.base_files.values(),
            changes=[Change(base=remove_uast(self.base_files[k]), head=self.head_files[k])
                     for k in common])
        required = FormatAnalyzer.check_training_required(
            model, self.ptr, get_config(), self.data_service)
        self.assertTrue(required)

    def test_train_cutoff_labels(self):
        self.data_service = FakeDataService(
            self.bblfsh_client, files=self.base_files.values(), changes=[])
        model1 = FormatAnalyzer.train(self.ptr, get_config(), self.data_service)
        self.assertIsInstance(model1, FormatModel)
        self.assertIn("javascript", model1, str(model1))
        model2 = FormatAnalyzer.train(self.ptr, get_config(), self.data_service)
        self.assertEqual(model1["javascript"].rules, model2["javascript"].rules)
        self.assertGreater(len(model1["javascript"]), 5)
        # Check that model can be saved without problems and then load back
        with TemporaryFile(prefix="analyzer_model-", suffix=".asdf") as f:
            model2.save(f)
            f.seek(0)
            model3 = FormatModel().load(f)
            compare_models(self, model2, model3)

    def test_analyze(self):
        common = self.base_files.keys() & self.head_files.keys()
        self.data_service = FakeDataService(
            self.bblfsh_client,
            files=self.base_files.values(),
            changes=[Change(base=remove_uast(self.base_files[k]), head=self.head_files[k])
                     for k in common])
        config = get_config()
        # Make uast_break_check only here
        config["analyze"]["language_defaults"]["uast_break_check"] = True
        model = FormatAnalyzer.train(self.ptr, config, self.data_service)
        analyzer = FormatAnalyzer(model, self.ptr.url, config)
        comments = analyzer.analyze(self.ptr, self.ptr, self.data_service)
        self.assertGreater(len(comments), 0)

    def test_file_filtering(self):
        self.data_service = FakeDataService(
            self.bblfsh_client, files=self.base_files.values(), changes=[])
        config = get_config()
        config["train"]["language_defaults"]["line_length_limit"] = 0
        model_trained = FormatAnalyzer.train(self.ptr, config, self.data_service)
        self.assertEqual(len(model_trained._rules_by_lang), 0)
        config["train"]["language_defaults"]["line_length_limit"] = 500
        model_trained = FormatAnalyzer.train(self.ptr, config, self.data_service)
        self.assertGreater(len(model_trained._rules_by_lang), 0)


if __name__ == "__main__":
    unittest.main()
