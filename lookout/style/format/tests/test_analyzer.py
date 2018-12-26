import glob
import logging
import os
from pathlib import Path
import shutil
import tarfile
from tempfile import TemporaryFile
from typing import NamedTuple
import unittest

import bblfsh
from lookout.core.analyzer import ReferencePointer
from lookout.core.api.service_data_pb2 import File
from lookout.core.lib import filter_files

from lookout.style.format.analyzer import FormatAnalyzer
from lookout.style.format.benchmarks.general_report import FakeDataService
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


class AnalyzerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logging.basicConfig(level=logging.INFO)
        logging.getLogger("FormatAnalyzer").setLevel(logging.DEBUG)
        cls.ptr = ReferencePointer("someurl", "someref", "somecommit")
        FeatureExtractor._log.level = logging.DEBUG
        cls.bblfsh_client = bblfsh.BblfshClient("0.0.0.0:9432")
        parent_loc = Path(__file__).parent.resolve()
        with tarfile.open(str(parent_loc / "jquery_noisy.tar.xz")) as tar:
            tar.extractall()
        cls.jquery_dir = str(Path(parent_loc) / "jquery_noisy")
        base_jquery_pattern = os.path.join(cls.jquery_dir, "jquery", "*.js")
        head_jquery_pattern = os.path.join(cls.jquery_dir, "jquery_noisy", "*.js")
        cls.base_filenames = glob.glob(base_jquery_pattern, recursive=True)
        cls.head_filenames = glob.glob(head_jquery_pattern, recursive=True)
        cls.base_files = filter_files(filenames=cls.base_filenames,
                                      line_length_limit=500,
                                      client=cls.bblfsh_client,
                                      language="javascript")
        cls.head_files = filter_files(filenames=cls.head_filenames,
                                      line_length_limit=500,
                                      client=cls.bblfsh_client,
                                      language="javascript")

    @classmethod
    def tearDownClass(cls):
        cls.bblfsh_client._channel.close()
        shutil.rmtree(cls.jquery_dir)

    def test_train(self):
        self.data_service = FakeDataService(bblfsh_client=self.bblfsh_client,
                                            files=self.base_files,
                                            changes=[])
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
        self.data_service = FakeDataService(bblfsh_client=self.bblfsh_client,
                                            files=self.base_files,
                                            changes=[])
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
        self.data_service = FakeDataService(
            bblfsh_client=self.bblfsh_client,
            files=self.base_files,
            changes=[Change(base=b, head=h) for b, h in zip(self.base_files, self.head_files)])
        config = get_train_config()
        # Make uast_break_check only here
        config["uast_break_check"] = True
        model = FormatAnalyzer.train(self.ptr, get_train_config(), self.data_service)
        analyzer = FormatAnalyzer(model, self.ptr.url, get_analyze_config())
        comments = analyzer.analyze(self.ptr, self.ptr, self.data_service)
        self.assertGreater(len(comments), 0)

    def test_file_filtering(self):
        self.data_service = FakeDataService(
            bblfsh_client=self.bblfsh_client,
            files=filter_files(filenames=self.base_filenames, line_length_limit=0,
                               client=self.bblfsh_client, language="javascript"),
            changes=[])
        config = get_train_config()
        model_trained = FormatAnalyzer.train(self.ptr, config, self.data_service)
        self.assertEqual(len(model_trained._rules_by_lang), 0)
        self.data_service = FakeDataService(bblfsh_client=self.bblfsh_client,
                                            files=self.base_files,
                                            changes=[])
        model_trained = FormatAnalyzer.train(self.ptr, config, self.data_service)
        self.assertGreater(len(model_trained._rules_by_lang), 0)


if __name__ == "__main__":
    unittest.main()
