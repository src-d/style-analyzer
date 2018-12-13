import glob
import io
import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest

from lookout.core.test_helpers import server

from lookout.style.format.benchmarks.general_report import print_reports, QualityReportAnalyzer
from lookout.style.format.benchmarks.top_repos_quality import _get_json_data, _get_model_summary, \
    _get_precision_recall_f1_support
from lookout.style.format.tests.test_analyzer import get_analyze_config, get_train_config
from lookout.style.format.tests.test_analyzer_integration import (
    FROM_COMMIT, TestAnalyzer, TO_COMMIT)


class PretrainedModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Prepare environment & train the model for tests."""
        # required config
        cls.bblfsh = "0.0.0.0:9432"
        cls.language = "javascript"

        # analyzer
        parent_loc = Path(__file__).parent.resolve()
        cls.base_dir_ = tempfile.TemporaryDirectory()
        cls.base_dir = cls.base_dir_.name
        cls.port = server.find_port()
        # extract repo
        cls.jquery_dir = os.path.join(cls.base_dir, "jquery")
        # str() is needed for Python 3.5
        with tarfile.open(str(parent_loc / "jquery.tar.xz")) as tar:
            tar.extractall(path=cls.base_dir)
        files = glob.glob(os.path.join(cls.jquery_dir, "**", "*"), recursive=True)
        assert len(files) == 15, len(files)
        cls.model_path = os.path.join(str(parent_loc), "model_jquery.asdf")

    @classmethod
    def tearDownClass(cls):
        """Remove temporary directory with jquery repository."""
        cls.base_dir_.cleanup()

    def setUp(self):
        self.port = server.find_port()
        self.db = tempfile.NamedTemporaryFile(dir=self.base_dir)
        self.fs = tempfile.TemporaryDirectory(dir=self.base_dir)

    def tearDown(self):
        self.fs.cleanup()
        self.db.close()


class Capturing(list):
    """Capture prints."""
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = io.StringIO()
        return self

    def __exit__(self, *args):
        value = self._stringio.getvalue()
        self.extend(value.splitlines())
        del self._stringio    # free up some memory
        sys.stdout = self._stdout


class QualityReportTests(PretrainedModelTests):
    def test_eval_empty_input(self):
        """Test on empty folder - expect only model report."""
        with tempfile.TemporaryDirectory() as folder:
            input_pattern = os.path.join(folder, "**", "*")
            with Capturing() as output:
                print_reports(input_pattern=input_pattern, bblfsh=self.bblfsh,
                              language=self.language, model_path=self.model_path,
                              config={"uast_break_check": False})
            self.assertEqual(
                output[:3], [
                    "# Model report for <unknown url> <unknown reference> <unknown commit>",
                    "",
                    "### Dump",
                ])
            self.assertNotIn("# Quality report", output)
            self.assertGreater(len(output), 100)
            output = "\n".join(output)
            data = _get_json_data(output)["javascript"]
            self.assertEqual(data["num_rules"], 1269)
            self.assertEqual(data["avg_rule_len"], 19.10401891252955)
            self.assertEqual(data["max_conf"], 0.9999756217002869)
            self.assertEqual(data["min_conf"], 0.19736842811107635)
            self.assertEqual(data["max_support"], 20528)
            self.assertEqual(data["min_support"], 16)
            lines = """|Min support|16|
|Max support|20528|
|Min confidence|0.19736842811107635|
|Max confidence|0.9999756217002869|""".splitlines()
            for line in lines:
                self.assertIn(line, output)
            num_rules, avg_len = _get_model_summary(output)
            self.assertEqual(num_rules, 1269)
            self.assertEqual(avg_len, 19.10401891252955)

    def test_eval(self):
        """Test on normal input."""
        input_pattern = os.path.join(self.jquery_dir, "**", "*")
        with Capturing() as output:
            print_reports(input_pattern=input_pattern, bblfsh=self.bblfsh,
                          language=self.language, model_path=self.model_path,
                          config={"uast_break_check": False})
        self.assertEqual([
            "# Quality report for javascript / <unknown url> <unknown reference> <unknown commit>",
            "",
            "### Classification report"],
            output[:3])
        qcount = output.count(
            "# Quality report for javascript / <unknown url> <unknown reference> <unknown commit>")
        self.assertEqual(qcount, 14)
        self.assertIn("### Summary", output)
        self.assertIn("# Model report for <unknown url> <unknown reference> <unknown commit>",
                      output)
        self.assertGreater(len(output), 100)
        self.assertIn("javascript", _get_json_data("\n".join(output)))

    def test_eval_aggregate(self):
        """Test on normal input, quality reports are aggregated."""
        input_pattern = os.path.join(self.jquery_dir, "**", "*")
        with Capturing() as output:
            print_reports(input_pattern=input_pattern, bblfsh=self.bblfsh,
                          language=self.language, model_path=self.model_path,
                          config={"uast_break_check": False, "aggregate": True})
        qcount = output.count(
            "# Quality report for javascript / <unknown url> <unknown reference> <unknown commit>")
        self.assertEqual(qcount, 1)
        output = "\n".join(output)
        output = output[:output.find("# Model report for <unknown url>")]
        metrics = _get_precision_recall_f1_support(output)
        self.assertEqual(
            metrics, (0.7221903953001138, 0.7723108245673567, 0.7246536747360008, 2947))

    def test_no_model(self):
        """Test on wrong path to model - expect fail."""
        with tempfile.TemporaryDirectory() as folder:
            input_pattern = os.path.join(folder, "**", "*")
            with tempfile.NamedTemporaryFile() as empty_model:
                with self.assertRaises(ValueError):
                    print_reports(
                        input_pattern=input_pattern, bblfsh=self.bblfsh, language=self.language,
                        model_path=empty_model, config={"uast_break_check": False})

    @unittest.skipUnless(os.getenv("LONG_TESTS", False),
                         "Time-consuming tests are skipped by default.")
    def test_train_review_analyzer_integration(self):
        """Integration test for review event."""
        with TestAnalyzer(port=self.port, db=self.db.name, fs=self.fs.name,
                          analyzer="lookout.style.format.benchmarks.general_report"):
            server.run("push", FROM_COMMIT, TO_COMMIT, port=self.port,
                       git_dir=self.jquery_dir, config_json=json.dumps({
                            QualityReportAnalyzer.name: get_train_config()}))
            server.run("review", FROM_COMMIT, TO_COMMIT, port=self.port,
                       git_dir=self.jquery_dir, config_json=json.dumps({
                            QualityReportAnalyzer.name: get_analyze_config()}))


if __name__ == "__main__":
    unittest.main()
