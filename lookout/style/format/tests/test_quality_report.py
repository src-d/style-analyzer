import glob
import io
import os
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest

from lookout.core.helpers.analyzer_context_manager import AnalyzerContextManager
from modelforge import slogging
from numpy.testing import assert_almost_equal

from lookout.style.format.benchmarks.general_report import print_reports, QualityReportAnalyzer
from lookout.style.format.benchmarks.quality_report import _get_json_data, _get_metrics, \
    _get_model_summary
from lookout.style.format.tests import long_test
from lookout.style.format.tests.test_analyzer import get_config
from lookout.style.format.tests.test_analyzer_integration import FROM_COMMIT, TO_COMMIT


class PretrainedModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Prepare environment & train the model for tests."""
        slogging.setup("DEBUG", False)
        # required config
        cls.bblfsh = "0.0.0.0:9432"
        cls.language = "javascript"

        # analyzer
        parent_loc = Path(__file__).parent.resolve()
        cls.base_dir_ = tempfile.TemporaryDirectory()
        cls.base_dir = cls.base_dir_.name
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
        """Test on empty folder - expect only model and test report."""
        config = {"analyze": {"language_defaults": {"uast_break_check": False}},
                  "aggregate": True}
        with tempfile.TemporaryDirectory() as folder:
            input_pattern = os.path.join(folder, "**", "*")
            with Capturing() as output:
                print_reports(input_pattern=input_pattern, bblfsh_addr=self.bblfsh,
                              language=self.language, model_path=self.model_path, config=config)
            self.assertEqual(
                output[:3], [
                    "# Model report for file:///var/folders/kw/93jybvs16_954hytgsq6ld7r0000gn/T/"
                    "top-repos-quality-repos-jigt1n8g/jquery HEAD "
                    "dae5f3ce3d2df27873d01f0d9682f6a91ad66b87",
                    "",
                    "### Dump",
                ])
            self.assertGreater(len(output), 100)
            output = "\n".join(output)
            self.assertNotIn("# Train report", output)
            test_report_start = output.find("Test report")
            self.assertNotEqual(test_report_start, -1)
            output = output[:test_report_start]
            model_data = _get_json_data(output)["javascript"]
            self.assertEqual(model_data, {
                "avg_rule_len": 11.214128035320089,
                "max_conf": 0.9999598264694214,
                "max_support": 21880,
                "min_conf": 0.9206641912460327,
                "min_support": 81,
                "num_rules": 453,
            })
            lines = ["|Min support|81|",
                     "|Max support|21880|",
                     "|Min confidence|0.9206641912460327|",
                     "|Max confidence|0.9999598264694214|"]
            for line in lines:
                self.assertIn(line, output)
            num_rules, avg_len = _get_model_summary(output)
            self.assertEqual(num_rules, 453)
            self.assertAlmostEqual(avg_len, 11.214128035320089)

    def test_eval(self):
        """Test on normal input."""
        q_report_header_train = "# Train report for javascript"
        q_report_header_test = "# Test report for javascript"
        input_pattern = os.path.join(self.jquery_dir, "**", "*")
        with Capturing() as output:
            print_reports(input_pattern=input_pattern, bblfsh_addr=self.bblfsh,
                          language=self.language, model_path=self.model_path,
                          config={"analyze": {"language_defaults": {"uast_break_check": False}}})
        self.assertIn(q_report_header_train, output[0])
        self.assertIn("### Classification report", output)
        self.assertGreater(len(output), 100)
        output = "\n".join(output)
        test_report_start = output.find("Test report")
        self.assertNotEqual(test_report_start, -1)
        qcount = output.count(q_report_header_train)
        self.assertEqual(qcount, 14)
        qcount = output.count(q_report_header_test)
        self.assertEqual(qcount, 14)

    def test_eval_aggregate(self):
        """Test on normal input, quality reports are aggregated."""
        q_report_header = "# Train report for javascript"
        input_pattern = os.path.join(self.jquery_dir, "**", "*")
        with Capturing() as output:
            print_reports(
                input_pattern=input_pattern, bblfsh_addr=self.bblfsh, language=self.language,
                config={"analyze": {"language_defaults": {"uast_break_check": False}},
                        "aggregate": True},
                model_path=self.model_path)
        output = "\n".join(output)
        qcount = output.count(q_report_header)
        self.assertEqual(qcount, 1)
        output = output[output.find("# Train report"):output.find("# Test report")]
        metrics = _get_metrics(output)
        expected_metrics = (0.9305755395683454, 0.9305755395683454,
                            0.8535136918508743, 0.9305755395683454,
                            0.8903803131991053, 0.9171890465193006,
                            2780, 3031)
        assert_almost_equal(metrics, expected_metrics, decimal=15)

    def test_no_model(self):
        """Test on wrong path to model - expect fail."""
        with tempfile.TemporaryDirectory() as folder:
            input_pattern = os.path.join(folder, "**", "*")
            with tempfile.NamedTemporaryFile() as empty_model:
                with self.assertRaises(ValueError):
                    print_reports(
                        input_pattern=input_pattern, bblfsh_addr=self.bblfsh,
                        language=self.language, model_path=empty_model,
                        config={"uast_break_check": False})

    @long_test
    def test_train_review_analyzer_integration(self):
        """Integration test for review event."""
        with AnalyzerContextManager(analyzer=QualityReportAnalyzer,
                                    db=self.db.name, fs=self.fs.name) as context:
            context.review(FROM_COMMIT, TO_COMMIT, git_dir=self.jquery_dir, config_json={
                           QualityReportAnalyzer.name: get_config()})


if __name__ == "__main__":
    unittest.main()
