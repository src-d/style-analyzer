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

from lookout.style.format.quality_report import quality_report, QualityReportAnalyzer
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
        cls.base_dir_ = tempfile.TemporaryDirectory(dir=str(parent_loc))
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
        """Test on empty folder - expect fail."""
        with tempfile.TemporaryDirectory() as folder:
            input_pattern = os.path.join(folder, "**", "*")
            with Capturing() as output:
                quality_report(input_pattern=input_pattern, bblfsh=self.bblfsh,
                               language=self.language, model_path=self.model_path,
                               config={"uast_break_check": False})
            self.assertEqual(output[:4],
                             ["", "# Model report for javascript", "", "### Rules summary:"])
            self.assertNotIn("# Quality report", output)

    def test_eval(self):
        """Test on normal input."""
        input_pattern = os.path.join(self.jquery_dir, "**", "*")
        with Capturing() as output:
            quality_report(input_pattern=input_pattern, bblfsh=self.bblfsh,
                           language=self.language, model_path=self.model_path,
                           config={"uast_break_check": False})
        self.assertEqual(["", "# Quality report", "", "### Classification report:"], output[:4])
        self.assertIn("### Rules summary:", output)

    def test_no_model(self):
        """Test on wrong path to model - expect fail."""
        with tempfile.TemporaryDirectory() as folder:
            input_pattern = os.path.join(folder, "**", "*")
            with tempfile.NamedTemporaryFile() as empty_model:
                with self.assertRaises(ValueError):
                    quality_report(
                        input_pattern=input_pattern, bblfsh=self.bblfsh, language=self.language,
                        model_path=empty_model, config={"uast_break_check": False})

    @unittest.skipUnless(os.getenv("LONG_TESTS", False),
                         "Time-consuming tests are skipped by default.")
    def test_train_review_analyzer_integration(self):
        """Integration test for review event."""
        with TestAnalyzer(port=self.port, db=self.db.name, fs=self.fs.name,
                          analyzer="lookout.style.format.quality_report"):
            server.run("push", FROM_COMMIT, TO_COMMIT, port=self.port,
                       git_dir=self.jquery_dir)
            server.run("review", FROM_COMMIT, TO_COMMIT, port=self.port,
                       git_dir=self.jquery_dir, config_json=json.dumps({
                            QualityReportAnalyzer.name: {"uast_break_check": False}}))


if __name__ == "__main__":
    unittest.main()
