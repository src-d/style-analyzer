import glob
import io
import os
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest

from lookout.core.tests.server import find_port, run as launch_server
from lookout.style.format.quality_report import quality_report
from lookout.style.format.tests.test_analyzer_integration import (FROM_COMMIT, TestAnalyzer,
                                                                  TO_COMMIT)


class PretrainedModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Prepare environment & train the model for tests."""
        # required config
        cls.bblfsh = "0.0.0.0:9432"
        cls.language = "javascript"

        # analyzer
        cls.parent_loc = Path(__file__).parent.resolve()
        cls.base_dir_ = tempfile.TemporaryDirectory(dir=str(cls.parent_loc))
        cls.base_dir = cls.base_dir_.name
        cls.db = tempfile.NamedTemporaryFile(dir=cls.base_dir)
        cls.fs = tempfile.TemporaryDirectory(dir=cls.base_dir)
        cls.port = find_port()
        # extract repo
        cls.jquery_dir = os.path.join(cls.base_dir_.name, "jquery")
        # str() is needed for Python 3.5
        with tarfile.open(str(cls.parent_loc / "jquery.tar.xz")) as tar:
            tar.extractall(path=cls.base_dir)
        assert len(glob.glob(os.path.join(cls.jquery_dir, "**", "*"), recursive=True)) > 49
        with TestAnalyzer(port=cls.port, db=cls.db.name, fs=cls.fs.name):
            # train the rules
            launch_server(
                "push",
                FROM_COMMIT,
                TO_COMMIT,
                cls.port, git_dir=cls.jquery_dir)

            # find the saved model
            filenames = glob.glob(os.path.join(cls.fs.name, "**", "*"), recursive=True)
            res = [file for file in filenames if file.endswith(".asdf")]
            cls.model_path = res[0]

    @classmethod
    def tearDownClass(cls):
        """Remove temporary directory with jquery repository."""
        cls.base_dir_.cleanup()

    def setUp(self, fs=None):
        self.port = find_port()
        self.db = tempfile.NamedTemporaryFile(dir=self.base_dir)
        if fs is None:
            self.fs = tempfile.TemporaryDirectory(dir=self.base_dir)
        else:
            self.fs = fs

    def tearDown(self, fs_cleanup=True):
        if fs_cleanup:
            self.fs.cleanup()
        if hasattr(self, "test_analyzer"):
            self.test_analyzer.__exit__(None, None, None)


class Capturing(list):
    """Capture prints."""
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = io.StringIO()
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio    # free up some memory
        sys.stdout = self._stdout


@unittest.skipUnless(os.getenv("LONG_TESTS", False),
                     "Time-consuming tests are skipped by default.")
class QualityReportTests(PretrainedModelTests):
    def test_eval_empty_input(self):
        """Test on empty folder - expect fail."""
        with tempfile.TemporaryDirectory() as folder:
            input_pattern = os.path.join(folder, "**", "*")
            with Capturing() as output:
                quality_report(input_pattern=input_pattern, bblfsh=self.bblfsh,
                               language=self.language, n_files=10, model_path=self.model_path)
            self.assertIn("Failed to parse files, aborting report...", output)

    def test_eval(self):
        """Test on normal input."""
        input_pattern = os.path.join(self.jquery_dir, "**", "*")
        with Capturing() as output:
            quality_report(input_pattern=input_pattern, bblfsh=self.bblfsh,
                           language=self.language, n_files=10, model_path=self.model_path)
        self.assertNotIn("Failed to parse files, aborting report...", output)

    def test_no_model(self):
        """Test on wrong path to model - expect fail."""
        with tempfile.TemporaryDirectory() as folder:
            input_pattern = os.path.join(folder, "**", "*")
            with tempfile.NamedTemporaryFile() as empty_model:
                with self.assertRaises(ValueError):
                    quality_report(
                        input_pattern=input_pattern, bblfsh=self.bblfsh, language=self.language,
                        n_files=10, model_path=empty_model
                    )

    def test_push_ananlyzer(self):
        """Test push event to analyzer."""
        try:
            analyzer = "lookout.style.format.quality_report"
            config_json = '{\"style.format.analyzer.QualityReportAnalyzer\": ' \
                          '{\"model_path\": \"%s\", \"aggregate\":true}}' % self.model_path
            self.test_analyzer = TestAnalyzer(port=self.port, db=self.db.name, fs=self.fs.name,
                                              analyzer=analyzer)
            self.test_analyzer.__enter__()
            launch_server("push", FROM_COMMIT, TO_COMMIT, port=self.port,
                          git_dir=self.jquery_dir, config_json=config_json)
        except Exception as e:
            self.fail("Unexpected exception %s" % e)

    def test_review_ananlyzer(self):
        """Test review event to analyzer."""
        try:
            analyzer = "lookout.style.format.quality_report"
            config_json = '{\"style.format.analyzer.QualityReportAnalyzer\": ' \
                          '{\"model_path\": \"%s\", \"aggregate\":true}}' % self.model_path
            self.test_analyzer = TestAnalyzer(port=self.port, db=self.db.name, fs=self.fs.name,
                                              analyzer=analyzer)
            self.test_analyzer.__enter__()
            launch_server("review", FROM_COMMIT, TO_COMMIT, port=self.port,
                          git_dir=self.jquery_dir, config_json=config_json)
        except Exception as e:
            self.fail("Unexpected exception %s" % e)


if __name__ == "__main__":
    unittest.main()
