import io
import os
import glob
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
import unittest

from lookout.core.tests.server import run as launch_server, find_port
from lookout.style.format.tests.test_analyzer_integration import TestAnalyzer
from lookout.style.format.quality_report import quality_report


class PretrainedModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Prepare environment & train the model for tests."""
        # required config
        cls.bblfsh = "0.0.0.0:9432"
        cls.language = "javascript"

        # analyzer
        cls.parent_loc = Path(__file__).parent.resolve()
        cls.base_dir = str(cls.parent_loc)
        cls.db = tempfile.NamedTemporaryFile(dir=cls.base_dir)
        cls.fs = tempfile.TemporaryDirectory(dir=cls.base_dir)
        cls.port = find_port()
        cls.analyzer = TestAnalyzer(port=cls.port, db=cls.db.name, fs=cls.fs.name).__enter__()

        # extract repo
        cls.jquery_dir = str(Path(__file__).parent.resolve() / "jquery")
        # str() is needed for Python 3.5
        with tarfile.open(str(cls.parent_loc / "jquery.tar.xz")) as tar:
            tar.extractall(path=cls.base_dir)

        # train the rules
        launch_server(
            "push",
            "fbd32214d1b08e09278f39c77979658cabde6c4d",
            "931442679ad04ebadad2b70a9f938fb0bc64d537",
            cls.port, git_dir=cls.jquery_dir)

        # find the saved model
        filenames = glob.glob(os.path.join(cls.fs.name, "**", "*"), recursive=True)
        res = [file for file in filenames if file.endswith(".asdf")]
        cls.model_path = res[0]

    @classmethod
    def tearDownClass(cls):
        """Remove temporary directory with jquery repository."""
        shutil.rmtree(cls.jquery_dir)
        cls.fs.cleanup()
        cls.analyzer.__exit__(None, None, None)


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


if __name__ == "__main__":
    unittest.main()
