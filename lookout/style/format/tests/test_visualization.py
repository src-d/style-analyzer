import os
import tempfile
import unittest

from lookout.style.format.tests.test_quality_report import Capturing, PretrainedModelTests
from lookout.style.format.visualization import visualize


@unittest.skipUnless(os.getenv("LONG_TESTS", False),
                     "Time-consuming tests are skipped by default.")
class VisualizationTests(PretrainedModelTests):
    def test_eval_empty_input(self):
        """Test on empty path - expect fail."""
        with tempfile.TemporaryDirectory() as folder:
            input_filename = os.path.join(folder, "unexisted_file.js")
            with self.assertRaises(AssertionError):
                visualize(input_filename=input_filename, bblfsh=self.bblfsh,
                          language=self.language, model_path=self.model_path)

    def test_eval_wrong_input(self):
        """Test on wrong file - expect fail."""
        input_filename = os.path.join(self.jquery_dir, ".git/hooks/pre-commit.sample")
        with self.assertRaises(AssertionError):
            visualize(input_filename=input_filename, bblfsh=self.bblfsh,
                      language=self.language, model_path=self.model_path)

    def test_eval(self):
        """Test on normal input."""
        input_filename = os.path.join(self.jquery_dir, "7-node_smoke_tests.js")
        with Capturing() as output:
            visualize(input_filename=input_filename, bblfsh=self.bblfsh,
                      language=self.language, model_path=self.model_path)
        self.assertNotIn("Failed to parse files, aborting report...", output)

    def test_no_model(self):
        """Test on wrong path to model - expect fail."""
        input_filename = os.path.join(self.jquery_dir, "7-node_smoke_tests.js")
        with tempfile.NamedTemporaryFile() as empty_model:
            with self.assertRaises(ValueError):
                visualize(
                    input_filename=input_filename, bblfsh=self.bblfsh, language=self.language,
                    model_path=empty_model
                )


if __name__ == "__main__":
    unittest.main()
