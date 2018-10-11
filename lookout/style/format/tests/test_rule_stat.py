import os
import tempfile
import unittest

from lookout.style.format.rule_stat import print_rules_report
from lookout.style.format.tests.test_quality_report import Capturing, PretrainedModelTests


@unittest.skipUnless(os.getenv("LONG_TESTS", False),
                     "Time-consuming tests are skipped by default.")
class RuleStatTests(PretrainedModelTests):
    def test_eval_empty_input(self):
        """Test on empty folder - expect fail."""
        with tempfile.TemporaryDirectory() as folder:
            input_pattern = os.path.join(folder, "**", "*")
            with Capturing() as output:
                print_rules_report(input_pattern=input_pattern, bblfsh=self.bblfsh,
                                   language=self.language, model_path=self.model_path)
            self.assertIn("Failed to parse files, aborting report...", output)

    def test_eval(self):
        """Test on normal input."""
        input_pattern = os.path.join(self.jquery_dir, "**", "*")
        with Capturing() as output:
            print_rules_report(input_pattern=input_pattern, bblfsh=self.bblfsh,
                               language=self.language, model_path=self.model_path)
        self.assertNotIn("Failed to parse files, aborting report...", output)

    def test_no_model(self):
        """Test on wrong path to model - expect fail."""
        with tempfile.TemporaryDirectory() as folder:
            input_pattern = os.path.join(folder, "**", "*")
            with tempfile.NamedTemporaryFile() as empty_model:
                with self.assertRaises(ValueError):
                    print_rules_report(
                        input_pattern=input_pattern, bblfsh=self.bblfsh, language=self.language,
                        model_path=empty_model
                    )


if __name__ == "__main__":
    unittest.main()
