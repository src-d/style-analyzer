import os
import tempfile
import unittest

from lookout.core.test_helpers.server import find_port, run as launch_server

from lookout.style.format.rule_stat import print_rules_report
from lookout.style.format.tests.test_analyzer_integration import (
    FROM_COMMIT, TestAnalyzer, TO_COMMIT)
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
                        model_path=empty_model,
                    )

    def test_push_ananlyzer(self):
        """Test push event to analyzer."""
        try:
            db = tempfile.NamedTemporaryFile(dir=self.base_dir)
            fs = tempfile.TemporaryDirectory(dir=self.base_dir)
            port = find_port()
            analyzer = "lookout.style.format.rule_stat"
            config_json = '{\"style.format.analyzer.RuleStatAnalyzer\": ' \
                          '{\"model_path\": \"%s\", \"aggregate\":true}}' % self.model_path
            self.test_analyzer = TestAnalyzer(port=port, db=db.name, fs=fs.name, analyzer=analyzer)
            self.test_analyzer.__enter__()
            launch_server("push", FROM_COMMIT, TO_COMMIT, port=port, git_dir=self.jquery_dir,
                          config_json=config_json)
        except Exception as e:
            self.fail("Unexpected exception %s" % e)

    def test_review_ananlyzer(self):
        """Test review event to analyzer."""
        try:
            db = tempfile.NamedTemporaryFile(dir=self.base_dir)
            fs = tempfile.TemporaryDirectory(dir=self.base_dir)
            port = find_port()
            analyzer = "lookout.style.format.rule_stat"
            config_json = '{\"style.format.analyzer.RuleStatAnalyzer\": ' \
                          '{\"model_path\": \"%s\", \"aggregate\":true}}' % self.model_path
            self.test_analyzer = TestAnalyzer(port=port, db=db.name, fs=fs.name, analyzer=analyzer)
            self.test_analyzer.__enter__()
            launch_server("review", FROM_COMMIT, TO_COMMIT, port=port, git_dir=self.jquery_dir,
                          config_json=config_json)
            self.test_analyzer.__exit__(None, None, None)
        except Exception as e:
            self.fail("Unexpected exception %s" % e)


if __name__ == "__main__":
    unittest.main()
