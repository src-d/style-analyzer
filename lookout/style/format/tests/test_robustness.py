import os
from pathlib import Path
import sys
import tarfile
import unittest

from lookout.style.format.robustness import style_robustness_report
from lookout.style.format.tests.test_quality_report import Capturing


class RobustnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bblfsh = "0.0.0.0:9432"
        cls.language = "javascript"

        cls.parent_loc = Path(__file__).parent.resolve()
        cls.base_dir = str(cls.parent_loc)

        with tarfile.open(str(cls.parent_loc / "jquery.tar.xz")) as tar:
            tar.extractall(path=cls.base_dir)
        cls.jquery_dir = str(Path(__file__).parent.resolve() / "jquery")
        with tarfile.open(str(cls.parent_loc / "jquery_noisy.tar.xz")) as tar:
            tar.extractall(path=cls.base_dir)
        cls.jquery_noisy_dir = str(Path(__file__).parent.resolve() / "jquery_noisy")

        cls.model_path = str(Path(__file__).parent.resolve() / "model_jquery5.asdf")

    @unittest.skipIf(sys.version_info.minor == 5, "Python 3.5 is not yet supported"
                                                  " by difflib")
    def test_style_robustness_report(self):
        input_pattern = os.path.join(self.jquery_dir, "**", "*.js")
        input_pattern_noisy = os.path.join(self.jquery_noisy_dir, "**", "*.js")
        with Capturing() as output:
            style_robustness_report(input_pattern, input_pattern_noisy, self.bblfsh,
                                    self.language, self.model_path)
        self.assertIn("precision: 1.0", output)
        self.assertIn("recall: 0.5", output)
        self.assertIn("F1 score: 0.667", output)

if __name__ == "__main__":
    unittest.main()
