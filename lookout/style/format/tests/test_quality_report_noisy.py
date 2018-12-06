import os
from pathlib import Path
import re
import sys
import tarfile
import tempfile
import unittest

from lookout.style.format.quality_report_noisy import quality_report_noisy
from lookout.style.format.tests.test_quality_report import Capturing


class RobustnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bblfsh = "0.0.0.0:9432"
        cls.language = "javascript"

        cls.parent_loc = Path(__file__).parent.resolve()
        cls.base_dir_ = tempfile.TemporaryDirectory(dir=str(cls.parent_loc))
        cls.base_dir = cls.base_dir_.name

        with tarfile.open(str(cls.parent_loc / "jquery.tar.xz")) as tar:
            tar.extractall(path=cls.base_dir)
        cls.jquery_dir = os.path.join(cls.base_dir, "jquery")
        with tarfile.open(str(cls.parent_loc / "jquery_noisy.tar.xz")) as tar:
            tar.extractall(path=cls.base_dir)
        cls.jquery_noisy_dir = os.path.join(cls.base_dir, "jquery_noisy")
        cls.input_pattern = os.path.join(cls.jquery_dir, "**", "*.js")
        cls.input_pattern_noisy = os.path.join(cls.jquery_noisy_dir, "**", "*.js")
        cls.model_path = str(Path(__file__).parent.resolve() / "model_jquery.asdf")

    @classmethod
    def tearDownClass(cls):
        cls.base_dir_.cleanup()

    @unittest.skipIf(sys.version_info.minor == 5, "Python 3.5 is not yet supported by difflib")
    def test_quality_report_noisy(self):
        with Capturing() as output:
            try:
                quality_report_noisy(true_repo=self.input_pattern,
                                     noisy_repo=self.input_pattern_noisy,
                                     bblfsh=self.bblfsh,
                                     language=self.language,
                                     model_path=self.model_path,
                                     confidence_threshold=0.8,
                                     support_threshold=20,
                                     precision_threshold=0.95,
                                     dir_output=tempfile.tempdir)
            except SystemExit:
                self.skipTest("Matplotlib is required to run this test")
        pattern = re.compile(r"((?:recall x)|(?:precision y)): \[(\d+.\d+(, \d+.\d+)+)\]")
        metrics = {}
        for line in output:
            match = pattern.search(line)
            if match:
                metric, scores_string = list(match.groups())[:2]
                scores_string = scores_string.split(", ")
                scores = [float(f) for f in scores_string]
                metrics[metric] = scores
        self.assertGreater(metrics["recall x"][-1], 0)
        self.assertGreater(metrics["precision y"][-1], 0)


if __name__ == "__main__":
    unittest.main()
