import os
from pathlib import Path
import re
import sys
import tarfile
import tempfile
import unittest

from lookout.core import slogging

from lookout.style.format.benchmarks.quality_report_noisy import quality_report_noisy
from lookout.style.format.tests.test_quality_report import Capturing


class RobustnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bblfsh = "0.0.0.0:9432"
        cls.language = "javascript"

        parent_loc = Path(__file__).parent.resolve()
        cls.base_dir_ = tempfile.TemporaryDirectory()
        cls.base_dir = cls.base_dir_.name
        with tarfile.open(str(parent_loc / "jquery_noisy.tar.xz")) as tar:
            tar.extractall(path=cls.base_dir)
        cls.jquery_dir = str(Path(cls.base_dir) / "jquery_noisy" / "jquery")
        os.symlink(str(parent_loc / "jquery_noisy.asdf"), str(Path(cls.jquery_dir) / "model.asdf"))

    @unittest.skipIf(sys.version_info.minor == 5, "Python 3.5 is not yet supported by difflib")
    def test_quality_report_noisy(self):
        slogging.setup("DEBUG", False)
        with Capturing() as output:
            try:
                quality_report_noisy(bblfsh=self.bblfsh,
                                     language=self.language,
                                     confidence_threshold=0.8,
                                     support_threshold=20,
                                     precision_threshold=0.95,
                                     dir_output=tempfile.tempdir,
                                     repos=self.jquery_dir)
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
