import re
import sys
import tempfile
import unittest

from lookout.core import slogging

from lookout.style.format.benchmarks.quality_report_noisy import quality_report_noisy
from lookout.style.format.tests.test_analyzer import get_config
from lookout.style.format.tests.test_quality_report import Capturing


REPOSITORIES = """
https://github.com/warenlg/axios,75c8b3f146aaa8a71f7dca0263686fb1799f8f31,b5d60bb7aaa1b3ba0f286a5dad3028968831fd1d
""".strip()


class RobustnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bblfsh = "0.0.0.0:9432"
        cls.language = "javascript"

    @unittest.skipIf(sys.version_info.minor == 5, "Python 3.5 is not yet supported by difflib")
    def test_quality_report_noisy(self):
        slogging.setup("DEBUG", False)
        with Capturing() as output, tempfile.TemporaryDirectory() as dir_output:
            try:
                quality_report_noisy(bblfsh=self.bblfsh,
                                     language=self.language,
                                     confidence_threshold=0.8,
                                     support_threshold=20,
                                     precision_threshold=0.95,
                                     dir_output=dir_output,
                                     config=get_config(),
                                     repos=REPOSITORIES)
            except SystemExit:
                self.skipTest("Matplotlib is required to run this test")
        pattern = re.compile(r"((?:prediction rate x)|(?:precision y)): \[(\d+.\d+(, \d+.\d+)+)\]")
        metrics = {}
        for line in output:
            match = pattern.search(line)
            if match:
                metric, scores_string = list(match.groups())[:2]
                scores_string = scores_string.split(", ")
                scores = [float(f) for f in scores_string]
                metrics[metric] = scores
        self.assertGreater(metrics["prediction rate x"][-1], 0)
        self.assertGreater(metrics["precision y"][-1], 0)


if __name__ == "__main__":
    unittest.main()
