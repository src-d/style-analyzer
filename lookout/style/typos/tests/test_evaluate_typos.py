import pathlib
import unittest

from lookout.style.typos.benchmarks.evaluate_typos import evaluate_typos_on_identifiers


TEST_DATA_PATH = pathlib.Path(__file__).parent


class TyposCorrectorTest(unittest.TestCase):
    def test_evaluate_typos_on_identifiers(self):
        config = {"processes_number": 8, "corrector": str(TEST_DATA_PATH / "test_corrector.asdf")}
        s = evaluate_typos_on_identifiers(str(TEST_DATA_PATH / "test_commits_with_typo.csv.xz"),
                                          config)
        self.assertGreater(len(s), 0)
