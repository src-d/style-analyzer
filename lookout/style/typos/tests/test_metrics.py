import pathlib
import unittest

import pandas

from lookout.style.typos.metrics import first_k_set, generate_report, get_scores, ScoreMode
from lookout.style.typos.utils import Columns


TEST_DATA_PATH = str(pathlib.Path(__file__).parent)


class MetricsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = pandas.DataFrame([["get", "get"],
                                     ["gwt", "get"],
                                     ["tokem", "token"],
                                     ["token", "token"]],
                                    columns=[Columns.Token, Columns.CorrectToken])
        cls.suggestions = {0: [("get", 1.0)],
                           1: [("got", 0.9), ("get", 0.5), ("gwt", 0.01)],
                           2: [("token", 0.98), ("taken", 0.3), ("tokem", 0.01)],
                           3: [("taken", 0.98), ("token", 0.9)]}

    def test_get_score(self):

        self.assertDictEqual(get_scores(self.data, self.suggestions, ScoreMode.detection),
                             {"accuracy": 0.75, "precision": 2 / 3, "recall": 1.0, "f1": 0.8})
        self.assertDictEqual(get_scores(self.data, self.suggestions, ScoreMode.correction, k=1),
                             {"accuracy": 0.5, "precision": 0.5, "recall": 0.5, "f1": 0.5})
        self.assertDictEqual(get_scores(self.data, self.suggestions, ScoreMode.on_corrected, k=2),
                             {"accuracy": 2 / 3, "precision": 2 / 3, "recall": 1.0, "f1": 0.8})

    def test_print_all_scores(self):
        report = generate_report(self.data, self.suggestions)
        print(report)
        self.assertEqual(len(report.split("\n")), 9)


if __name__ == "__main__":
    unittest.main()
