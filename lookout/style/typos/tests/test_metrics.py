import pathlib
import unittest

import pandas

from lookout.style.typos.metrics import generate_report, get_scores, ScoreMode
from lookout.style.typos.utils import Candidate, Columns


TEST_DATA_PATH = str(pathlib.Path(__file__).parent)


class MetricsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = pandas.DataFrame([["get", "get"],
                                     ["gwt", "get"],
                                     ["tokem", "token"],
                                     ["token", "token"]],
                                    columns=[Columns.Token, Columns.CorrectToken])
        cls.suggestions = {0: [Candidate("get", 1.0)],
                           1: [Candidate("got", 0.9), Candidate("get", 0.5),
                               Candidate("gwt", 0.01)],
                           2: [Candidate("token", 0.98), Candidate("taken", 0.3),
                               Candidate("tokem", 0.01)],
                           3: [Candidate("taken", 0.98), Candidate("token", 0.9)]}

    def test_get_score(self):
        self.assertDictEqual(get_scores(self.data, self.suggestions, ScoreMode.detection),
                             {"accuracy": 0.75, "precision": 2 / 3, "recall": 1.0, "f1": 0.8})
        self.assertDictEqual(get_scores(self.data, self.suggestions, ScoreMode.correction, k=1),
                             {"accuracy": 0.5, "precision": 0.5, "recall": 0.5, "f1": 0.5})
        self.assertDictEqual(get_scores(self.data, self.suggestions, ScoreMode.on_typoed, k=2),
                             {"accuracy": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0})

    def test_print_all_scores(self):
        report = generate_report(self.data, self.suggestions)
        print(report)


if __name__ == "__main__":
    unittest.main()
