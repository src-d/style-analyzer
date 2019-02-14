import pathlib
import unittest

import pandas

from lookout.style.typos.metrics import Scores, get_score
from lookout.style.typos.utils import Columns


TEST_DATA_PATH = str(pathlib.Path(__file__).parent)


class ScoreClassTest(unittest.TestCase):
    def test_score_functions(self):
        scores = Scores(tp=12, fp=3, tn=3, fn=12)
        self.assertDictEqual(
            {"accuracy": 0.5, "precision": 0.8, "recall": 0.5, "f1": 2 / (1 / 0.8 + 1 / 0.5)},
            scores.get_metrics())


class MetricsTest(unittest.TestCase):
    def test_get_score(self):
        data = pandas.DataFrame([["get", "get"],
                                 ["gwt", "get"],
                                 ["tokem", "token"],
                                 ["token", "token"]],
                                columns=[Columns.Split, Columns.Token])
        suggestions = {0: [("get", 1.0)],
                       1: [("got", 0.9), ("get", 0.5), ("gwt", 0.01)],
                       2: [("token", 0.98), ("taken", 0.3), ("tokem", 0.01)],
                       3: [("taken", 0.98), ("token", 0.9)]}
        self.assertEqual(get_score(data, suggestions, mode="detection"),
                         Scores(2, 1, 1, 0))
        self.assertEqual(get_score(data, suggestions, mode="correction", k=1),
                         Scores(1, 1, 1, 1))
        self.assertEqual(get_score(data, suggestions, mode="on_corrected", k=2),
                         Scores(3, 0, 0, 0))


if __name__ == "__main__":
    unittest.main()
