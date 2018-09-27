import unittest

import pandas

from lookout.style.typos.model import IdentifiersTyposModel, NODE_ID_COLUMN
from lookout.style.typos.utils import SPLIT_COLUMN, TYPO_COLUMN


class IdentifiersTyposModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.checker = IdentifiersTyposModel(confidence_threshold=0.2, n_candidates=3)
        cls.identifiers = ["get", "gpt_tokeb"]
        cls.test_df = pandas.DataFrame([[0, "get", "get"],
                                        [1, "gpt tokeb", "gpt"],
                                        [1, "gpt tokeb", "tokeb"]],
                                       columns=[NODE_ID_COLUMN, SPLIT_COLUMN, TYPO_COLUMN])
        cls.suggestions = {1: [("get", 0.9),
                               ("gpt", 0.3)],
                           2: [("token", 0.98),
                               ("taken", 0.3),
                               ("tokem", 0.01)]}
        cls.filtered_suggestions = {1: [("get", 0.9)],
                                    2: [("token", 0.98),
                                        ("taken", 0.3)]}

    def test_filter_suggestions(self):
        self.assertDictEqual(self.checker.filter_suggestions(self.test_df, self.suggestions),
                             self.filtered_suggestions)

    def test_group_by_node_id(self):
        grouped_suggestions = {1: {"gpt": [("get", 0.9)],
                                   "tokeb": [("token", 0.98), ("taken", 0.3)]}}
        self.assertDictEqual(self.checker.group_by_node_id(self.test_df,
                                                           self.filtered_suggestions),
                             grouped_suggestions)

    def test_check_identifiers(self):
        suggestions = self.checker.check_identifiers(self.identifiers)
        self.assertTrue(set(suggestions.keys()).issubset(set(range(len(self.identifiers)))))


if __name__ == "__main__":
    unittest.main()
