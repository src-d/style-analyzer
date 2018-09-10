import unittest
import pickle

import pandas
import numpy
from pandas.util.testing import assert_frame_equal

from lookout.style.typos.utils import (flatten_data, add_context_info, rank_candidates, filter_suggestions,
                   suggestions_to_df, suggestions_to_flat_df)

TEST_DATA_PATH = "lookout/style/typos/tests"


class DataTransformationsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = pandas.read_csv(TEST_DATA_PATH + "test_data.csv", index_col=0)
        cls.custom_data = pandas.DataFrame([[["get", "tokens", "num"]],
                                            [["use", "class"]]], columns=["token_split"])
        cls.flat_custom_data = pandas.DataFrame([[["get", "tokens", "num"], "get"],
                                                 [["get", "tokens", "num"], "tokens"],
                                                 [["get", "tokens", "num"], "num"],
                                                 [["use", "class"], "use"],
                                                 [["use", "class"], "class"]],
                                                columns=["token_split", "typo"])

    def test_flatten_data(self):
        flat_data = pandas.read_csv(TEST_DATA_PATH + "test_flatten_data.csv",
                                    index_col=0).infer_objects()
        assert_frame_equal(flatten_data(self.data, "token"), flat_data)
        assert_frame_equal(flatten_data(self.custom_data, "typo"), self.flat_custom_data)

    def test_add_context_info(self):
        context_added = pandas.read_csv(TEST_DATA_PATH + "test_add_context_info.csv",
                                        index_col=0).infer_objects()
        context_added["after"] = pandas.eval(context_added["after"])
        context_added["before"] = pandas.eval(context_added["before"])
        assert_frame_equal(add_context_info(self.data.copy()), context_added)

        added_context_custom = pandas.DataFrame(
            [[["get", "tokens", "num"], "get", [], ["tokens", "num"]],
             [["get", "tokens", "num"], "tokens", ["get"], ["num"]],
             [["get", "tokens", "num"], "num", ["get", "tokens"], []],
             [["use", "class"], "use", [], ["class"]],
             [["use", "class"], "class", ["use"], []]],
            columns=["token_split", "typo", "before", "after"])
        assert_frame_equal(add_context_info(self.flat_custom_data.copy()), added_context_custom)


class RankCandidatesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = pandas.read_csv(TEST_DATA_PATH + "test_data.csv", index_col=0)
        with open(TEST_DATA_PATH + "test_data_candidates_suggestions.pkl", "br") as f:
            cls.suggestions = pickle.load(f)

        cls.custom_data = pandas.DataFrame([[["get", "tokens", "num"], "get"],
                                            [["gwt", "tokens"], "gwt"],
                                            [["get", "tokem"], "tokem"]],
                                           columns=["token_split", "typo"])
        cls.custom_candidates = pandas.DataFrame([[0, "get", "get"],
                                                  [1, "gwt", "get"],
                                                  [1, "gwt", "gpt"],
                                                  [2, "tokem", "tokem"],
                                                  [2, "tokem", "taken"],
                                                  [2, "tokem", "token"]],
                                                 columns=["id", "typo", "candidate"])
        cls.custom_suggestions = {0: [["get", 1.0]],
                                  1: [["get", 0.9],
                                      ["gpt", 0.05]],
                                  2: [["token", 0.98],
                                      ["taken", 0.3],
                                      ["tokem", 0.01]]}
        cls.custom_filtered_suggestions = {1: [["get", 0.9],
                                               ["gpt", 0.05]],
                                           2: [["token", 0.98],
                                               ["taken", 0.3]]}

    def test_rank_candidates(self):
        candidates = pandas.read_csv(TEST_DATA_PATH + "test_data_candidates.csv")
        proba = numpy.load(TEST_DATA_PATH + "test_data_candidates_proba.pkl")
        self.assertEqual(rank_candidates(candidates, proba, n_candidates=3), self.suggestions)

        proba = numpy.array([1.0, 0.9, 0.05, 0.01, 0.3, 0.98], dtype=float)
        self.assertEqual(rank_candidates(self.custom_candidates, proba, n_candidates=3),
                         self.custom_suggestions)
        self.assertEqual(rank_candidates(self.custom_candidates, proba, n_candidates=2,
                                         return_all=False),
                         self.custom_filtered_suggestions)

    def test_filter_suggestions(self):
        with open(TEST_DATA_PATH + "test_data_candidates_filtered_suggestions.pkl", "br") as f:
            filtered_suggestions = pickle.load(f)
        self.assertEqual(filter_suggestions(self.data, self.suggestions,
                                            n_candidates=2, return_all=False),
                         filtered_suggestions)

        self.assertEqual(filter_suggestions(self.custom_data, self.custom_suggestions,
                                            n_candidates=2, return_all=False),
                         self.custom_filtered_suggestions)

    def test_suggestions_to_df(self):
        suggestions_df = pandas.read_csv(TEST_DATA_PATH +
                                         "test_data_candidates_suggestions_df.csv",
                                         index_col=0)
        suggestions_df.suggestions = pandas.eval(suggestions_df.suggestions)
        assert_frame_equal(suggestions_to_df(self.data, self.suggestions), suggestions_df)

        custom_suggestions_df = pandas.DataFrame([[0, "get", [["get", 1.0]]],
                                                  [1, "gwt", [["get", 0.9],
                                                              ["gpt", 0.05]]],
                                                  [2, "tokem", [["token", 0.98],
                                                                ["taken", 0.3],
                                                                ["tokem", 0.01]]]],
                                                 columns=["id", "typo", "suggestions"],
                                                 index=[0, 1, 2])
        assert_frame_equal(suggestions_to_df(self.custom_data, self.custom_suggestions),
                           custom_suggestions_df)

    def test_suggestions_to_flat_df(self):
        suggestions_flat_df = pandas.read_csv(TEST_DATA_PATH +
                                              "test_data_candidates_suggestions_flat_df.csv",
                                              index_col=0)
        assert_frame_equal(suggestions_to_flat_df(self.data, self.suggestions),
                           suggestions_flat_df)

        custom_suggestions_flat_df = pandas.DataFrame([[0, "get", "get", 1.0],
                                                       [1, "gwt", "get", 0.9],
                                                       [1, "gwt", "gpt", 0.05],
                                                       [2, "tokem", "token", 0.98],
                                                       [2, "tokem", "taken", 0.3],
                                                       [2, "tokem", "tokem", 0.01]],
                                                      columns=["id", "typo", "candidate",
                                                               "proba"])
        assert_frame_equal(suggestions_to_flat_df(self.custom_data, self.custom_suggestions),
                           custom_suggestions_flat_df)


if __name__ == "__main__":
    unittest.main()
