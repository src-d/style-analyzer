from os.path import join
import pathlib
import pickle
import unittest

import numpy
import pandas
from pandas.util.testing import assert_frame_equal

from lookout.style.typos.utils import (
    add_context_info, AFTER_COLUMN, BEFORE_COLUMN, CANDIDATE_COLUMN, filter_suggestions,
    flatten_data, ID_COLUMN, PROBABILITY_COLUMN, rank_candidates, SPLIT_COLUMN, SUGGESTIONS_COLUMN,
    suggestions_to_df, suggestions_to_flat_df, TYPO_COLUMN,
)

TEST_DATA_PATH = str(pathlib.Path(__file__).parent)


class DataTransformationsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = pandas.read_csv(join(TEST_DATA_PATH, "test_data.csv.xz"),
                                   index_col=0).infer_objects()
        cls.custom_data = pandas.DataFrame([[["get", "tokens", "num"]],
                                            [["use", "class"]]], columns=[SPLIT_COLUMN])
        cls.flat_custom_data = pandas.DataFrame([[["get", "tokens", "num"], "get"],
                                                 [["get", "tokens", "num"], "tokens"],
                                                 [["get", "tokens", "num"], "num"],
                                                 [["use", "class"], "use"],
                                                 [["use", "class"], "class"]],
                                                columns=[SPLIT_COLUMN, TYPO_COLUMN])

    def test_flatten_data(self):
        flat_data = pandas.read_csv(join(TEST_DATA_PATH, "test_flatten_data.csv.xz"),
                                    index_col=0).infer_objects()
        assert_frame_equal(flatten_data(self.data, "token"), flat_data)
        assert_frame_equal(flatten_data(self.custom_data, TYPO_COLUMN), self.flat_custom_data)

    def test_add_context_info(self):
        context_added = pandas.read_csv(join(TEST_DATA_PATH, "test_add_context_info.csv.xz"),
                                        index_col=0).infer_objects()
        context_added[AFTER_COLUMN] = pandas.eval(context_added[AFTER_COLUMN])
        context_added[BEFORE_COLUMN] = pandas.eval(context_added[BEFORE_COLUMN])
        assert_frame_equal(add_context_info(self.data.copy()), context_added)

        added_context_custom = pandas.DataFrame(
            [[["get", "tokens", "num"], "get", [], ["tokens", "num"]],
             [["get", "tokens", "num"], "tokens", ["get"], ["num"]],
             [["get", "tokens", "num"], "num", ["get", "tokens"], []],
             [["use", "class"], "use", [], ["class"]],
             [["use", "class"], "class", ["use"], []]],
            columns=[SPLIT_COLUMN, TYPO_COLUMN, BEFORE_COLUMN, AFTER_COLUMN])
        assert_frame_equal(add_context_info(self.flat_custom_data.copy()), added_context_custom)


class RankCandidatesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = pandas.read_csv(join(TEST_DATA_PATH, "test_data.csv.xz"),
                                   index_col=0).infer_objects()
        with open(join(TEST_DATA_PATH, "test_data_candidates_suggestions.pkl"), "br") as f:
            cls.suggestions = pickle.load(f)

        cls.custom_data = pandas.DataFrame([[["get", "tokens", "num"], "get"],
                                            [["gwt", "tokens"], "gwt"],
                                            [["get", "tokem"], "tokem"]],
                                           columns=[SPLIT_COLUMN, TYPO_COLUMN])
        cls.custom_candidates = pandas.DataFrame([[0, "get", "get"],
                                                  [1, "gwt", "get"],
                                                  [1, "gwt", "gpt"],
                                                  [2, "tokem", "tokem"],
                                                  [2, "tokem", "taken"],
                                                  [2, "tokem", "token"]],
                                                 columns=[ID_COLUMN, TYPO_COLUMN,
                                                          CANDIDATE_COLUMN])
        cls.custom_suggestions = {0: [("get", 1.0)],
                                  1: [("get", 0.9),
                                      ("gpt", 0.05)],
                                  2: [("token", 0.98),
                                      ("taken", 0.3),
                                      ("tokem", 0.01)]}
        cls.custom_filtered_suggestions = {1: [("get", 0.9),
                                               ("gpt", 0.05)],
                                           2: [("token", 0.98),
                                               ("taken", 0.3)]}

    def test_rank_candidates(self):
        candidates = pandas.read_csv(join(TEST_DATA_PATH, "test_data_candidates.csv.xz"),
                                     index_col=0).infer_objects()
        proba = numpy.load(join(TEST_DATA_PATH, "test_data_candidates_proba.pkl"))
        self.assertDictEqual(rank_candidates(candidates, proba, n_candidates=3), self.suggestions)

        proba = numpy.array([1.0, 0.9, 0.05, 0.01, 0.3, 0.98], dtype=float)
        self.assertDictEqual(rank_candidates(self.custom_candidates, proba, n_candidates=3),
                             self.custom_suggestions)
        self.assertDictEqual(rank_candidates(self.custom_candidates, proba, n_candidates=2,
                                             return_all=False),
                             self.custom_filtered_suggestions)

    def test_filter_suggestions(self):
        with open(join(TEST_DATA_PATH, "test_data_candidates_filtered_suggestions.pkl"),
                  "br") as f:
            filtered_suggestions = pickle.load(f)
        self.assertEqual(filter_suggestions(self.data, self.suggestions,
                                            n_candidates=2, return_all=False),
                         filtered_suggestions)

        self.assertEqual(filter_suggestions(self.custom_data, self.custom_suggestions,
                                            n_candidates=2, return_all=False),
                         self.custom_filtered_suggestions)

    def test_suggestions_to_df(self):
        suggestions_df = pandas.read_csv(join(TEST_DATA_PATH,
                                         "test_data_candidates_suggestions_df.csv.xz"),
                                         index_col=0).infer_objects()
        suggestions_df.suggestions = pandas.eval(suggestions_df.suggestions)
        assert_frame_equal(suggestions_to_df(self.data, self.suggestions), suggestions_df)

        custom_suggestions_df = pandas.DataFrame([[0, "get", [["get", 1.0]]],
                                                  [1, "gwt", [["get", 0.9],
                                                              ["gpt", 0.05]]],
                                                  [2, "tokem", [["token", 0.98],
                                                                ["taken", 0.3],
                                                                ["tokem", 0.01]]]],
                                                 columns=[ID_COLUMN, TYPO_COLUMN,
                                                          SUGGESTIONS_COLUMN],
                                                 index=[0, 1, 2])
        assert_frame_equal(suggestions_to_df(self.custom_data, self.custom_suggestions),
                           custom_suggestions_df)

    def test_suggestions_to_flat_df(self):
        suggestions_flat_df = pandas.read_csv(join(TEST_DATA_PATH,
                                              "test_data_candidates_suggestions_flat_df.csv.xz"),
                                              index_col=0).infer_objects()
        assert_frame_equal(suggestions_to_flat_df(self.data, self.suggestions),
                           suggestions_flat_df)

        custom_suggestions_flat_df = pandas.DataFrame([[0, "get", "get", 1.0],
                                                       [1, "gwt", "get", 0.9],
                                                       [1, "gwt", "gpt", 0.05],
                                                       [2, "tokem", "token", 0.98],
                                                       [2, "tokem", "taken", 0.3],
                                                       [2, "tokem", "tokem", 0.01]],
                                                      columns=[ID_COLUMN, TYPO_COLUMN,
                                                               CANDIDATE_COLUMN,
                                                               PROBABILITY_COLUMN])
        assert_frame_equal(suggestions_to_flat_df(self.custom_data, self.custom_suggestions),
                           custom_suggestions_flat_df)


if __name__ == "__main__":
    unittest.main()
