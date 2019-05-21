from os.path import join
import pathlib
import pickle
import tempfile
import unittest

import numpy
import pandas
from pandas.util.testing import assert_frame_equal

from lookout.style.typos.utils import (
    add_context_info, Candidate, Columns, filter_splits, flatten_df_by_column, print_frequencies,
    rank_candidates, read_frequencies, read_vocabulary, suggestions_to_df, suggestions_to_flat_df)


TEST_DATA_PATH = str(pathlib.Path(__file__).parent)


class PreprocessingTest(unittest.TestCase):
    def test_filter_splits(self):
        data = pandas.DataFrame([["get value", "get"],
                                 ["get value", "value"],
                                 ["gut", "gut"],
                                 ["put tok", "put"],
                                 ["put tok", "tok"],
                                 ["put", "put"]], columns=[Columns.Split, Columns.Token])
        vocabulary = {"get", "value", "put"}
        result = pandas.DataFrame([["get value", "get"],
                                   ["get value", "value"],
                                   ["put", "put"]], columns=[Columns.Split, Columns.Token],
                                  index=[0, 1, 5])
        assert_frame_equal(filter_splits(data, vocabulary), result)

    def test_print_frequencies(self):
        vocabulary = {"get": 23, "set": 17}
        with tempfile.NamedTemporaryFile() as vocabulary_file:
            print_frequencies(vocabulary, vocabulary_file.name)
            with open(vocabulary_file.name) as f:
                lines = []
                for line in f:
                    lines.append(line.strip())
            self.assertEqual(len(lines), 2)
            self.assertIn("get,23", lines)
            self.assertIn("set,17", lines)


class ReadDataTest(unittest.TestCase):
    def test_read_functions(self):
        vocabulary = read_vocabulary(join(TEST_DATA_PATH, "test_frequencies.csv.xz"))
        frequencies = read_frequencies(join(TEST_DATA_PATH, "test_frequencies.csv.xz"))
        self.assertEqual(len(vocabulary), 100)
        self.assertSetEqual(set(vocabulary), set(frequencies.keys()))


class DataTransformationsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.custom_data = pandas.DataFrame([["get tokens num"],
                                            ["use class"]], columns=[Columns.Split])
        cls.flat_custom_data = pandas.DataFrame([["get tokens num", "get"],
                                                 ["get tokens num", "tokens"],
                                                 ["get tokens num", "num"],
                                                 ["use class", "use"],
                                                 ["use class", "class"]],
                                                columns=[Columns.Split, Columns.Token])
        cls.flat_data = pandas.read_csv(join(TEST_DATA_PATH, "test_flatten_data.csv.xz"),
                                        index_col=0, keep_default_na=False)

    def test_flatten_data(self):
        raw_data = pandas.read_csv(join(TEST_DATA_PATH, "raw_test_data.csv.xz"),
                                   index_col=0, keep_default_na=False)
        assert_frame_equal(flatten_df_by_column(raw_data, Columns.Split, Columns.Token, str.split),
                           self.flat_data)
        assert_frame_equal(flatten_df_by_column(self.custom_data, Columns.Split, Columns.Token,
                                                str.split), self.flat_custom_data)

    def test_add_context_info(self):
        context_added = pandas.read_csv(join(TEST_DATA_PATH, "test_add_context_info.csv.xz"),
                                        index_col=0, keep_default_na=False).fillna("")
        assert_frame_equal(add_context_info(self.flat_data), context_added)
        added_context_custom = pandas.DataFrame(
            [["get tokens num", "get", "", "tokens num"],
             ["get tokens num", "tokens", "get", "num"],
             ["get tokens num", "num", "get tokens", ""],
             ["use class", "use", "", "class"],
             ["use class", "class", "use", ""]],
            columns=[Columns.Split, Columns.Token, Columns.Before, Columns.After])
        assert_frame_equal(add_context_info(self.flat_custom_data.copy()), added_context_custom)


class RankCandidatesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = pandas.read_csv(join(TEST_DATA_PATH, "test_data.csv.xz"),
                                   index_col=0, keep_default_na=False)
        with open(join(TEST_DATA_PATH, "test_data_candidates_suggestions.pickle"), "br") as f:
            cls.suggestions = pickle.load(f)
            cls.suggestions = {i: [Candidate(*c) for c in candidates]
                               for i, candidates in cls.suggestions.items()}

        cls.custom_data = pandas.DataFrame([["get tokens num", "get"],
                                            ["gwt tokens", "gwt"],
                                            ["get tokem", "tokem"]],
                                           columns=[Columns.Split, Columns.Token])
        cls.custom_candidates = pandas.DataFrame([[0, "get", "get"],
                                                  [1, "gwt", "get"],
                                                  [1, "gwt", "gpt"],
                                                  [2, "tokem", "tokem"],
                                                  [2, "tokem", "taken"],
                                                  [2, "tokem", "token"]],
                                                 columns=[Columns.Id, Columns.Token,
                                                          Columns.Candidate])
        cls.custom_suggestions = {0: [Candidate("get", 1.0)],
                                  1: [Candidate("get", 0.9),
                                      Candidate("gpt", 0.05)],
                                  2: [Candidate("token", 0.98),
                                      Candidate("taken", 0.3),
                                      Candidate("tokem", 0.01)]}
        cls.custom_filtered_suggestions = {1: [Candidate("get", 0.9),
                                               Candidate("gpt", 0.05)],
                                           2: [Candidate("token", 0.98),
                                               Candidate("taken", 0.3)]}

    def test_rank_candidates(self):
        candidates = pandas.read_csv(join(TEST_DATA_PATH, "test_data_candidates.csv.xz"),
                                     index_col=0, keep_default_na=False)
        with open(join(TEST_DATA_PATH, "test_data_candidates_proba.pickle"), "rb") as fin:
            proba = pickle.load(fin)
        self.assertDictEqual(rank_candidates(candidates, proba, n_candidates=3), self.suggestions)

        proba = numpy.array([1.0, 0.9, 0.05, 0.01, 0.3, 0.98], dtype=float)
        self.assertDictEqual(rank_candidates(self.custom_candidates, proba, n_candidates=3),
                             self.custom_suggestions)
        self.assertDictEqual(rank_candidates(self.custom_candidates, proba, n_candidates=2,
                                             return_all=False),
                             self.custom_filtered_suggestions)

    def test_suggestions_to_df(self):
        suggestions_df = pandas.read_csv(join(TEST_DATA_PATH,
                                         "test_data_candidates_suggestions_df.csv.xz"),
                                         index_col=0, keep_default_na=False)
        suggestions_df.suggestions = pandas.eval(suggestions_df.suggestions)
        assert_frame_equal(suggestions_to_df(self.data, self.suggestions), suggestions_df)

        custom_suggestions_df = pandas.DataFrame([[0, "get", [["get", 1.0]]],
                                                  [1, "gwt", [["get", 0.9],
                                                              ["gpt", 0.05]]],
                                                  [2, "tokem", [["token", 0.98],
                                                                ["taken", 0.3],
                                                                ["tokem", 0.01]]]],
                                                 columns=[Columns.Id, Columns.Token,
                                                          Columns.Suggestions],
                                                 index=[0, 1, 2])
        assert_frame_equal(suggestions_to_df(self.custom_data, self.custom_suggestions),
                           custom_suggestions_df)

    def test_suggestions_to_flat_df(self):
        suggestions_flat_df = pandas.read_csv(join(TEST_DATA_PATH,
                                              "test_data_candidates_suggestions_flat_df.csv.xz"),
                                              index_col=0, keep_default_na=False)
        assert_frame_equal(suggestions_to_flat_df(self.data, self.suggestions),
                           suggestions_flat_df)

        custom_suggestions_flat_df = pandas.DataFrame([[0, "get", "get", 1.0],
                                                       [1, "gwt", "get", 0.9],
                                                       [1, "gwt", "gpt", 0.05],
                                                       [2, "tokem", "token", 0.98],
                                                       [2, "tokem", "taken", 0.3],
                                                       [2, "tokem", "tokem", 0.01]],
                                                      columns=[Columns.Id, Columns.Token,
                                                               Columns.Candidate,
                                                               Columns.Probability])
        assert_frame_equal(suggestions_to_flat_df(self.custom_data, self.custom_suggestions),
                           custom_suggestions_flat_df)


if __name__ == "__main__":
    unittest.main()
