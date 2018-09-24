from os.path import join
import pathlib
import unittest

import numpy
from numpy.testing import assert_array_equal
import pandas
from pandas.util.testing import assert_frame_equal

from lookout.style.typos.generation import (get_candidates_tokens, get_candidates_features,
                                            CandidatesGenerator)
from lookout.style.typos.utils import (CANDIDATE_COLUMN, CORRECT_TOKEN_COLUMN,
                                       ID_COLUMN, SPLIT_COLUMN, TYPO_COLUMN)


TEST_DATA_PATH = pathlib.Path(__file__).parent
FASTTEXT_DUMP_FILE = str(TEST_DATA_PATH / "id_vecs_10.bin")
VOCABULARY_FILE = str(TEST_DATA_PATH / "test_frequencies.csv.xz")
FREQUENCIES_FILE = str(TEST_DATA_PATH / "frequencies.csv")


class CandidatesSplitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.candidates = pandas.DataFrame([[0, "gut", "get", 0.1, 0.2],
                                           [1, "cit", "cut", 0.8, 0.3],
                                           [1, "cit", "cit", 0.1, 0.5]],
                                          columns=[ID_COLUMN, TYPO_COLUMN,
                                                   CANDIDATE_COLUMN, 0, 1])

    def test_get_candidates_tokens(self):
        tokens = pandas.DataFrame([[0, "gut", "get"],
                                   [1, "cit", "cut"],
                                   [1, "cit", "cit"]],
                                  columns=[ID_COLUMN, TYPO_COLUMN, CANDIDATE_COLUMN])
        assert_frame_equal(get_candidates_tokens(self.candidates), tokens)

    def test_get_candidates_features(self):
        features = numpy.array([[0.1, 0.2],
                                [0.8, 0.3],
                                [0.1, 0.5]])
        assert_array_equal(get_candidates_features(self.candidates), features)


class GeneratorTest(unittest.TestCase):
    def test_generate_candidates(self):
        generator = CandidatesGenerator()
        generator.construct(VOCABULARY_FILE, FREQUENCIES_FILE, FASTTEXT_DUMP_FILE,
                            neighbors_number=3, taken_for_distance=3, max_distance=3, radius=3)

        data = pandas.read_csv(str(TEST_DATA_PATH / "test_data.csv"),
                               index_col=0).infer_objects()
        custom_data = pandas.DataFrame([[["get", "tokens", "num"], "tokens", "tokens"],
                                        [["gwt", "tokens"], "gwt", "get"],
                                        [["get", "tokem"], "tokem", "token"]],
                                       columns=[SPLIT_COLUMN, TYPO_COLUMN, CORRECT_TOKEN_COLUMN])
        for test_data in [data, custom_data]:
            candidates = generator.generate_candidates(test_data, threads_number=1,
                                                       start_pool_size=len(test_data) + 1)
            self.assertFalse(candidates.isnull().values.any())
            self.assertSequenceEqual(set(candidates[ID_COLUMN].values), set(test_data.index))
            self.assertSequenceEqual(set(candidates[TYPO_COLUMN].values),
                                     set(test_data[TYPO_COLUMN].values))


if __name__ == "__main__":
    unittest.main()
