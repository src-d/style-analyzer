import io
from os.path import join
import pathlib
import unittest

import numpy
import pandas

from lookout.style.typos.generation import get_candidates_features, get_candidates_metadata
from lookout.style.typos.ranking import CandidatesRanker
from lookout.style.typos.utils import (CANDIDATE_COLUMN, CORRECT_TOKEN_COLUMN, ID_COLUMN,
                                       SPLIT_COLUMN, TYPO_COLUMN)

TEST_DATA_PATH = str(pathlib.Path(__file__).parent)


class CandidatesRankerTest(unittest.TestCase):
    def test_custom_ranker(self):
        custom_data = pandas.DataFrame([[["get", "tokens", "num"], "get", "get"],
                                        [["gwt", "tokens"], "gwt", "get"],
                                        [["get", "tokem"], "tokem", "token"]],
                                       columns=[SPLIT_COLUMN, TYPO_COLUMN, CORRECT_TOKEN_COLUMN])
        custom_candidates_tokens = pandas.DataFrame([[0, "get", "get"],
                                                     [1, "gwt", "get"],
                                                     [1, "gwt", "gpt"],
                                                     [2, "tokem", "tokem"],
                                                     [2, "tokem", "taken"],
                                                     [2, "tokem", "token"]],
                                                    columns=[ID_COLUMN, TYPO_COLUMN,
                                                             CANDIDATE_COLUMN])
        custom_candidates_features = numpy.array([[10.0, 1.0, 3.5],
                                                  [9.6, 1.3, 2.3],
                                                  [0.23, -1.3, 156.3],
                                                  [5.6, 0.4, 32.65],
                                                  [-0.03, 0.2, 678.4],
                                                  [8.9, 0.8, 5.2]])

        ranker = CandidatesRanker()
        ranker.fit(custom_data[CORRECT_TOKEN_COLUMN], custom_candidates_tokens,
                   custom_candidates_features, val_part=0.5)
        suggestions = ranker.rank(custom_candidates_tokens, custom_candidates_features,
                                  n_candidates=1, return_all=True)
        self.assertSetEqual(set(suggestions.keys()), set(custom_data.index))

    def test_ranker(self):
        data = pandas.read_csv(join(TEST_DATA_PATH, "test_data.csv.xz"),
                               index_col=0).infer_objects()
        candidates = pandas.read_pickle(join(TEST_DATA_PATH, "test_data_candidates_full.pkl"))
        ranker = CandidatesRanker()
        ranker.fit(data[CORRECT_TOKEN_COLUMN], get_candidates_metadata(candidates),
                   get_candidates_features(candidates))
        suggestions = ranker.rank(get_candidates_metadata(candidates),
                                  get_candidates_features(candidates),
                                  n_candidates=3, return_all=True)
        self.assertSetEqual(set(suggestions.keys()), set(data.index))

    def test_save_load(self):
        data = pandas.read_csv(join(TEST_DATA_PATH, "test_data.csv.xz"),
                               index_col=0).infer_objects()
        candidates = pandas.read_pickle(join(TEST_DATA_PATH, "test_data_candidates_full.pkl"))
        ranker = CandidatesRanker()
        ranker.fit(data[CORRECT_TOKEN_COLUMN], get_candidates_metadata(candidates),
                   get_candidates_features(candidates))
        with io.BytesIO() as buffer:
            ranker.save(buffer)
            print(buffer.tell())
            buffer.seek(0)
            ranker2 = CandidatesRanker().load(buffer)
        print(ranker)
        self.assertTrue(ranker == ranker2)

    def test_eq(self):
        self.assertTrue(CandidatesRanker() == CandidatesRanker())
        data = pandas.read_csv(join(TEST_DATA_PATH, "test_data.csv.xz"),
                               index_col=0).infer_objects()
        candidates = pandas.read_pickle(join(TEST_DATA_PATH, "test_data_candidates_full.pkl"))
        ranker = CandidatesRanker()
        ranker.fit(data[CORRECT_TOKEN_COLUMN], get_candidates_metadata(candidates),
                   get_candidates_features(candidates))
        self.assertFalse(ranker == CandidatesRanker())


if __name__ == "__main__":
    unittest.main()
