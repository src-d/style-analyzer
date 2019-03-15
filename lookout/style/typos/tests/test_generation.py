import io
import pathlib
import unittest

import numpy
from numpy.testing import assert_array_equal
import pandas
from pandas.util.testing import assert_frame_equal

from lookout.style.typos.generation import (CandidatesGenerator, get_candidates_features,
                                            get_candidates_metadata)
from lookout.style.typos.utils import Columns

TEST_DATA_PATH = pathlib.Path(__file__).parent
FASTTEXT_DUMP_FILE = str(TEST_DATA_PATH / "test_ft.bin")
VOCABULARY_FILE = str(TEST_DATA_PATH / "test_frequencies.csv.xz")


class CandidatesSplitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.candidates = pandas.DataFrame(
            [[0, "gut", "get", numpy.array([0.1, 0.2], dtype="float32")],
             [1, "cit", "cut", numpy.array([0.8, 0.3], dtype="float32")],
             [1, "cit", "cit", numpy.array([0.1, 0.5], dtype="float32")]],
            columns=[Columns.Id, Columns.Token, Columns.Candidate, Columns.Features])

    def test_get_candidates_tokens(self):
        tokens = pandas.DataFrame([[0, "gut", "get"],
                                   [1, "cit", "cut"],
                                   [1, "cit", "cit"]],
                                  columns=[Columns.Id, Columns.Token, Columns.Candidate])
        assert_frame_equal(get_candidates_metadata(self.candidates), tokens)

    def test_get_candidates_features(self):
        features = numpy.array([[0.1, 0.2],
                                [0.8, 0.3],
                                [0.1, 0.5]], dtype="float32")
        assert_array_equal(get_candidates_features(self.candidates), features)


class GeneratorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = {"neighbors_number": 3, "edit_dist_number": 3, "max_distance": 3}
        cls.generator = CandidatesGenerator()
        cls.generator.construct(VOCABULARY_FILE, VOCABULARY_FILE, FASTTEXT_DUMP_FILE,
                                cls.config)

    @unittest.skip("CandidatesGenerator.__eq__ needs refactoring. Test is currently flaky.")
    def test_save_load(self):
        with io.BytesIO() as buffer:
            self.generator.save(buffer)
            print(buffer.tell())
            buffer.seek(0)
            generator2 = CandidatesGenerator().load(buffer)

        self.assertTrue(self.generator == generator2)

    def test_generate_candidates(self):
        data = pandas.read_csv(str(TEST_DATA_PATH / "test_data.csv.xz"),
                               index_col=0, keep_default_na=False)
        custom_data = pandas.DataFrame([["get tokens num", "tokens", "tokens"],
                                        ["gwt tokens", "gwt", "get"],
                                        ["get tokem", "tokem", "token"]],
                                       columns=[Columns.Split, Columns.Token,
                                                Columns.CorrectToken])
        for test_data in [data, custom_data]:
            candidates = self.generator.generate_candidates(
                test_data, processes_number=1)
            self.assertFalse(candidates.isnull().values.any())
            self.assertSetEqual(set(candidates[Columns.Id].values), set(test_data.index))
            self.assertSetEqual(set(candidates[Columns.Token].values),
                                set(test_data[Columns.Token].values))

    def test_expand_vocabulary(self):
        generator = CandidatesGenerator()
        generator.construct(VOCABULARY_FILE, VOCABULARY_FILE, FASTTEXT_DUMP_FILE, self.config)
        additional_tokens = {"a", "aaa", "123", "get", "341"}
        vocabulary = generator.tokens
        generator.expand_vocabulary(additional_tokens)
        self.assertSetEqual(generator.tokens, vocabulary.union(additional_tokens))


if __name__ == "__main__":
    unittest.main()
