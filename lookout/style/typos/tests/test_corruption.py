import unittest

import pandas

from lookout.style.typos.corruption import (
    _rand_typo, corrupt_tokens_in_df, rand_delete, rand_insert, rand_substitution, rand_swap)
from lookout.style.typos.symspell import EditDistance
from lookout.style.typos.utils import Columns


class CorruptionTest(unittest.TestCase):
    def test_corruptions(self):
        token = "abcdefgh"
        dist_calculator = EditDistance(token, "damerau")
        for corruption, distance in [(rand_delete, -1), (rand_insert, 1),
                                     (rand_substitution, 0), (rand_swap, 0)]:
            for _ in range(10):
                corrupted = corruption(token)
                self.assertEqual(dist_calculator.compare(corrupted, 1), 1)
                self.assertEqual(len(corrupted), len(token) + distance)
        for _ in range(10):
            corrupted, _ = _rand_typo((token, token, 1), 0.0)
            self.assertEqual(dist_calculator.compare(corrupted, 1), 1)

    def test_corrupt_tokens_in_df(self):
        data = pandas.DataFrame([["get value", "get"],
                                 ["get value", "value"],
                                 ["gut", "gut"],
                                 ["put tok", "put"],
                                 ["put tok", "tok"],
                                 ["put", "put"]], columns=[Columns.Split, Columns.Token])
        corrupted = corrupt_tokens_in_df(data, 0.5, 0.1, processes_number=1)
        self.assertSetEqual(set(corrupted.columns), {Columns.Split, Columns.Token,
                                                     Columns.CorrectToken, Columns.CorrectSplit})
        self.assertListEqual(list(corrupted[Columns.CorrectSplit]), list(data[Columns.Split]))
        self.assertListEqual(list(corrupted[Columns.CorrectToken]), list(data[Columns.Token]))


if __name__ == "__main__":
    unittest.main()
