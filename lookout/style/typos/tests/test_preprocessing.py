import unittest

import pandas
from pandas.util.testing import assert_frame_equal

from lookout.style.typos.preprocessing import check_split, filter_splits
from lookout.style.typos.utils import Columns


class PreprocessingTest(unittest.TestCase):
    def test_check_split(self):
        self.assertTrue(check_split("get the value", {"get", "the", "value", "here"}))
        self.assertFalse(check_split("get the value", {"get", "value", "here"}))

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


if __name__ == "__main__":
    unittest.main()
