import unittest

import pandas

from lookout.style.typos.preprocessing import pick_subset_of_df_rows
from lookout.style.typos.utils import Columns


class PreprocessingTest(unittest.TestCase):
    def test_pick_subset_of_df_rows(self):
        data = pandas.DataFrame([["get value", "get"],
                                 ["get value", "value"],
                                 ["gut", "gut"],
                                 ["put tok", "put"],
                                 ["put tok", "tok"],
                                 ["put", "put"]], columns=[Columns.Split, Columns.Token])
        self.assertTupleEqual(pick_subset_of_df_rows(data, size=4).shape, (4, 3))


if __name__ == "__main__":
    unittest.main()
