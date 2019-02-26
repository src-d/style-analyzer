import os
import unittest

import pandas
from pandas.util.testing import assert_frame_equal

from lookout.style.typos.run_corrector import defaults_for_preparation, prepare_data
from lookout.style.typos.utils import read_vocabulary, read_frequencies


class PreparationTest(unittest.TestCase):
    def test_prepare_data(self):
        params = {
            
        }
        data = prepare_data({"vocabulary_size": 5000})
        vocabulary = read_vocabulary(defaults_for_preparation["vocabulary_path"])
        self.assertEqual(len(vocabulary), 5000)
        self.assertTrue(os.path.exists(defaults_for_preparation["frequencies_path"]))
        frequencies = read_frequencies(defaults_for_preparation["frequencies_path"])



if __name__ == "__main__":
    unittest.main()
