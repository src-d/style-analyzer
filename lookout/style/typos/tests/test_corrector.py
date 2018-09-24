from os.path import join
import pathlib
import unittest

import pandas

from lookout.style.typos.corrector import TyposCorrector
from lookout.style.typos.utils import CORRECT_TOKEN_COLUMN, SPLIT_COLUMN, TYPO_COLUMN


TEST_DATA_PATH = str(pathlib.Path(__file__).parent)
FASTTEXT_DUMP_FILE = "lookout/style/typos/id_vecs_10.bin"
VOCABULARY_FILE = "lookout/style/typos/16k_vocabulary.csv"
FREQUENCIES_FILE = "lookout/style/typos/frequencies.csv"


class TyposCorrectorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = pandas.read_csv(join(TEST_DATA_PATH, "test_data.csv"),
                                   index_col=0).infer_objects()
        cls.corrector = TyposCorrector()
        cls.corrector.create_model(VOCABULARY_FILE, FREQUENCIES_FILE, FASTTEXT_DUMP_FILE)

    def test_corrector_on_df(self):
        custom_data = pandas.DataFrame([[["get", "tokens", "num"], "get", "get"],
                                        [["gwt", "tokens"], "gwt", "get"],
                                        [["get", "tokem"], "tokem", "token"]],
                                       columns=[SPLIT_COLUMN, TYPO_COLUMN, CORRECT_TOKEN_COLUMN])
        self.corrector.train(self.data)
        suggestions = self.corrector.suggest(custom_data)
        self.assertSetEqual(set(suggestions.keys()), set(custom_data.index))

    def test_corrector_on_file(self):
        self.corrector.train_on_file(join(TEST_DATA_PATH, "test_data.csv"))
        suggestions = self.corrector.suggest_file(join(TEST_DATA_PATH, "test_data.csv"))
        self.assertSetEqual(set(suggestions.keys()), set(self.data.index))


if __name__ == "__main__":
    unittest.main()
