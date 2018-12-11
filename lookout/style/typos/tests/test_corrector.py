import io
from os.path import join
import pathlib
import unittest

import pandas

from lookout.style.typos.corrector import TyposCorrector
from lookout.style.typos.utils import CORRECT_TOKEN_COLUMN, SPLIT_COLUMN, TYPO_COLUMN

TEST_DATA_PATH = str(pathlib.Path(__file__).parent)
FASTTEXT_DUMP_FILE = str(pathlib.Path(__file__).parent / "id_vecs_10.bin")
VOCABULARY_FILE = str(pathlib.Path(__file__).parent / "test_frequencies.csv.xz")


class TyposCorrectorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = pandas.read_csv(join(TEST_DATA_PATH, "test_data.csv.xz"),
                                   index_col=0).infer_objects()
        cls.corrector = TyposCorrector()
        cls.corrector.initialize_generator(VOCABULARY_FILE, VOCABULARY_FILE, FASTTEXT_DUMP_FILE)

    def test_corrector_on_df(self):
        custom_data = pandas.DataFrame([[["get", "tokens", "num"], "get", "get"],
                                        [["gwt", "tokens"], "gwt", "get"],
                                        [["get", "tokem"], "tokem", "token"]],
                                       columns=[SPLIT_COLUMN, TYPO_COLUMN, CORRECT_TOKEN_COLUMN])
        self.corrector.train(self.data)
        suggestions = self.corrector.suggest(custom_data)
        self.assertSetEqual(set(suggestions.keys()), set(custom_data.index))

    def test_corrector_on_file(self):
        self.corrector.train_on_file(join(TEST_DATA_PATH, "test_data.csv.xz"))
        suggestions = self.corrector.suggest_file(join(TEST_DATA_PATH, "test_data.csv.xz"))
        self.assertSetEqual(set(suggestions.keys()), set(self.data.index))

    @unittest.skip("CandidatesGenerator.__eq__ needs refactoring. Test is currently flaky.")
    def test_save_load(self):
        self.corrector.train(self.data)
        with io.BytesIO() as buffer:
            self.corrector.save(buffer)
            print(buffer.tell())
            buffer.seek(0)
            corrector2 = TyposCorrector().load(buffer)
        self.assertEqual(self.corrector, corrector2)


if __name__ == "__main__":
    unittest.main()
