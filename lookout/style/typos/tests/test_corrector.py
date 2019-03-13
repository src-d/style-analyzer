import io
from os.path import join
import pathlib
import unittest

import pandas

from lookout.style.typos.corrector import TyposCorrector
from lookout.style.typos.utils import Columns

TEST_DATA_PATH = str(pathlib.Path(__file__).parent)
FASTTEXT_DUMP_FILE = str(pathlib.Path(__file__).parent / "test_ft.bin")
VOCABULARY_FILE = str(pathlib.Path(__file__).parent / "test_frequencies.csv.xz")


class TyposCorrectorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = pandas.read_csv(join(TEST_DATA_PATH, "test_data.csv.xz"),
                                   index_col=0, keep_default_na=False)
        cls.custom_data = pandas.DataFrame([["get tokens num", "get", "get"],
                                            ["gwt tokens", "gwt", "get"],
                                            ["get tokem", "tokem", "token"]],
                                           columns=[Columns.Split, Columns.Token,
                                                    Columns.CorrectToken])
        cls.corrector = TyposCorrector()
        cls.corrector.initialize_generator(VOCABULARY_FILE, VOCABULARY_FILE, FASTTEXT_DUMP_FILE)
        cls.corrector.train(cls.data)

    def test_processes_number_setter(self):
        # Use unlikely number of processes for test
        self.corrector.processes_number = 5
        self.assertEqual(self.corrector.processes_number, 5)
        self.corrector.processes_number = 1

    def test_corrector_on_df(self):
        custom_data = pandas.DataFrame([["get tokens num", "get", "get"],
                                        ["gwt tokens", "gwt", "get"],
                                        ["get tokem", "tokem", "token"]],
                                       columns=[Columns.Split, Columns.Token,
                                                Columns.CorrectToken])
        self.corrector.train(self.data)
        suggestions = self.corrector.suggest(custom_data)
        self.assertSetEqual(set(suggestions.keys()), set(custom_data.index))
        suggestions_by_batches = self.corrector.suggest_by_batches(custom_data)
        self.assertSetEqual(set(suggestions_by_batches.keys()), set(custom_data.index))
        self.corrector.evaluate(custom_data)
        self.assertEqual(len(self.corrector.metrics), 4)

    def test_corrector_on_file(self):
        self.corrector.train_on_file(join(TEST_DATA_PATH, "test_data.csv.xz"))
        suggestions = self.corrector.suggest_on_file(join(TEST_DATA_PATH, "test_data.csv.xz"))
        self.assertSetEqual(set(suggestions.keys()), set(self.data.index))

    def test_expand_vocabulary(self):
        additional_tokens = {"a", "aaa", "123", "get", "341"}
        vocabulary = self.corrector.generator.tokens
        self.corrector.expand_vocabulary(additional_tokens)
        self.assertSetEqual(self.corrector.generator.tokens, vocabulary.union(additional_tokens))

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
