import os
import pathlib
import tempfile
import unittest

from gensim.models import FastText
import pandas

from lookout.style.typos.preparation import (get_datasets, prepare_data, train_and_evaluate,
                                             train_fasttext)
from lookout.style.typos.utils import Columns, read_frequencies, read_vocabulary


TEST_DATA_DIR = pathlib.Path(__file__).parent


class PreparationTest(unittest.TestCase):
    def test_prepare_data_with_load(self):
        with tempfile.TemporaryDirectory(prefix="lookout_typos_prepare_load_") as temp_dir:
            params = {
                "data_dir": temp_dir,
                "dataset_url": "https://docs.google.com/uc?export=download&"
                               "id=1htVU1UR0gSmopVbvU6_Oc-4iD0cw1ldo",
                "input_path": None,
                "raw_data_filename": "raw_test_data.csv.xz",
                "vocabulary_size": 10,
                "frequencies_size": 20,
                "vocabulary_filename": "vocabulary.csv",
                "frequencies_filename": "frequencies.csv",
            }
            data = prepare_data(params)
            vocabulary = read_vocabulary(os.path.join(temp_dir, params["vocabulary_filename"]))
            self.assertEqual(len(vocabulary), params["vocabulary_size"])
            self.assertTrue(set(data[Columns.Token]).issubset(set(vocabulary)))
            frequencies = read_frequencies(os.path.join(temp_dir, params["frequencies_filename"]))
            self.assertEqual(len(frequencies), params["frequencies_size"])
            self.assertTrue(set(vocabulary).issubset(set(frequencies.keys())))
            self.assertTrue({Columns.Token, Columns.Split}.issubset(data.columns))

    def test_prepare_data_from_file(self):
        with tempfile.TemporaryDirectory(prefix="lookout_typos_prepare_local_") as temp_dir:
            params = {
                "data_dir": temp_dir,
                "input_path": str(TEST_DATA_DIR / "raw_test_data.csv.xz"),
                "vocabulary_size": 10,
                "frequencies_size": 20,
                "vocabulary_filename": "vocabulary.csv",
                "frequencies_filename": "frequencies.csv",
            }
            data = prepare_data(params)
            vocabulary = read_vocabulary(os.path.join(temp_dir, params["vocabulary_filename"]))
            self.assertEqual(len(vocabulary), params["vocabulary_size"])
            self.assertTrue(set(data[Columns.Token]).issubset(set(vocabulary)))
            frequencies = read_frequencies(os.path.join(temp_dir, params["frequencies_filename"]))
            self.assertEqual(len(frequencies), params["frequencies_size"])
            self.assertTrue(set(vocabulary).issubset(set(frequencies.keys())))
            self.assertTrue({Columns.Token, Columns.Split}.issubset(data.columns))


class DatasetsTest(unittest.TestCase):
    def test_get_datasets(self):
        prepared = pandas.read_csv(str(TEST_DATA_DIR / "prepared_data.csv.xz"),
                                   index_col=0)
        train, test = get_datasets(prepared, train_size=1000, test_size=100,
                                   typo_probability=0.5, add_typo_probability=0.5)
        self.assertTrue({Columns.Token, Columns.CorrectToken, Columns.Split,
                         Columns.CorrectSplit}.issubset(set(train.columns)))
        corrupted = sum(train[Columns.Token] != train[Columns.CorrectToken])
        self.assertAlmostEqual(corrupted, 500, delta=50)
        self.assertEqual(len(train), 1000)
        self.assertEqual(len(test), 100)
        print(len(set(train[Columns.Token]).intersection(set(test[Columns.Token]))))


class FasttextTest(unittest.TestCase):
    def test_get_fasttext_model(self):
        data = pandas.read_csv(str(TEST_DATA_DIR / "prepared_data.csv.xz"),
                               index_col=0)
        with tempfile.TemporaryDirectory(prefix="lookout_typos_fasttext_") as temp_dir:
            params = {"size": 100, "fasttext_path": os.path.join(temp_dir, "ft.bin"), "dim": 5}
            train_fasttext(data, params)
            model = FastText.load_fasttext_format(params["fasttext_path"])
            self.assertTupleEqual(model.wv["get"].shape, (5,))


class TrainingTest(unittest.TestCase):
    def test_train_and_evaluate(self):
        data = pandas.read_csv(str(TEST_DATA_DIR / "test_data.csv.xz"), index_col=0)
        vocabulary_file = str(TEST_DATA_DIR / "test_frequencies.csv.xz")
        model = train_and_evaluate(data, data, vocabulary_file, vocabulary_file,
                                   str(TEST_DATA_DIR / "test_ft.bin"))
        suggestions = model.suggest_on_file(str(TEST_DATA_DIR / "test_data.csv.xz"))
        self.assertSetEqual(set(suggestions.keys()), set(data.index))


if __name__ == "__main__":
    unittest.main()
