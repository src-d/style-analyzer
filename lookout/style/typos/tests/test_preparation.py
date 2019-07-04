import os
import pathlib
import tempfile
import unittest

from gensim.models.fasttext import load_facebook_vectors
import pandas

from lookout.style.typos.preparation import (generate_vocabulary, get_datasets, prepare_data,
                                             train_and_evaluate, train_fasttext,
                                             train_from_scratch)
from lookout.style.typos.utils import Columns, read_frequencies, read_vocabulary


TEST_DATA_DIR = pathlib.Path(__file__).parent


class PreparationTest(unittest.TestCase):
    def test_get_vocabulary(self):
        config = {
            "stable": 0,
            "suspicious": 0,
            "non_suspicious": 0,
            "processes_number": 1,
        }
        vocabulary = generate_vocabulary(str(TEST_DATA_DIR / "test_frequencies.csv.xz"), config)
        self.assertEqual(len(vocabulary), 98)

    def test_prepare_data_with_load(self):
        with tempfile.TemporaryDirectory(prefix="lookout_typos_prepare_load_") as temp_dir:
            config = {
                "data_dir": temp_dir,
                "dataset_url": "https://docs.google.com/uc?export=download&"
                               "id=1htVU1UR0gSmopVbvU6_Oc-4iD0cw1ldo",
                "input_path": None,
                "raw_data_filename": "raw_test_data.csv.xz",
                "vocabulary_filename": "vocabulary.csv",
                "frequencies_filename": "frequencies.csv",
            }
            data = prepare_data(config)
            vocabulary = read_vocabulary(os.path.join(temp_dir, config["vocabulary_filename"]))
            frequencies = read_frequencies(os.path.join(temp_dir, config["frequencies_filename"]))
            self.assertTrue(set(data[Columns.Token]).issubset(set(frequencies.keys())))
            self.assertTrue(set(vocabulary).issubset(set(frequencies.keys())))
            self.assertTrue({Columns.Token, Columns.Split}.issubset(data.columns))

    def test_prepare_data_from_file(self):
        with tempfile.TemporaryDirectory(prefix="lookout_typos_prepare_local_") as temp_dir:
            config = {
                "data_dir": temp_dir,
                "input_path": str(TEST_DATA_DIR / "raw_test_data.csv.xz"),
                "vocabulary_filename": "vocabulary.csv",
                "frequencies_filename": "frequencies.csv",
            }
            data = prepare_data(config)
            vocabulary = read_vocabulary(os.path.join(temp_dir, config["vocabulary_filename"]))
            frequencies = read_frequencies(os.path.join(temp_dir, config["frequencies_filename"]))
            self.assertTrue(set(data[Columns.Token]).issubset(set(frequencies.keys())))
            self.assertTrue(set(vocabulary).issubset(set(frequencies.keys())))
            self.assertTrue({Columns.Token, Columns.Split}.issubset(data.columns))


class DatasetsTest(unittest.TestCase):
    def test_get_datasets(self):
        prepared = pandas.read_csv(str(TEST_DATA_DIR / "prepared_data.csv.xz"),
                                   index_col=0, keep_default_na=False)
        config = {
            "train_size": 1000,
            "test_size": 100,
            "typo_probability": 0.5,
            "add_typo_probability": 0.05,
            "train_path": None,
            "test_path": None,
        }
        train, test = get_datasets(prepared, config)
        self.assertTrue({Columns.Token, Columns.CorrectToken, Columns.Split,
                         Columns.CorrectSplit}.issubset(set(train.columns)))
        corrupted = sum(train[Columns.Token] != train[Columns.CorrectToken])
        self.assertEqual(corrupted, 500)
        self.assertEqual(len(train), 1000)
        self.assertEqual(len(test), 100)
        print(len(set(train[Columns.Token]).intersection(set(test[Columns.Token]))))


class FasttextTest(unittest.TestCase):
    def test_get_fasttext_model(self):
        data = pandas.read_csv(str(TEST_DATA_DIR / "prepared_data.csv.xz"),
                               index_col=0, keep_default_na=False)
        with tempfile.TemporaryDirectory(prefix="lookout_typos_fasttext_") as temp_dir:
            config = {"size": 100, "path": os.path.join(temp_dir, "ft.bin"), "dim": 5}
            train_fasttext(data, config)
            wv = load_facebook_vectors(config["path"])
            self.assertTupleEqual(wv["get"].shape, (5,))


class TrainingTest(unittest.TestCase):
    def test_train_and_evaluate(self):
        data = pandas.read_csv(str(TEST_DATA_DIR / "test_data.csv.xz"), index_col=0,
                               keep_default_na=False)
        vocabulary_file = str(TEST_DATA_DIR / "test_frequencies.csv.xz")
        model = train_and_evaluate(data, data, vocabulary_file, vocabulary_file,
                                   str(TEST_DATA_DIR / "test_ft.bin"))
        suggestions = model.suggest_on_file(str(TEST_DATA_DIR / "test_data.csv.xz"))
        self.assertSetEqual(set(suggestions.keys()), set(data.index))

    def test_train_from_scratch(self):
        with tempfile.TemporaryDirectory(prefix="lookout_typos_prepare_load_") as temp_dir:
            config = {
                "preparation": {
                    "data_dir": temp_dir,
                    "dataset_url": "https://docs.google.com/uc?export=download&"
                                   "id=1htVU1UR0gSmopVbvU6_Oc-4iD0cw1ldo",
                    "input_path": None,
                    "raw_data_filename": "raw_test_data.csv.xz",
                    "vocabulary_size": 10,
                    "frequencies_size": 20,
                    "vocabulary_filename": "vocabulary.csv",
                    "frequencies_filename": "frequencies.csv",
                },
                "fasttext": {
                    "size": 100,
                    "path": os.path.join(temp_dir, "ft.bin"),
                    "dim": 5,
                },
                "datasets": {
                    "train_size": 1000,
                    "test_size": 100,
                    "typo_probability": 0.5,
                    "add_typo_probability": 0.05,
                    "train_path": None,
                    "test_path": None,
                },
                "corrector_path": None,
            }
            model = train_from_scratch(config)
        test_data = pandas.read_csv(str(TEST_DATA_DIR / "test_data.csv.xz"), index_col=0,
                                    keep_default_na=False)
        suggestions = model.suggest(test_data)
        self.assertSetEqual(set(suggestions.keys()), set(test_data.index))


if __name__ == "__main__":
    unittest.main()
