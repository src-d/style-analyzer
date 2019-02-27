import os
import pathlib
import shutil
import tempfile
import unittest

from gensim.models import FastText
import pandas

from lookout.style.typos.datasets_preparation import get_datasets, prepare_data, tune_fasttext_model
from lookout.style.typos.utils import Columns, read_frequencies, read_vocabulary


class PreparationTest(unittest.TestCase):
    def test_prepare_data_with_load(self):
        temp_dir = tempfile.mkdtemp()
        params = {
            "data_dir": temp_dir,
            "input_path": None,
            "vocabulary_size": 1000,
            "frequencies_size": 10000,
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
        shutil.rmtree(temp_dir)

    def test_prepare_data_from_file(self):
        temp_dir = tempfile.mkdtemp()
        params = {
            "data_dir": temp_dir,
            "input_path": str(pathlib.Path(__file__).parent / "raw_test_data.csv.xz"),
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
        shutil.rmtree(temp_dir)


class DatasetsTest(unittest.TestCase):
    def test_get_datasets(self):
        prepared = pandas.read_csv(str(pathlib.Path(__file__).parent / "prepared_data.csv.xz"),
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
        data = pandas.read_csv(str(pathlib.Path(__file__).parent / "prepared_data.csv.xz"),
                               index_col=0)
        with tempfile.NamedTemporaryFile() as ft_file:
            params = {"size": 1000, "fasttext_path": ft_file.name, "dim": 5}
            tune_fasttext_model(data, params)
            model = FastText.load_fasttext_format(ft_file.name)
            self.assertTupleEqual(model.wv["get"].shape, (5,))


if __name__ == "__main__":
    unittest.main()
