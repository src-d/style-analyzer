import tempfile
import unittest

from lookout.style.typos.general import prepare_data
from lookout.style.typos.utils import Columns, read_frequencies, read_vocabulary


class PreparationTest(unittest.TestCase):
    def test_prepare_data(self):
        with tempfile.NamedTemporaryFile() as input_file:
            with tempfile.NamedTemporaryFile() as voc_file:
                with tempfile.NamedTemporaryFile() as freq_file:
                    params = {
                        "load_from_drive": True,
                        "input_path": input_file.name,
                        "vocabulary_size": 5000,
                        "frequencies_size": None,
                        "vocabulary_path": voc_file.name,
                        "frequencies_path": freq_file.name,
                    }
                    data = prepare_data(params)
                    vocabulary = read_vocabulary(params["vocabulary_path"])
                    self.assertEqual(len(vocabulary), params["vocabulary_size"])
                    self.assertSetEqual(set(data[Columns.Token]), set(vocabulary))
                    frequencies = read_frequencies(params["frequencies_path"])
                    self.assertTrue(set(vocabulary).issubset(set(frequencies.keys())))
                    self.assertTrue({Columns.Token, Columns.Split}.issubset(data.columns))


if __name__ == "__main__":
    unittest.main()
