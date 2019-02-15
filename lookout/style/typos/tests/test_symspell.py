import lzma
import pathlib
import tempfile
import unittest

from lookout.style.typos.symspell import EditDistance, SymSpell

TEST_DATA_PATH = str(pathlib.Path(__file__).parent)
VOCABULARY_FILE = str(pathlib.Path(__file__).parent / "test_frequencies.csv.xz")


class EditDistanceTest(unittest.TestCase):
    def test_compare(self):
        dist_calculator = EditDistance("token", "damerau")
        self.assertEqual(dist_calculator.compare("tokem", 1), 1)
        self.assertEqual(dist_calculator.compare("tokems", 1), -1)
        self.assertEqual(dist_calculator.compare("tokems", 2), 2)


class SymSpellTest(unittest.TestCase):
    def test_generate_candidates(self):
        symspell = SymSpell(max_dictionary_edit_distance=1)
        vocabulary_file = tempfile.NamedTemporaryFile()
        with lzma.open(VOCABULARY_FILE, "rt") as compressed:
            with open(vocabulary_file.name, "w") as f:
                f.write(compressed.read())
        symspell.load_dictionary(vocabulary_file.name)
        self.assertEqual(symspell.lookup("ofset", 0, 1)[0].term, "offset")
        vocabulary_file.close()


if __name__ == "__main__":
    unittest.main()
