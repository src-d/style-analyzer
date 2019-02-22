import pathlib
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
        symspell.load_dictionary(VOCABULARY_FILE)
        self.assertEqual(symspell.lookup("ofset", 0, 1)[0].term, "offset")


if __name__ == "__main__":
    unittest.main()
