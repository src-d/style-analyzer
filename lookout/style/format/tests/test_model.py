import os
import tempfile
import unittest

from lookout.style.format.model import FormatModel
from lookout.style.format.rules import TrainableRules
from lookout.style.format.tests.test_rules import load_abalone_data


class FormatModelTests(unittest.TestCase):
    def setUp(self):
        (self.train_x, self.test_x, self.train_y, self.test_y), _, _ = load_abalone_data()
        trainer = TrainableRules("sklearn.tree.DecisionTreeClassifier",
                                 prune_branches_algorithms=[], prune_attributes=False,
                                 min_samples_leaf=26, random_state=1989)
        trainer.fit(self.test_x, self.test_y)
        self.rules = trainer.rules
        self.fm = FormatModel().load(os.path.join(os.path.dirname(__file__), "format-model.asdf"))
        self.maxDiff = None

    def test_save_and_load(self):
        fm1 = FormatModel()
        fm1["js"] = self.rules
        fm1["js2"] = self.rules
        fm1["js3"] = self.rules
        with tempfile.NamedTemporaryFile(prefix="lookout-") as f:
            fm1.save(f.name)
            fm2 = FormatModel().load(f.name)
            self.assertEqual(fm1.languages, fm2.languages)
            for lang in fm1.languages:
                # clean None values in fm1 to match asdf cleaning:
                fm1[lang]._origin = {k: v for k, v in fm1[lang]._origin.items() if v is not None}
                self.assertEqual(fm1[lang].origin, fm2[lang].origin)
                for rule1, rule2 in zip(fm1[lang].rules, fm2[lang].rules):
                    self.assertEqual(rule1.stats[0], rule2.stats[0])
                    self.assertAlmostEqual(rule1.stats[1], rule2.stats[1])
                    for r1, r2 in zip(rule1.attrs, rule2.attrs):
                        self.assertEqual(r1[0], r2[0])
                        self.assertEqual(r1[1], r2[1])
                        self.assertAlmostEqual(r1[2], r2[2], places=6)

    def test_dump(self):
        fm = FormatModel()
        self.assertEqual(fm.dump(), "<unknown name>/[1, 0, 0] <unknown url> <unknown commit>")

        DUMP = """<unknown name>/[1, 0, 2] <unknown url> <unknown commit>

# js
85 rules, avg.len. 4.9

# js2
85 rules, avg.len. 4.9

# js3
85 rules, avg.len. 4.9"""
        self.assertEqual(self.fm.dump(), DUMP)

    def test_len(self):
        fm = FormatModel()
        self.assertEqual(len(fm), 0)
        fm["js"] = self.rules
        self.assertEqual(len(fm), 1)
        fm["js2"] = self.rules
        self.assertEqual(len(fm), 2)

    def test_iter(self):
        langs = set(self.fm.languages)
        for item in self.fm:
            self.assertIn(item, langs)
            langs.remove(item)
        self.assertEqual(len(langs), 0)


if __name__ == "__main__":
    unittest.main()
