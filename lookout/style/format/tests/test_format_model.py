import os
import tempfile
import unittest

from sklearn import tree

from lookout.style.format.rules import Rules, FormatModel
from lookout.style.format.tests.test_rules import load_abalone_data


class FormatModelTests(unittest.TestCase):
    def setUp(self):
        self.train_x, self.test_x, self.train_y, self.test_y = load_abalone_data()
        self.model = tree.DecisionTreeClassifier(min_samples_leaf=26, random_state=1989)
        self.model = self.model.fit(self.train_x, self.train_y)
        self.rules = Rules(self.model, prune_branches=False, prune_attributes=False)
        self.rules.fit(self.test_x, self.test_y)
        self.fm = FormatModel().load(os.path.join(os.path.dirname(__file__), "format-model.asdf"))
        pass

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
                self.assertEqual(fm1[lang].get_saved_params(),
                                 fm2[lang].get_saved_params())
                for rule1, rule2 in zip(fm1[lang]._rules, fm2[lang]._rules):
                    self.assertEqual(rule1.stats[0], rule2.stats[0])
                    self.assertAlmostEqual(rule1.stats[1], rule2.stats[1])
                    for r1, r2 in zip(rule1.attrs, rule2.attrs):
                        self.assertEqual(r1[0], r2[0])
                        self.assertEqual(r1[1], r2[1])
                        self.assertAlmostEqual(r1[2], r2[2])

    def test_dump(self):
        fm = FormatModel()
        self.assertEqual(fm.dump(), "<empty FormatModel>")

        DUMP = "Model languages: ['js', 'js2', 'js3'].\n" \
               "First model's params: prune_attributes=False, prune_branches=False, " \
               "prune_branches_algorithm=top-down-greedy, top_down_greedy_budget=" \
               "TopDownGreedyBudget(absolute=False, value=1.0), " \
               "uncertain_attributes=True\n" \
               "First model's rules number: 85.\n"
        self.assertEqual(self.fm.dump(), DUMP)

    def test_len(self):
        fm = FormatModel()
        self.assertEqual(len(fm), 0)
        fm["js"] = self.rules
        self.assertEqual(len(fm), 1)
        fm["js2"] = self.rules
        self.assertEqual(len(fm), 2)

    def test_set_unfitted_item(self):
        rules = Rules(self.model, prune_branches=False, prune_attributes=False)
        fm = FormatModel()
        with self.assertRaises(ValueError):
            fm[""] = rules

    def test_iter(self):
        langs = set(self.fm.languages)
        for item in self.fm:
            self.assertIn(item, langs)
            langs.remove(item)
        self.assertEqual(len(langs), 0)

    def test_constuct(self):
        fm = FormatModel()
        fm.construct((
            ("name1", self.rules),
            ("name2", self.rules),
            ("name3", self.rules),
        ))
        self.assertIn("name1", fm)
        self.assertIn("name2", fm)
        self.assertIn("name3", fm)


if __name__ == "__main__":
    unittest.main()
