import os
import tempfile
import unittest

from lookout.style.format.model import FormatModel
from lookout.style.format.rules import TrainableRules
from lookout.style.format.tests.test_rules import load_abalone_data


def compare_models(test_case: unittest.TestCase,
                   format_model1: FormatModel,
                   format_model2: FormatModel):
    test_case.assertEqual(format_model1.languages, format_model2.languages)
    for lang in format_model1.languages:
        # clean None values in format_model1 to match asdf cleaning:`
        test_case.assertEqual(format_model1[lang]._origin_config.keys(),
                              format_model2[lang]._origin_config.keys())
        c = {k: v for k, v in format_model1[lang].origin_config["trainable_rules"].items()
             if v is not None}
        test_case.assertEqual(c, format_model2[lang].origin_config["trainable_rules"])
        for rule1, rule2 in zip(format_model1[lang].rules, format_model2[lang].rules):
            test_case.assertEqual(rule1.stats[0], rule2.stats[0])
            test_case.assertAlmostEqual(rule1.stats[1], rule2.stats[1])
            for r1, r2 in zip(rule1.attrs, rule2.attrs):
                test_case.assertEqual(r1[0], r2[0])
                test_case.assertEqual(r1[1], r2[1])
                test_case.assertAlmostEqual(r1[2], r2[2], places=6)


class FormatModelTests(unittest.TestCase):
    def setUp(self):
        (self.train_x, self.test_x, self.train_y, self.test_y), _, _ = load_abalone_data()
        self.config = {
            "trainable_rules": {
                "base_model_name": "sklearn.tree.DecisionTreeClassifier",
                "prune_branches_algorithms": [],
                "prune_attributes": False,
                "min_samples_leaf": 26,
                "random_state": 1989,
            }
        }
        trainer = TrainableRules(**self.config["trainable_rules"], origin_config=self.config)
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
            compare_models(self, fm1, fm2)

    def test_dump(self):
        fm = FormatModel()
        self.assertEqual(fm.dump(), "<unknown name>/[1, 0, 0] <unknown url> <unknown commit>")

        DUMP = """<unknown name>/['1'] <unknown url> <unknown commit>

# javascript
45 rules, avg.len. 6.3"""
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
