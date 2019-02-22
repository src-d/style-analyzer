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
            },
        }
        trainer = TrainableRules(**self.config["trainable_rules"], origin_config=self.config)
        trainer.fit(self.test_x, self.test_y)
        self.rules = trainer.rules
        self.fm = FormatModel().load(os.path.join(os.path.dirname(__file__), "model_jquery.asdf"))
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
        self.assertEqual(fm.dump(), "generic/[1, 0, 0] <unknown url> <unknown commit>")

        DUMP = """code-format/[1] file:///var/folders/kw/93jybvs16_954hytgsq6ld7r0000gn/T/top-repos-quality-repos-jigt1n8g/jquery dae5f3ce3d2df27873d01f0d9682f6a91ad66b87

# javascript
1159 rules, avg.len. 12.7
## train
PPCR: 0.993413
### report
macro
{'f1-score': 0.7270769669476458,
 'precision': 0.8106858458605273,
 'recall': 0.7061608014058862,
 'support': 163931}
micro
{'f1-score': 0.9704570825530253,
 'precision': 0.9704570825530253,
 'recall': 0.9704570825530253,
 'support': 163931}
weighted
{'f1-score': 0.9682573644719648,
 'precision': 0.9688067324990776,
 'recall': 0.9704570825530253,
 'support': 163931}
### report_full
macro
{'f1-score': 0.7207757082876136,
 'precision': 0.8106858458605273,
 'recall': 0.6958571203075536,
 'support': 165018}
micro
{'f1-score': 0.9672502424387974,
 'precision': 0.9704570825530253,
 'recall': 0.9640645262941012,
 'support': 165018}
weighted
{'f1-score': 0.964254372281313,
 'precision': 0.967999533892513,
 'recall': 0.9640645262941012,
 'support': 165018}
## test
PPCR: 0.992673
### report
macro
{'f1-score': 0.670106403195044,
 'precision': 0.7675483060510728,
 'recall': 0.667193540547618,
 'support': 39563}
micro
{'f1-score': 0.9646134014104087,
 'precision': 0.9646134014104087,
 'recall': 0.9646134014104087,
 'support': 39563}
weighted
{'f1-score': 0.9623528642977015,
 'precision': 0.964064574202937,
 'recall': 0.9646134014104087,
 'support': 39563}
### report_full
macro
{'f1-score': 0.6645299678762785,
 'precision': 0.7675483060510728,
 'recall': 0.6569592424077594,
 'support': 39855}
micro
{'f1-score': 0.961066760683976,
 'precision': 0.9646134014104087,
 'recall': 0.9575461046292811,
 'support': 39855}
weighted
{'f1-score': 0.9579239894541836,
 'precision': 0.9627543953777487,
 'recall': 0.9575461046292811,
 'support': 39855}"""  # noqa
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
