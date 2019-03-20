import lzma
from pathlib import Path
import unittest

import bblfsh
from lookout.core.analyzer import UnicodeFile
from sklearn import model_selection, tree

from lookout.style.format.analyzer import FormatAnalyzer
from lookout.style.format.feature_extractor import FeatureExtractor
from lookout.style.format.rules import TrainableRules
from lookout.style.format.tests.test_analyzer import get_config


class RulesIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base = Path(__file__).parent
        # str() is needed for Python 3.5
        with lzma.open(str(base / "benchmark.js.xz"), mode="rt") as fin:
            contents = fin.read()
        with lzma.open(str(base / "benchmark.uast.xz")) as fin:
            uast = bblfsh.Node.FromString(fin.read())
        file = UnicodeFile(content=contents, uast=uast, path="", language="javascript")
        cls.files = [file]
        config = FormatAnalyzer._load_config(get_config())["train"]
        cls.extractor = FeatureExtractor(
            language="javascript", **config["javascript"]["feature_extractor"])

    def test_integration(self):
        res = self.extractor.extract_features(self.files)
        self.assertIsNotNone(res, "Failed to parse files.")
        X, y, _, = res
        train_X, test_X, train_y, test_y = \
            model_selection.train_test_split(X, y, random_state=1989)

        model = tree.DecisionTreeClassifier(min_samples_leaf=26, random_state=1989, max_depth=None,
                                            max_features="auto", min_samples_split=2)
        model.fit(train_X, train_y)
        rules = TrainableRules(base_model_name="sklearn.tree.DecisionTreeClassifier",
                               prune_branches_algorithms=[],
                               prune_attributes=False, min_samples_leaf=26, random_state=1989,
                               max_depth=None, max_features="auto", min_samples_split=2,
                               confidence_threshold=0)
        rules.fit(train_X, train_y)
        model_score_train = model.score(train_X, train_y)
        model_score_test = model.score(test_X, test_y)
        rules_score_train = rules.score(train_X, train_y)
        rules_score_test = rules.score(test_X, test_y)
        self.assertEqual(rules_score_train, model_score_train)
        self.assertEqual(rules_score_test, model_score_test)


if __name__ == "__main__":
    unittest.main()
