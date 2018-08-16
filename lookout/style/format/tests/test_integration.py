import lzma
import unittest
from pathlib import Path

from sklearn import model_selection, tree

import bblfsh
from lookout.core.api.service_data_pb2 import File
from lookout.style.format.features import FeatureExtractor
from lookout.style.format.rules import Rules


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base = Path(__file__).parent
        # str() is needed for Python 3.5
        with lzma.open(str(base / "benchmark.js.xz"), mode="rt") as fin:
            contents = fin.read()
        with lzma.open(str(base / "benchmark.uast.xz")) as fin:
            uast = bblfsh.Node.FromString(fin.read())
        file = File(content=bytes(contents, 'utf-8'),
                    uast=uast)
        cls.files = [file]
        cls.extractor = FeatureExtractor("javascript",
                                         parents_depth=2,
                                         siblings_window=5)

    def test_integration(self):
        X, y = self.extractor.extract_features(self.files)
        train_X, test_X, train_y, test_y = \
            model_selection.train_test_split(X, y, random_state=1989)

        model = tree.DecisionTreeClassifier(min_samples_leaf=26, random_state=1989)
        model.fit(train_X, train_y)
        rules = Rules(model, prune_branches=False, prune_attributes=False)
        rules.fit(train_X, train_y)
        model_score_train = model.score(train_X, train_y)
        model_score_test = model.score(test_X, test_y)
        rules_score_train = rules.score(train_X, train_y)
        rules_score_test = rules.score(test_X, test_y)
        self.assertEqual(model_score_train, rules_score_train)
        self.assertEqual(model_score_test, rules_score_test)


if __name__ == "__main__":
    unittest.main()
