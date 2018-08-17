import os
import unittest

import numpy
import pandas
from sklearn import model_selection, tree, ensemble
from sklearn.exceptions import NotFittedError

from lookout.style.format.rules import Rules


def load_abalone_data(filepath=os.path.join(os.path.dirname(__file__), "abalone.data.xz")):
    column_names = ["sex", "length", "diameter", "height", "whole weight",
                    "shucked weight", "viscera weight", "shell weight", "rings"]
    data = pandas.read_csv(filepath, names=column_names)
    for label in "MFI":
        data[label] = data["sex"] == label
    del data["sex"]
    y = data.rings.values
    del data["rings"]
    x = data.values.astype(numpy.float32)
    interval = len(y) / 2
    ymap = {}
    accum = 0
    i = 0
    for val in range(y.min(), y.max() + 1):
        delta = (y == val).sum()
        accum += delta
        if accum > interval:
            accum = delta
            i += 1
        ymap[val] = i
    mapped_y = numpy.zeros(len(y), dtype=int)
    for i, v in enumerate(y):
        mapped_y[i] = ymap[v]

    return model_selection.train_test_split(x, mapped_y, random_state=1989)


class RulesTests(unittest.TestCase):
    def setUp(self):
        self.train_x, self.test_x, self.train_y, self.test_y = load_abalone_data()

    def test_set_unfitted_item(self):
        model = tree.DecisionTreeClassifier(min_samples_leaf=26, random_state=1989)
        with self.assertRaises(NotFittedError):
            Rules(model, prune_branches=False, prune_attributes=False)

    def test_tree_no_pruning(self):
        model = tree.DecisionTreeClassifier(min_samples_leaf=26, random_state=1989)
        model = model.fit(self.train_x, self.train_y)
        rules = Rules(model, prune_branches=False, prune_attributes=False)
        rules.fit(self.train_x, self.train_y)
        tree_score = model.score(self.train_x, self.train_y)
        rules_score = rules.score(self.train_x, self.train_y)
        self.assertAlmostEqual(tree_score, rules_score)

    def test_forest_no_pruning(self):
        model = ensemble.RandomForestClassifier(n_estimators=50, min_samples_leaf=26,
                                                random_state=1989)
        model = model.fit(self.train_x, self.train_y)
        rules = Rules(model, prune_branches=False, prune_attributes=False)
        rules.fit(self.train_x, self.train_y)
        forest_score = model.score(self.train_x, self.train_y)
        rules_score = rules.score(self.train_x, self.train_y)
        self.assertGreater(rules_score * 1.1, forest_score)

    def test_tree_attr_pruning(self):
        model = tree.DecisionTreeClassifier(min_samples_leaf=26, random_state=1989)
        model = model.fit(self.train_x, self.train_y)
        rules = Rules(model, prune_branches=False, prune_attributes=True)
        rules.fit(self.train_x, self.train_y)
        tree_score = model.score(self.test_x, self.test_y)
        rules_score = rules.score(self.test_x, self.test_y)
        self.assertGreater(rules_score, tree_score - 0.001)

    def test_prune_branches_top_down_greedy(self):
        model = tree.DecisionTreeClassifier(min_samples_leaf=26, random_state=1989)
        model = model.fit(self.train_x, self.train_y)

        def test_budget(budget):
            rules = Rules(model, prune_branches_algorithm="top-down-greedy", prune_branches=True,
                          prune_attributes=False, top_down_greedy_budget=(False, budget))
            rules.fit(self.train_x, self.train_y)
            return rules.score(self.train_x, self.train_y)
        scores = [test_budget(x) for x in numpy.linspace(0, 1, 10)]
        for a, b in zip(range(len(scores)), range(1, len(scores))):
            self.assertGreater(b, a - 0.00001)


if __name__ == "__main__":
    unittest.main()
