import os
import unittest

import numpy
import pandas
from scipy import sparse
from scipy.sparse import csr_matrix
from sklearn import ensemble, model_selection, tree
from sklearn.exceptions import NotFittedError
from sklearn.tree import _tree

from lookout.style.format.rules import TrainableRules


def load_abalone_data(filepath=os.path.join(os.path.dirname(__file__), "abalone.data.xz")):
    column_names = ["sex", "length", "diameter", "height", "whole weight",
                    "shucked weight", "viscera weight", "shell weight", "rings"]
    data = pandas.read_csv(filepath, names=column_names)
    for label in "MFI":
        data[label] = data["sex"] == label
    del data["sex"]
    y = data.rings.values
    del data["rings"]
    x = csr_matrix(data.values.astype(numpy.float32))
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
    return model_selection.train_test_split(x, mapped_y, random_state=1989), x, mapped_y


class RulesTests(unittest.TestCase):
    def setUp(self):
        (self.train_x, self.test_x, self.train_y, self.test_y), self.x, self.y \
            = load_abalone_data()

    def test_predict_unfitted(self):
        rules = TrainableRules(base_model_name="sklearn.tree.DecisionTreeClassifier",
                               prune_branches_algorithms=[],
                               prune_attributes=False,
                               confidence_threshold=0)
        with self.assertRaises(NotFittedError):
            rules.predict(self.test_x)

    def test_tree_no_pruning(self):
        model = tree.DecisionTreeClassifier(min_samples_leaf=26, random_state=1989)
        model = model.fit(self.train_x, self.train_y)
        rules = TrainableRules(base_model_name="sklearn.tree.DecisionTreeClassifier",
                               prune_branches_algorithms=[], confidence_threshold=0,
                               prune_attributes=False, min_samples_leaf=26, random_state=1989)
        rules.fit(self.train_x, self.train_y)
        tree_score = model.score(self.train_x, self.train_y)
        rules_score = rules.score(self.train_x, self.train_y)
        self.assertGreater(rules_score * 1.1, tree_score)

    def test_forest_no_pruning(self):
        model = ensemble.RandomForestClassifier(n_estimators=50, min_samples_leaf=26,
                                                random_state=1989)
        model = model.fit(self.train_x, self.train_y)
        rules = TrainableRules(base_model_name="sklearn.ensemble.RandomForestClassifier",
                               prune_branches_algorithms=[], prune_attributes=False,
                               n_estimators=50, min_samples_leaf=26, random_state=1989,
                               confidence_threshold=0)
        rules.fit(self.train_x, self.train_y)
        forest_score = model.score(self.train_x, self.train_y)
        rules_score = rules.score(self.train_x, self.train_y)
        self.assertGreater(rules_score * 1.1, forest_score)

    def test_tree_attr_pruning(self):
        model = tree.DecisionTreeClassifier(min_samples_leaf=26, random_state=1989)
        model = model.fit(self.train_x, self.train_y)
        rules = TrainableRules(base_model_name="sklearn.tree.DecisionTreeClassifier",
                               prune_branches_algorithms=[],
                               prune_attributes=True, min_samples_leaf=26, random_state=1989,
                               confidence_threshold=0)
        rules.fit(self.train_x, self.train_y)
        tree_score = model.score(self.test_x, self.test_y)
        rules_score = rules.score(self.test_x, self.test_y)
        self.assertGreater(rules_score * 1.1, tree_score)

    def test_prune_branches_top_down_greedy(self):
        def test_budget(budget):
            rules = TrainableRules(base_model_name="sklearn.tree.DecisionTreeClassifier",
                                   prune_branches_algorithms=["top-down-greedy"],
                                   prune_attributes=False, top_down_greedy_budget=(False, budget),
                                   random_state=1989, confidence_threshold=0)
            rules.fit(self.train_x, self.train_y)
            return rules.score(self.train_x, self.train_y)
        scores = [test_budget(x) for x in numpy.linspace(0, 1, 10)]
        for a, b in zip(range(len(scores)), range(1, len(scores))):
            self.assertGreater(b, a - 0.00001)

    def test_reduced_error_prune(self):
        LEAF = _tree.TREE_LEAF
        UNDEFINED = _tree.TREE_UNDEFINED

        class FakeFeature:
            def __getitem__(self, item):
                pass

            def __setitem__(self, key, value):
                pass

        class FakeTree:
            def __init__(self, *args):
                self.children_left = numpy.array(args[0])
                self.children_right = numpy.array(args[1])
                self.feature = FakeFeature()
                self.value = numpy.array([
                    [[50, 100]],
                    [[45, 50]],
                    [[5, 40]],
                    [[40, 10]],
                    [[2, 20]],
                    [[3, 20]],
                    [[20, 5]],
                    [[20, 5]],
                    [[1, 19]],
                    [[2, 1]],
                    [[5, 50]],
                ])

        class FakeModel:
            def __init__(self):
                self.tree_ = FakeTree(
                    [1, 2, 4, 6, 8, LEAF, LEAF, LEAF, LEAF, LEAF, LEAF],
                    [10, 3, 5, 7, 9, LEAF, LEAF, LEAF, LEAF, LEAF, LEAF],
                )
                self.classes_ = [0, 1]

            def decision_path(self, X):
                return sparse.csr_matrix([
                    [1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0],
                    [1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0],
                    [1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0],
                    [1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0],
                    [1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                ])

            def predict(self, X):
                return numpy.array([1, 0, 1, 0, 0, 1])

        class FakeX:
            shape = [6]

        model = FakeModel()
        pruned_model = TrainableRules._prune_reduced_error(
            model, FakeX, numpy.array([1, 1, 1, 0, 0, 1]))
        self.assertEqual(list(pruned_model.tree_.children_left),
                         [1, 2, LEAF, LEAF, UNDEFINED, UNDEFINED, UNDEFINED, UNDEFINED, UNDEFINED,
                          UNDEFINED, LEAF])
        self.assertEqual(list(pruned_model.tree_.children_right),
                         [10, 3, LEAF, LEAF, UNDEFINED, UNDEFINED, UNDEFINED, UNDEFINED, UNDEFINED,
                          UNDEFINED, LEAF])

    def test_rules_estimator(self):
        estimator = TrainableRules(base_model_name="sklearn.tree.DecisionTreeClassifier",
                                   prune_branches_algorithms=[], prune_attributes=False,
                                   confidence_threshold=0)
        scores = model_selection.cross_val_score(estimator, self.x, self.y)
        score = sum(scores) / len(scores)
        self.assertGreater(score, .5)

    def test_predict_winner_indices(self):
        rules = TrainableRules(base_model_name="sklearn.tree.DecisionTreeClassifier",
                               prune_branches_algorithms=[],
                               prune_attributes=False, min_samples_leaf=26, random_state=1989,
                               confidence_threshold=0)
        rules.fit(self.train_x, self.train_y)
        pred_y, winners = rules.rules.apply(self.train_x, return_winner_indices=True)
        for ycls, w in zip(pred_y, winners):
            self.assertEqual(ycls, rules.rules.rules[w].stats.cls)


if __name__ == "__main__":
    unittest.main()
