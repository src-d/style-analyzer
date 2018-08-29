from collections import defaultdict
from copy import deepcopy
import functools
import importlib
import logging
from typing import Any, Dict, Iterable, List, Mapping, NamedTuple, Sequence, Set, Tuple, Union, \
    Type

import numpy
from numpy import count_nonzero
from scipy.stats import fisher_exact
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import _tree as Tree, DecisionTreeClassifier

RuleAttribute = NamedTuple(
    "RuleAttribute", (("feature", int), ("cmp", bool), ("threshold", float)))
"""
`feature` is the feature taken for comparison
`cmp` is the comparison type: True is "x > v", False is "x <= v"
`threshold` is "v", the threshold value
"""

RuleStats = NamedTuple("RuleStats", (("cls", int), ("conf", float)))
"""
`cls` is the predicted class
`conf` is the rule confidence \\in [0, 1], "1" means super confident
"""

Rule = NamedTuple("RuleType", (("attrs", Tuple[RuleAttribute, ...]), ("stats", RuleStats)))


class Rules:
    CompiledNegatedRules = NamedTuple("CompiledNegatedRules", (
        ("false", numpy.ndarray), ("true", numpy.ndarray)))
    """
    Each ndarray contains the rule indices which are **false** given
    the corresponding feature, threshold value and the comparison type ("false" and "true").
    """
    CompiledFeatureRules = NamedTuple("CompiledRule", (
        ("values", numpy.ndarray), ("negated", Tuple[CompiledNegatedRules, ...])))

    CompiledRulesType = Dict[int, CompiledFeatureRules]

    _log = logging.getLogger("Rules")

    def __init__(self, rules: List[Rule], origin: Mapping[str, Any]):
        """
        Initializes the rules so that it is possible to call predict() afterwards.

        :param rules: the list of rules to assign.
        :param origin: the dictionary of parameters used to train the rules.
        """
        super().__init__()
        assert rules is not None, "rules may not be None"
        self._rules = rules
        self._compiled = self._compile(rules)
        self._origin = origin

    def __str__(self):
        return "%d rules, avg.len. %.1f" % (len(self._rules), self.avg_rule_len)

    def __len__(self):
        return len(self._rules)

    def predict(self, X: numpy.ndarray, return_winner_indices=False
                ) -> Union[numpy.ndarray, Tuple[numpy.ndarray, numpy.ndarray]]:
        """
        Evaluates the rules against the given features.

        :param X: input features.
        :param return_winner_indices: whether to return the winning rule index for each sample.
        :return: array of the same length as X with predictions or tuple of two arrays of the same\
                 length as X containing (predictions, winner rule indices).
        """
        self._log.debug("predicting %d samples using %d rules", len(X), len(self._rules))
        rules = self._rules
        _compute_triggered = self._compute_triggered
        prediction = numpy.zeros(len(X), dtype=numpy.int32)
        if return_winner_indices:
            winner_indices = numpy.zeros(len(X), dtype=numpy.int32)
        for xi, x in enumerate(X):
            ris = _compute_triggered(self._compiled, rules, x)
            if len(ris) == 0:
                # self._log.warning("no rule!")
                continue
            if len(ris) > 1:
                confs = numpy.zeros(len(ris), dtype=numpy.float32)
                for i, ri in enumerate(ris):
                    confs[i] = rules[ri].stats.conf
                winner_index = ris[numpy.argmax(confs)]
            else:
                winner_index = ris[0]
            prediction[xi] = rules[winner_index].stats.cls
            if return_winner_indices:
                winner_indices[xi] = winner_index
        if return_winner_indices:
            return prediction, winner_indices
        return prediction

    @property
    def rules(self) -> List[Rule]:
        return self._rules

    @property
    def origin(self) -> Mapping[str, Any]:
        return self._origin

    @property
    def avg_rule_len(self) -> float:
        if not self._rules:
            return 0
        return sum(len(r.attrs) for r in self._rules) / len(self._rules)

    @classmethod
    def _compile(cls, rules: Sequence[Rule]) -> CompiledRulesType:
        cls._log.debug("compiling %d rules", len(rules))
        attrs = defaultdict(lambda: defaultdict(lambda: [[], []]))
        for i, (branch, _) in enumerate(rules):
            for rule in branch:
                attrs[rule.feature][rule.threshold][int(rule.cmp)].append(i)
        compiled_attrs = {}
        for key, attr in attrs.items():
            vals = sorted(attr)
            false_rules = set()
            true_rules = set()
            vr = [[None, None] for _ in vals]
            for i in range(len(vals)):
                false_rules.update(attr[vals[i]][False])
                true_rules.update(attr[vals[len(vals) - i - 1]][True])
                vr[i][False] = numpy.array(sorted(false_rules))
                vr[len(vr) - i - 1][True] = numpy.array(sorted(true_rules))
            compiled_attrs[key] = cls.CompiledFeatureRules(
                numpy.array(vals, dtype=numpy.float32),
                tuple(cls.CompiledNegatedRules(*v) for v in vr))
        return compiled_attrs

    @classmethod
    def _compute_triggered(cls, compiled_rules: CompiledRulesType,
                           rules: Sequence[Rule], x: numpy.ndarray
                           ) -> numpy.ndarray:
        searchsorted = numpy.searchsorted
        triggered = numpy.full(len(rules), 0xff, dtype=numpy.int8)
        for i, v in enumerate(x):
            try:
                vals, arules = compiled_rules[i]
            except KeyError:
                continue
            border = searchsorted(vals, v)
            if border > 0:
                indices = arules[border - 1][False]
                if len(indices):
                    triggered[indices] = 0
            if border < len(arules):
                indices = arules[border][True]
                if len(indices):
                    triggered[indices] = 0
        return numpy.nonzero(triggered)[0]


TopDownGreedyBudget = NamedTuple("TopDownGreedyBudget", (("absolute", bool),
                                                         ("value", Union[float, int])))


class TrainableRules(BaseEstimator, ClassifierMixin):

    _log = logging.getLogger("TrainableRules")

    def __init__(self, base_model_name: str = "sklearn.tree.DecisionTreeClassifier",
                 prune_branches_algorithms=("reduced-error", "top-down-greedy"),
                 top_down_greedy_budget=TopDownGreedyBudget(False, 1.0), prune_attributes=True,
                 uncertain_attributes=True, prune_dataset_ratio=.2, n_estimators=10,
                 max_depth=None, max_features=None, min_samples_leaf=1, min_samples_split=2,
                 random_state=42):
        """
        Initializes a new instance of Rules class.

        :param base_model_name: fully qualified type name of the base model to train. \
                                Must be either "sklearn.tree.DecisionTreeClassifier" or \
                               "sklearn.ensemble.RandomForestClassifier".
        :param prune_branches_algorithms: branch pruning algorithms to use.
        :param top_down_greedy_budget: how many the branches to leave, either a floating point \
                                       number from 0 to 1 or the exact quantity.
        :param prune_attributes: indicates whether to remove useless parts of rules.
        :param uncertain_attributes: indicates whether to **retain** parts of rules with low \
                                     certainty (see "Generating Production Rules From Decision \
                                     Trees" by J.R. Quinlan).
        :param prune_base_model: indicates whether to prune base_model via reduced error pruning \
                                 algorithm. Available for DecisionTreeClassifier model only.
        """
        super().__init__()
        self.base_model_name = base_model_name
        self.prune_branches_algorithms = prune_branches_algorithms
        self.top_down_greedy_budget = top_down_greedy_budget
        self.prune_attributes = prune_attributes
        self.uncertain_attributes = uncertain_attributes
        self.prune_dataset_ratio = prune_dataset_ratio
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self._rules = None  # type: Rules

    def fit(self, X: numpy.ndarray, y: numpy.ndarray) -> "TrainableRules":
        """
        Trains the rules using the base tree model and the samples (X, y). If `base_model` is
        already fitted, the samples may be different from the ones that were used.

        :param X: input features.
        :param y: input labels - the same length as X.
        :return: self
        """
        base_model = self._base_model_class(max_depth=self.max_depth,
                                            max_features=self.max_features,
                                            min_samples_leaf=self.min_samples_leaf,
                                            min_samples_split=self.min_samples_split,
                                            random_state=self.random_state)

        if self.prune_branches_algorithms or self.prune_attributes:
            X_train, X_prune, y_train, y_prune = train_test_split(
                X, y, test_size=self.prune_dataset_ratio, random_state=42)
        else:
            X_train, y_train = X, y
        base_model.fit(X_train, y_train)

        if isinstance(base_model, DecisionTreeClassifier):
            if "reduced-error" in self.prune_branches_algorithms:
                old = (count_nonzero(base_model.tree_.children_left == Tree.TREE_LEAF)
                       + count_nonzero(base_model.tree_.children_right == Tree.TREE_LEAF))
                base_model = self._prune_reduced_error(base_model, X_prune, y_prune)
                new = (count_nonzero(base_model.tree_.children_left == Tree.TREE_LEAF)
                       + count_nonzero(base_model.tree_.children_right == Tree.TREE_LEAF))
                self._log.debug("pruned %d/%d branches w/ reduced error pruning", old - new, old)
            rules, leaf2rule_dict = self._tree_to_rules(base_model)
            leaf2rule = [leaf2rule_dict]
        else:
            rules = []
            offset = 0
            leaf2rule = []
            old, new = 0, 0
            for i, estimator in enumerate(base_model.estimators_):
                if "reduced-error" in self.prune_branches_algorithms:
                    old += (count_nonzero(estimator.tree_.children_left == Tree.TREE_LEAF)
                            + count_nonzero(estimator.tree_.children_right == Tree.TREE_LEAF))
                    estimator = self._prune_reduced_error(estimator, X_prune, y_prune)
                    new += (count_nonzero(estimator.tree_.children_left == Tree.TREE_LEAF)
                            + count_nonzero(estimator.tree_.children_right == Tree.TREE_LEAF))
                rules_partial, leaf2rule_partial = self._tree_to_rules(estimator, offset=offset)
                offset += len(rules_partial)
                leaf2rule.append(leaf2rule_partial)
                rules.extend(rules_partial)
            if "reduced-error" in self.prune_branches_algorithms:
                self._log.debug("pruned %d/%d branches w/ reduced error pruning", old - new, old)

        def count_attrs():
            return sum(len(r.attrs) for r in rules)

        old = count_attrs()
        rules = self._merge_rules(rules)
        self._log.debug("merged %d/%d attributes", old - count_attrs(), old)
        if "top-down-greedy" in self.prune_branches_algorithms:
            old = len(rules)
            rules = self._prune_branches_top_down_greedy(
                base_model, rules, X_prune, y_prune, leaf2rule, self.top_down_greedy_budget)
            self._log.debug("pruned %d/%d rules w/ greedy pruning", old - len(rules), old)
        if self.prune_attributes:
            old = count_attrs()
            rules = self._prune_attributes(rules, X_prune, y_prune, not self.uncertain_attributes)
            self._log.debug("pruned %d/%d attributes", old - count_attrs(), old)
        self._rules = Rules(rules, self._sanitize_params(self.get_params(False)))
        return self

    @property
    def base_model_name(self) -> str:
        return self._base_model_name

    @base_model_name.setter
    def base_model_name(self, value: Union[str, Type[DecisionTreeClassifier],
                                           Type[RandomForestClassifier]]):
        if isinstance(value, str):
            self._base_model_name = value
            base_model_module_name, base_model_class_name = value.rsplit(".", 1)
            base_model_module = importlib.import_module(base_model_module_name)
            value = getattr(base_model_module, base_model_class_name)
        else:
            self._base_model_name = "%s.%s" % (value.__module__, value.__name__)
        if not issubclass(value, (DecisionTreeClassifier, RandomForestClassifier)):
            raise TypeError("%s base model type is not allowed" % value)
        self._base_model_class = value

    @property
    def fitted(self):
        return self._rules is not None

    def _check_fitted(func):
        @functools.wraps(func)
        def wrapped_check_fitted(self: "TrainableRules", *args, **kwargs):
            if not self.fitted:
                raise NotFittedError
            return func(self, *args, **kwargs)

        return wrapped_check_fitted

    @_check_fitted
    def predict(self, X: numpy.ndarray) -> numpy.ndarray:
        """
        Evaluates the rules against the given features.

        :param X: input features.
        :return: array of the same length as X with predictions.
        """
        return self._rules.predict(X)

    _check_fitted = staticmethod(_check_fitted)

    @property
    def rules(self) -> Rules:
        return self._rules

    @classmethod
    def _tree_to_rules(cls, tree: DecisionTreeClassifier, offset: int = 0
                       ) -> Tuple[List[Rule], Mapping[int, int]]:
        """
        Converts the sklearn's decision tree to the set of rules.
        Each rule is a branch in the tree.

        :param tree: input decision tree.
        :return: list of extracted rules.
        """
        tree_ = tree.tree_
        feature_names = [i if i != Tree.TREE_UNDEFINED else None for i in tree_.feature]
        queue = [(0, tuple())]
        rules = []
        leaf2rule = {}
        while queue:
            node, path = queue.pop()
            if tree_.feature[node] != Tree.TREE_UNDEFINED:
                name = feature_names[node]
                threshold = tree_.threshold[node]
                queue.append(
                    (tree_.children_left[node], path + (RuleAttribute(name, False, threshold),)))
                queue.append(
                    (tree_.children_right[node], path + (RuleAttribute(name, True, threshold),)))
            else:
                freqs = tree_.value[node][0]
                # why -0.5? See the papers mentioned in _prune_attributes()
                conf = (freqs.max() - 0.5) / freqs.sum()
                leaf2rule[node] = len(rules) + offset
                rules.append(Rule(path, RuleStats(tree.classes_[numpy.argmax(freqs)], conf)))
        return rules, leaf2rule

    @classmethod
    def _merge_rules(cls, rules: List[Rule]) -> List[Rule]:
        new_rules = []
        for rule, stats in rules:
            min_vals = {}
            max_vals = {}
            flags = defaultdict(int)
            for name, cmp, val in rule:
                if cmp:
                    min_vals[name] = max(min_vals.get(name, val), val)
                    flags[name] |= 1
                else:
                    max_vals[name] = min(max_vals.get(name, val), val)
                    flags[name] |= 2
            new_rule = []
            for key, bits in sorted(flags.items()):
                if bits & 2:
                    new_rule.append(RuleAttribute(key, False, max_vals[key]))
                if bits & 1:
                    new_rule.append(RuleAttribute(key, True, min_vals[key]))
            new_rules.append(Rule(tuple(new_rule), stats))
        return new_rules

    @classmethod
    def _prune_reduced_error(cls, model: DecisionTreeClassifier, X: numpy.array, y: numpy.array,
                             step_score_drop: float = 0,
                             max_score_drop: float = 0) -> DecisionTreeClassifier:
        def _prune_tree(tree, node_to_prune):
            child_left = tree.children_left[node_to_prune]
            child_right = tree.children_right[node_to_prune]
            tree.children_left[child_left] = Tree.TREE_UNDEFINED
            tree.children_left[child_right] = Tree.TREE_UNDEFINED
            tree.children_right[child_left] = Tree.TREE_UNDEFINED
            tree.children_right[child_right] = Tree.TREE_UNDEFINED
            tree.children_left[node_to_prune] = Tree.TREE_LEAF
            tree.children_right[node_to_prune] = Tree.TREE_LEAF
            tree.feature[node_to_prune] = Tree.TREE_UNDEFINED

        model = deepcopy(model)
        tree = model.tree_
        changes = True
        checked = set()
        parents = {x: i for i, x in enumerate(tree.children_left) if x != Tree.TREE_LEAF}
        parents.update({x: i for i, x in enumerate(tree.children_right) if x != Tree.TREE_LEAF})
        leaves = list(numpy.where(tree.children_left == Tree.TREE_LEAF)[0])
        decision_path = {leaf: d.nonzero()[1] for leaf, d in
                         zip(leaves, model.decision_path(X).T[leaves])}
        y_predicted = model.predict(X)
        init_score = current_score = accuracy_score(y, y_predicted)
        while changes:
            changes = False
            for leaf_index, leaf1 in enumerate(leaves):
                if leaf1 not in parents:
                    continue
                parent = parents[leaf1]
                if parent in checked:
                    continue
                leaf2 = tree.children_right[parent]
                leaf2 = leaf2 if leaf2 != leaf1 else tree.children_left[parent]
                if tree.children_left[leaf2] != Tree.TREE_LEAF or \
                        tree.children_right[leaf2] != Tree.TREE_LEAF:
                    continue

                data_leaf1_index = decision_path[leaf1]
                data_leaf2_index = decision_path[leaf2]
                data_parent_index = numpy.concatenate((data_leaf1_index, data_leaf2_index))
                y_predicted_leaf1 = model.classes_[numpy.argmax(tree.value[leaf1, 0, :])]
                y_predicted_leaf2 = model.classes_[numpy.argmax(tree.value[leaf2, 0, :])]
                new_y = model.classes_[numpy.argmax(tree.value[parent, 0, :])]

                score_delta = (numpy.sum(new_y == y[data_parent_index]) -
                               numpy.sum(y_predicted_leaf1 == y[data_leaf1_index]) -
                               numpy.sum(y_predicted_leaf2 == y[data_leaf2_index])) \
                    / X.shape[0]

                if init_score != 0 and score_delta / init_score < max_score_drop or \
                        current_score != 0 and score_delta / current_score < step_score_drop:
                    checked.add(parent)
                    continue
                else:
                    current_score += score_delta
                    leaves.remove(leaf2)
                    leaves[leaf_index] = parent
                    _prune_tree(tree, parent)
                    y_predicted[data_parent_index] = new_y
                    del decision_path[leaf1], decision_path[leaf2]
                    decision_path[parent] = data_parent_index
                    changes = True
                    break
        return model

    def _build_instances_index(
            self, base_model: Union[DecisionTreeClassifier, RandomForestClassifier],
            X: numpy.ndarray, leaf2rule: Sequence[Mapping[int, int]]) -> Dict[int, Set[int]]:
        instances_index = defaultdict(set)

        if isinstance(base_model, DecisionTreeClassifier):
            leaves = base_model.apply(X)  # ndim = 1
            for i, leaf in enumerate(leaves):
                instances_index[leaf2rule[0][leaf]].add(i)
        else:
            leaves = base_model.apply(X)  # ndim = 2
            for i, col in enumerate(leaves):
                for leaf, l2r in zip(col, leaf2rule):
                    instances_index[l2r[leaf]].add(i)
        return instances_index

    def _prune_branches_top_down_greedy(
            self, base_model: Union[DecisionTreeClassifier, RandomForestClassifier],
            rules: Sequence[Rule], X: numpy.ndarray, Y: numpy.ndarray,
            leaf2rule: Sequence[Mapping[int, int]], budget: TopDownGreedyBudget) -> List[Rule]:
        absolute, value = budget
        if absolute:
            assert isinstance(value, int)
            n_budget = max(0, min(value, len(rules)))
        else:
            assert value >= 0 and value <= 1
            n_budget = int(max(0, min(value * len(rules), len(rules))))
        instances_index = self._build_instances_index(base_model, X, leaf2rule)
        confs_index = numpy.full(X.shape[0], -1.)
        clss_index = numpy.full(X.shape[0], -1)
        candidate_rules = set(range(len(rules)))
        selected_rules = set()
        for iteration in range(n_budget):
            scores = []
            for rule_id in candidate_rules:
                triggered_instances = instances_index[rule_id]
                matched_delta = 0
                stats = rules[rule_id].stats
                for triggered_instance in triggered_instances:
                    if (stats.conf > confs_index[triggered_instance]
                            and stats.cls != clss_index[triggered_instance]):
                        if Y[triggered_instance] == clss_index[triggered_instance]:
                            matched_delta -= 1
                        elif Y[triggered_instance] == stats.cls:
                            matched_delta += 1
                scores.append((matched_delta, rule_id))
            best_matched_delta, best_rule_id = max(scores)
            for triggered_instance in instances_index[best_rule_id]:
                stats = rules[best_rule_id].stats
                confs_index[triggered_instance] = rules[rule_id].stats.conf
                clss_index[triggered_instance] = stats.cls
            candidate_rules.remove(best_rule_id)
            selected_rules.add(best_rule_id)
        return [rules[rule_id] for rule_id in selected_rules]

    @classmethod
    def _prune_attributes(cls, rules: Iterable[Rule],
                          X: numpy.ndarray, Y: numpy.ndarray,
                          prune_uncertain: bool) -> List[Rule]:
        """
        Removes the attribute comparisons which do not influence the rule decision.

        Based on:

        "Generating Production Rules From Decision Trees" by J. R. Quinlan.
        https://www.ijcai.org/Proceedings/87-1/Papers/063.pdf

        "Simplifying Decision Trees" by J. R. Quinlan.
        https://dspace.mit.edu/bitstream/handle/1721.1/6453/AIM-930.pdf

        :param rules: list of rules to simplify.
        :param X: input features, used to exclude the irrelevant attributes.
        :param Y: input labels.
        :return: new list of simplified rules.
        """

        def confidence(v, not_v):
            return (v - 0.5) / (v + not_v)

        new_rules = []
        intervals = {}
        attrs = defaultdict(set)
        for i, (branch, _) in enumerate(rules):
            for rule in branch:
                attrs[rule.feature].add(rule.threshold)
        for key, vals in attrs.items():
            attrs[key] = numpy.array(sorted(vals))
            intervals[key] = [defaultdict(int) for _ in range(len(vals) + 1)]
        searchsorted = numpy.searchsorted
        for i, (x, y) in enumerate(zip(X, Y)):
            for attr, val in enumerate(x):
                interval = intervals.get(attr)
                if interval is not None:
                    interval[searchsorted(attrs[attr], val)][y] += 1
        for key, vals in attrs.items():
            attrs[key] = {v: i for i, v in enumerate(vals)}
        for vals in intervals.values():
            for vec in vals:
                vec[-1] = sum(vec.values())
        for rule, stats in rules:
            c = stats.cls
            new_verbs = []
            for feature, cmp, thr in rule:
                table = numpy.zeros((2, 2), dtype=numpy.int32)
                for i, interval in enumerate(intervals[feature]):
                    row = int((i <= attrs[feature][thr]) == cmp)
                    num_same_cls = interval[c]
                    table[row, 0] += num_same_cls
                    table[row, 1] += interval[-1] - num_same_cls
                if prune_uncertain:
                    if confidence(table[0, 0] + table[1, 0], table[0, 1] + table[1, 1]) \
                            >= confidence(table[0, 0], table[0, 1]):
                        continue
                _, p = fisher_exact(table)
                if p < 0.01:
                    new_verbs.append(RuleAttribute(feature, cmp, thr))
            if new_verbs:
                new_rules.append(Rule(tuple(new_verbs), stats))
        return new_rules

    @staticmethod
    def _sanitize_params(params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalizes the parameters from get_params() so that they are suitable for serialization.

        :param params: dict from get_params().
        :return: normalized dict.
        """
        sanitized = {}
        for k, v in params.items():
            if isinstance(v, tuple):
                # fix namedtuple-s in ASDF
                v = list(v)
            sanitized[k] = v
        return sanitized
