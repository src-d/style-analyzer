from collections import defaultdict
from itertools import chain
import functools
import logging
from typing import Union, List, Tuple, Dict, Iterable, NamedTuple, Sequence, Set

import modelforge
import numpy
from sklearn.exceptions import NotFittedError
from scipy.stats import fisher_exact
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier, _tree as Tree
from sklearn.ensemble import RandomForestClassifier


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


class FormatModel(modelforge.Model):
    def construct(self, rules: Iterable[Rule]):
        pass

    def dump(self) -> str:
        pass

    def _generate_tree(self) -> dict:
        pass

    def _load_tree(self, tree: dict) -> None:
        pass


class Rules(BaseEstimator, ClassifierMixin):
    CompiledNegatedRules = NamedTuple("CompiledNegatedRules", (
        ("false", numpy.ndarray), ("true", numpy.ndarray)))
    """
    Each ndarray contains the rule indices which are **false** given
    the corresponding feature, threshold value and the comparison type ("false" and "true").
    """
    CompiledFeatureRules = NamedTuple("CompiledRule", (
        ("values", numpy.ndarray), ("negated", Tuple[CompiledNegatedRules, ...])))

    CompiledRulesType = Dict[int, CompiledFeatureRules]

    TopDownGreedyBudget = NamedTuple("TopDownGreedyBudget", (
        ("absolute", bool), ("value", Union[float, int])))

    log = logging.getLogger("Rules")

    def __init__(self,
                 base_model: Union[DecisionTreeClassifier, RandomForestClassifier],
                 prune_branches=True, prune_branches_algorithm="top-down-greedy",
                 top_down_greedy_budget=1.0, prune_attributes=True,
                 uncertain_attributes=True):
        """
        Initializes a new instance of Rules class.

        :param base_model: trained decision tree or random forest. \
                           The rules will be extracted from it.
        :param prune_branches: indicates whether to remove useless rules.
        :param prune_attributes: indicates whether to remove useless parts of rules.
        :param uncertain_attributes: indicates whether to **retain** parts of rules with low \
                                     certainty (see "Generating Production Rules From Decision
                                     Trees" by J.R. Quinlan).
        """
        self.base_model = base_model
        self.prune_branches = prune_branches
        self.prune_branches_algorithm = prune_branches_algorithm
        self.top_down_greedy_budget = top_down_greedy_budget
        self.prune_attributes = prune_attributes
        self.uncertain_attributes = uncertain_attributes
        self._cache = None, None  # type: Tuple[Rules.CompiledRulesType, List[Rule]]

    @property
    def base_model(self) -> Union[DecisionTreeClassifier, RandomForestClassifier]:
        return self._base_model

    @base_model.setter
    def base_model(self, value: Union[DecisionTreeClassifier, RandomForestClassifier]):
        if not isinstance(value, (DecisionTreeClassifier, RandomForestClassifier)):
            raise TypeError("base_model must be an instance of DecisionTreeClassifier or "
                            "RandomForestClassifier")
        self._base_model = value

    def fit(self, X: numpy.ndarray, y: numpy.ndarray) -> "Rules":
        """
        Trains the rules using the base tree model and the samples (X, y). The samples may be
        different from the ones the base model was trained on (actually, they should).

        :param X: input features.
        :param y: input labels - the same length as X.
        :return: self
        """
        if isinstance(self.base_model, DecisionTreeClassifier):
            rules = self._tree_to_rules(self.base_model)
        else:
            rules = list(chain.from_iterable(self._tree_to_rules(tree)
                                             for tree in self.base_model.estimators_))

        def count_attrs():
            return sum(len(r.attrs) for r in rules)

        self.log.debug("Initial number of rules: %d", len(rules))
        self.log.debug("Initial number of attributes: %d", count_attrs())
        rules = self._merge_rules(rules)
        self.log.debug("Merged number of attributes: %d", count_attrs())
        if self.prune_branches:
            if self.prune_branches_algorithm == "top-down-greedy":
                rules = self._prune_branches_top_down_greedy(rules, X, y,
                                                             self.top_down_greedy_budget)
            else:
                rules = self._prune_branches(rules, X, y)
        if self.prune_attributes:
            rules = self._prune_attributes(rules, X, y, not self.uncertain_attributes)
            self.log.debug("Pruned number of attributes (2): %d", len(rules))
            self.log.debug("Pruned number of attributes: %d", count_attrs())
        self._cache = self._compile_rules(rules), rules
        return self

    def _check_fitted(func):
        @functools.wraps(func)
        def wrapped_check_fitted(self: "Rules", *args, **kwargs):
            if None in self._cache:
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
        compiled, rules = self._cache
        prediction = numpy.zeros(len(X), dtype=int)
        for xi, x in enumerate(X):
            ris = self._compute_triggered(compiled, rules, x)
            if len(ris) == 0:
                # self.log.warning("no rule!")
                continue
            if len(ris) > 1:
                confs = numpy.zeros(len(ris), dtype=numpy.float32)
                for i, ri in enumerate(ris):
                    confs[i] = rules[ri].stats.conf
                winner = rules[ris[numpy.argmax(confs)]].stats.cls
            else:
                winner = rules[ris[0]].stats.cls
            prediction[xi] = winner
        return prediction

    @_check_fitted
    def to_modelforge(self) -> FormatModel:
        pass

    _check_fitted = staticmethod(_check_fitted)

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

    @classmethod
    def _tree_to_rules(cls, tree: DecisionTreeClassifier) -> List[Rule]:
        """
        Converts the sklearn's decision tree to the set of rules.
        Each rule is a branch in the tree.

        :param tree: input decision tree.
        :return: list of extracted rules.
        """
        tree = tree.tree_
        feature_names = [i if i != Tree.TREE_UNDEFINED else None for i in tree.feature]
        queue = [(0, tuple())]
        rules = []
        while queue:
            node, path = queue.pop()
            if tree.feature[node] != Tree.TREE_UNDEFINED:
                name = feature_names[node]
                threshold = tree.threshold[node]
                queue.append(
                    (tree.children_left[node], path + (RuleAttribute(name, False, threshold),)))
                queue.append(
                    (tree.children_right[node], path + (RuleAttribute(name, True, threshold),)))
            else:
                freqs = tree.value[node][0]
                # why -0.5? See the papers mentioned in _prune_attributes()
                conf = (freqs.max() - 0.5) / freqs.sum()
                rules.append(Rule(path, RuleStats(numpy.argmax(freqs), conf)))
        return rules

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
    def _compile_rules(cls, rules: Sequence[Rule]) -> CompiledRulesType:
        attrs = defaultdict(lambda: defaultdict(lambda: [[], []]))
        for i, (branch, _) in enumerate(rules):
            for rule in branch:
                attrs[rule.feature][rule.threshold][rule.cmp].append(i)
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
    def _prune_branches(cls, rules: Sequence[Rule],
                        X: numpy.ndarray, Y: numpy.ndarray) -> List[Rule]:
        # TODO(vmarkovtsev): implement this function
        return rules

    @classmethod
    def _build_instances_index(cls, rules: Sequence[Rule],
                               X: numpy.ndarray) -> Dict[int, Set[int]]:
        instances_index = defaultdict(set)
        compiled = cls._compile_rules(rules)
        for xi, x in enumerate(X):
            for triggered_rule in cls._compute_triggered(compiled, rules, x):
                instances_index[triggered_rule].add(xi)
        return instances_index

    @classmethod
    def _prune_branches_top_down_greedy(cls, rules: Sequence[Rule], X: numpy.ndarray,
                                        Y: numpy.ndarray,
                                        budget: TopDownGreedyBudget) -> List[Rule]:
        absolute, value = budget
        if absolute:
            assert isinstance(value, int)
            n_budget = max(0, min(value, len(rules)))
        else:
            assert value >= 0 and value <= 1
            n_budget = int(max(0, min(value * len(rules), len(rules))))
        instances_index = cls._build_instances_index(rules, X)
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
            cls.log.debug('iteration %d: selected rule %3d with %3d difference in matched Ys'
                          % (iteration, best_rule_id, best_matched_delta))
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

        # TODO(vmarkovtsev): optimize this the same way we optimized predict()
        new_rules = []
        for attrs, stats in rules:
            c = stats.cls
            new_verbs = []
            for feature, cmp, thr in attrs:
                table = numpy.zeros((2, 2), dtype=int)
                for x, y in zip(X, Y):
                    table[int((x[feature] <= thr) == cmp), int(c != y)] += 1
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
