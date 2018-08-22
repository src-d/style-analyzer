from collections import defaultdict
import functools
from itertools import islice
import logging
from typing import Any, Dict, Iterable, List, Mapping, NamedTuple, Sequence, Set, Tuple, Union

import modelforge
import numpy
from scipy.stats import fisher_exact
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.tree import _tree as Tree, DecisionTreeClassifier
from sklearn.utils.validation import check_is_fitted


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
    """
    A modelforge model to store Rules instances.
    It is required to store all the Rules for different programming languages in a single model,
    named after each language.
    Note that Rules must be fitted and Rules.base_model is not saved.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._rules_by_lang = {}

    @property
    def languages(self):
        return sorted(self._rules_by_lang)

    def construct(self, ruless: Iterable[Tuple[str, "Rules"]]) -> "FormatModel":
        for name, rules in ruless:
            self[name] = rules
        return self

    def dump(self) -> str:
        if len(self) == 0:
            return "<empty FormatModel>"
        languages = self.languages
        param_names = self[languages[0]].get_saved_param_names()
        return "Model languages: %s.\n" \
               "First model's params: %s\n" \
               "First model's rules number: %d.\n" % \
               (languages,
                ", ".join(["%s=%s" % (p, str(getattr(self[languages[0]], p)))
                           for p in param_names]),
                len(self[languages[0]]._rules))

    def _generate_tree(self) -> dict:
        languages = self.languages
        return dict(
            languages=languages,
            paramss=[self[lang].get_saved_params() for lang in languages],
            ruless=[self._disassemble_rules(self[lang]._rules) for lang in languages],
        )

    def _load_tree(self, tree: dict) -> None:
        for name, params, rules in zip(tree["languages"], tree["paramss"], tree["ruless"]):
            params = dict(params)
            params["top_down_greedy_budget"] = Rules.TopDownGreedyBudget(
                *params["top_down_greedy_budget"])
            params["_rules"] = self._assemble_rules(rules)
            _rules = Rules.__new__(Rules)
            _rules.__setstate__(params)
            self[name] = _rules

    def __len__(self) -> int:
        return len(self._rules_by_lang)

    def __getitem__(self, lang: str) -> "Rules":
        """
        Get the Rules estimator by its language.
        :param lang: Estimator language.
        :return: Rules estimator instance.
        """
        return self._rules_by_lang[lang]

    def __setitem__(self, lang: str, rules: "Rules"):
        """
        Set a new Rules estimator to the model by its language.
        """
        if not rules.fitted:
            raise NotFittedError("Rules estimator should be fitted before adding to FormatModel.")
        self._rules_by_lang[lang] = rules

    def __iter__(self):
        yield from self._rules_by_lang.__iter__()

    def __contains__(self, item):
        return item in self._rules_by_lang

    @staticmethod
    def _assemble_rules(rules_tree: dict) -> List[Rule]:
        rules = []
        rule_attrs = (RuleAttribute(*params) for params in
                      zip(rules_tree["features"],  rules_tree["cmps"], rules_tree["thresholds"]))
        for cls, conf, length in zip(rules_tree["cls"], rules_tree["conf"], rules_tree["lengths"]):
            rules.append(Rule(tuple(islice(rule_attrs, int(length))), RuleStats(cls, conf)))
        return rules

    @staticmethod
    def _disassemble_rules(rules: Iterable[Rule]):
        def disassemble_rule(rule: Rule) -> tuple:
            rule_len = len(rule.attrs)
            features, cmps, thresholds = zip(*rule.attrs)
            features = numpy.fromiter(features, numpy.uint16, rule_len)
            cmps = numpy.fromiter(cmps, numpy.bool, rule_len)
            thresholds = numpy.fromiter(thresholds, numpy.float32, rule_len)
            return (rule.stats.cls, rule.stats.conf, features, cmps, thresholds, rule_len)

        disassembled_rules = list(zip(*[disassemble_rule(rule) for rule in rules]))
        return dict(
            cls=numpy.array(disassembled_rules[0], dtype=numpy.uint16),
            conf=numpy.array(disassembled_rules[1], dtype=numpy.float32),
            features=numpy.concatenate(disassembled_rules[2]),
            cmps=numpy.concatenate(disassembled_rules[3]),
            thresholds=numpy.concatenate(disassembled_rules[4]),
            lengths=numpy.array(disassembled_rules[5], dtype=numpy.uint16),
        )


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
                 prune_branches: bool = True, prune_branches_algorithm: str = "top-down-greedy",
                 top_down_greedy_budget: TopDownGreedyBudget = (False, 1.0),
                 prune_attributes: bool = True, uncertain_attributes: bool = True):
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
        if isinstance(base_model, DecisionTreeClassifier):
            check_is_fitted(base_model, "tree_")
        elif isinstance(base_model, RandomForestClassifier):
            check_is_fitted(base_model, "estimators_")
        else:
            raise ValueError("base_model must be a DecisionTreeClassifier or a "
                             "RandomForestClassifier.")
        self.base_model = base_model
        self.prune_branches = prune_branches
        self.prune_branches_algorithm = prune_branches_algorithm
        self.top_down_greedy_budget = top_down_greedy_budget
        self.prune_attributes = prune_attributes
        self.uncertain_attributes = uncertain_attributes
        self._compiled = None  # type: Rules.CompiledRulesType
        self._rules = None  # type: List[Rule]

    @property
    def base_model(self) -> Union[DecisionTreeClassifier, RandomForestClassifier]:
        return self._base_model

    @base_model.setter
    def base_model(self, value: Union[DecisionTreeClassifier, RandomForestClassifier]):
        if not isinstance(value, (DecisionTreeClassifier, RandomForestClassifier)):
            raise TypeError("base_model must be an instance of DecisionTreeClassifier or "
                            "RandomForestClassifier.")
        self._base_model = value

    def _check_fittable(func):
        @functools.wraps(func)
        def wrapped_check_fittable(self: "Rules", *args, **kwargs):
            if not self.fittable:
                raise ValueError("This method requires a fittable instance of Rules.")
            return func(self, *args, **kwargs)
        return wrapped_check_fittable

    @_check_fittable
    def fit(self, X: numpy.ndarray, y: numpy.ndarray) -> "Rules":
        """
        Trains the rules using the base tree model and the samples (X, y). If `base_model` is
        already fitted, the samples may be different from the ones that were used.

        :param X: input features.
        :param y: input labels - the same length as X.
        :return: self
        """
        if isinstance(self.base_model, DecisionTreeClassifier):
            rules, leaf2rule_dict = self._tree_to_rules(self.base_model)
            leaf2rule = [leaf2rule_dict]
        else:
            rules = []
            offset = 0
            leaf2rule = []
            for i, estimator in enumerate(self.base_model.estimators_):
                rules_partial, leaf2rule_partial = self._tree_to_rules(estimator, offset=offset)
                offset += len(rules_partial)
                leaf2rule.append(leaf2rule_partial)
                rules.extend(rules_partial)

        def count_attrs():
            return sum(len(r.attrs) for r in rules)

        self.log.debug("Initial number of rules: %d", len(rules))
        self.log.debug("Initial number of attributes: %d", count_attrs())
        rules = self._merge_rules(rules)
        self.log.debug("Merged number of attributes: %d", count_attrs())
        if self.prune_branches:
            if self.prune_branches_algorithm == "top-down-greedy":
                rules = self._prune_branches_top_down_greedy(rules, X, y, leaf2rule,
                                                             self.top_down_greedy_budget)
            else:
                rules = self._prune_branches(rules, X, y)
        if self.prune_attributes:
            rules = self._prune_attributes(rules, X, y, not self.uncertain_attributes)
            self.log.debug("Pruned number of attributes (2): %d", len(rules))
            self.log.debug("Pruned number of attributes: %d", count_attrs())
        self._rules = rules
        self._compiled = self._compile_rules(self._rules)
        return self

    def get_saved_param_names(self) -> Sequence[str]:
        param_names = self._get_param_names()
        param_names.remove("base_model")
        return param_names

    def get_saved_params(self) -> Mapping[str, Any]:
        params = self.get_params(False)
        return {param_name: param for param_name, param in params.items()
                if param_name in self.get_saved_param_names()}

    @property
    def fitted(self):
        return self._rules is not None

    @property
    def fittable(self):
        return self.base_model is not None

    def _check_fitted(func):
        @functools.wraps(func)
        def wrapped_check_fitted(self: "Rules", *args, **kwargs):
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
        rules = self._rules
        prediction = numpy.zeros(len(X), dtype=int)
        for xi, x in enumerate(X):
            ris = self._compute_triggered(self._compiled, rules, x)
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

    _check_fitted = staticmethod(_check_fitted)
    _check_fittable = staticmethod(_check_fittable)

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

    def _build_instances_index(self, rules: Sequence[Rule], X: numpy.ndarray,
                               leaf2rule: Sequence[Mapping[int, int]]) -> Dict[int, Set[int]]:

        self.log.debug("building instances index")

        instances_index = defaultdict(set)

        if isinstance(self.base_model, DecisionTreeClassifier):
            leaves = self.base_model.apply(X)  # ndim = 1
            for i, leaf in enumerate(leaves):
                instances_index[leaf2rule[0][leaf]].add(i)
        else:
            leaves = self.base_model.apply(X)  # ndim = 2
            for i, col in enumerate(leaves):
                for leaf, l2r in zip(col, leaf2rule):
                    instances_index[l2r[leaf]].add(i)
        return instances_index

    def _prune_branches_top_down_greedy(self, rules: Sequence[Rule], X: numpy.ndarray,
                                        Y: numpy.ndarray, leaf2rule: Sequence[Mapping[int, int]],
                                        budget: TopDownGreedyBudget) -> List[Rule]:
        absolute, value = budget
        if absolute:
            assert isinstance(value, int)
            n_budget = max(0, min(value, len(rules)))
        else:
            assert value >= 0 and value <= 1
            n_budget = int(max(0, min(value * len(rules), len(rules))))
        instances_index = self._build_instances_index(rules, X, leaf2rule)
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
            self.log.debug("iteration %d: selected rule %3d with %3d difference in matched Ys"
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

    def __setstate__(self, state):
        state["_base_model"] = None
        super().__setstate__(state)
        if self._rules is not None:
            self._compiled = self._compile_rules(self._rules)

    def __getstate__(self):
        state = super().__getstate__()
        del state["_base_model"]
        del state["_compiled"]
        return state
