"""Train and compile rules for multi-class classification using an sklearn base model."""
from collections import defaultdict, OrderedDict
from copy import deepcopy
import functools
from importlib import import_module
from itertools import islice
import logging
import sys
from typing import (Any, Dict, Iterable, Iterator, List, Mapping, NamedTuple, Optional,
                    Sequence, Set, Tuple, Union)

from bblfsh import role_name
from igraph import Graph
from lookout.core import slogging
from lookout.core.ports import Type
import numpy
from numpy import count_nonzero
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score, confusion_matrix, \
    precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.tree import _tree as Tree, DecisionTreeClassifier
from tqdm import tqdm

from lookout.style.format.classes import CLASS_INDEX, CLS_DOUBLE_QUOTE, CLS_SINGLE_QUOTE
from lookout.style.format.feature_extractor import FeatureExtractor
from lookout.style.format.features import CategoricalFeature, Feature, FeatureGroup, FeatureId
from lookout.style.format.utils import get_classification_report
from lookout.style.format.virtual_node import VirtualNode

RuleAttribute = NamedTuple(
    "RuleAttribute", (("feature", int), ("cmp", bool), ("threshold", float)))
"""
`feature` is the feature taken for comparison
`cmp` is the comparison type: True is "x > v", False is "x <= v"
`threshold` is "v", the threshold value
"""


RuleStats = NamedTuple("RuleStats", (("cls", int), ("conf", float), ("support", int)))
"""
`cls` is the predicted class
`conf` is the rule confidence \\in [0, 1], "1" means super confident
"""


class Rule(NamedTuple("RuleType", (("attrs", Tuple[RuleAttribute, ...]), ("stats", RuleStats),
                                   ("artificial", bool)))):
    """
    Decision rule which consists of a series of attribute comparisons, statistics and the flag \
    which indicates whether the rule was created outside of the training (notably, \
    in Rules.harmonize_quotes()). The statistics contain the predicted class index.
    """

    def group_features(self, feature_extractor: FeatureExtractor) -> Iterator[
            Tuple[Feature, FeatureId, List[RuleAttribute], int, FeatureGroup]]:
        """
        Generate rule splits grouped by feature type.

        Attribute indexes are from the original sequence before feature selection!

        :param feature_extractor: The FeatureExtractor used to create those rules.
        :return: generator
        """
        if feature_extractor.features is None or feature_extractor.index_to_feature is None:
            raise NotFittedError()
        grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for attr in self.attrs:
            group, node_index, feature_id, original_feature_index = \
                feature_extractor.index_to_feature[attr.feature]
            grouped[group][node_index][feature_id].append(RuleAttribute(
                original_feature_index, attr.cmp, attr.threshold))
        for group, nodes in sorted(grouped.items()):
            for node_index, feature_ids in sorted(nodes.items()):
                for feature_id, splits in sorted(feature_ids.items()):
                    feature = feature_extractor.features[group][node_index][feature_id]
                    yield feature, feature_id, splits, node_index, group


QuotedNodeTriple = NamedTuple("QuotedNodeTriple", (("left", VirtualNode), ("target", VirtualNode),
                                                   ("right", VirtualNode)))
QuotedNodeTripleMapping = Mapping[int, Optional[QuotedNodeTriple]]


class Rules:
    """Store already trained rules for downstream prediction tasks."""

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

    def __init__(self, rules: List[Rule], origin_config: Mapping[str, Any]):
        """
        Initialize the rules so that it is possible to call predict() afterwards.

        :param rules: List of rules to assign.
        :param origin_config: All parameters that are used for the model training.
        """
        super().__init__()
        assert rules is not None, "rules may not be None"
        self._rules = tuple(rules)  # Rule list is constant
        self._compiled = self._compile(rules)
        self._origin_config = origin_config
        self._classification_report = {"test": {}, "train": {}}  # type: Dict[str, Dict[str, Any]]

    def __str__(self):
        return "%d rules, avg.len. %.1f" % (len(self._rules), self.avg_rule_len)

    def __len__(self):
        return len(self._rules)

    @property
    def classification_report(self) -> Dict[str, Dict]:
        """
        Property for classification report with quality metrics.

        Return empty dict if unset.
        Can be set for a dataset with generate_classification_report() method.
        :return: Classification report.
        """
        return self._classification_report

    def apply(self, X_csr: csr_matrix, return_winner_indices=False,
              ) -> Union[numpy.ndarray, Tuple[numpy.ndarray, numpy.ndarray]]:
        """
        Evaluate the rules against the given features.

        :param X_csr: input features.
        :param return_winner_indices: whether to return the winning rule index for each sample.
        :return: array of the same length as X with predictions or tuple of two arrays of the same\
                 length as X containing (predictions, winner rule indices). In case no rule was \
                 triggered for feature row, corresponding result equals to -1.
        """
        X = X_csr.toarray()
        self._log.debug("predicting %d samples using %d rules", len(X), len(self._rules))
        rules = self._rules
        _compute_triggered = self._compute_triggered
        prediction = numpy.full(len(X), -1, dtype=numpy.int32)
        if return_winner_indices:
            winner_indices = numpy.full(len(X), -2, dtype=numpy.int32)
        for xi, x in enumerate(X):
            ris = _compute_triggered(self._compiled, rules, x)
            if len(ris) == 0:
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
        self._log.debug("No rule was triggered in %d cases.", numpy.sum(prediction == -1))
        if return_winner_indices:
            return prediction, winner_indices
        return prediction

    def predict(
            self, X: csr_matrix, vnodes_y: Sequence[VirtualNode], vnodes: Sequence[VirtualNode],
            feature_extractor: FeatureExtractor,
            ) -> Tuple[numpy.ndarray, numpy.ndarray, "Rules", QuotedNodeTripleMapping]:
        """
        Predict classes given the input features and metadata.

        :param X: Numpy 1-dimensional array of input features.
        :param vnodes_y: Sequence of the labeled `VirtualNode`-s corresponding to labeled samples.
        :param vnodes: Sequence of all the `VirtualNode`-s corresponding to the input.
        :param feature_extractor: FeatureExtractor used to extract features.
        :return: The predictions, the winning rules and the new Rules.
        """
        y_pred, winners = self.apply(X, True)
        triggered = y_pred > 0
        vnodes_y_triggered = [vny for t, vny in zip(triggered, vnodes_y) if t]
        grouped_quote_predictions = self._group_quote_predictions(vnodes_y_triggered, vnodes)
        y_pred[triggered], winners[triggered], new_rules = self.harmonize_quotes(
            y_pred=y_pred[triggered], vnodes_y=vnodes_y_triggered, vnodes=vnodes,
            winners=winners[triggered], feature_extractor=feature_extractor,
            grouped_quote_predictions=grouped_quote_predictions)
        return y_pred, winners, new_rules, grouped_quote_predictions

    @staticmethod
    def fill_missing_predictions(y: numpy.ndarray, y_fallback: numpy.ndarray,
                                 ) -> numpy.ndarray:
        """
        Fill missing predictions with original labels.

        :param y: Array with predictions. Negative values are considered as missing predictions.
        :param y_fallback: Original labels. Vector should have the same length as `y`.
        :return: Filled array with labels. The array have the same size as original.
        """
        assert y.shape == y_fallback.shape, "y and y_fallback should have the same shape."
        no_rule_triggered = y < -1
        y = y.copy()
        y[no_rule_triggered] = y_fallback[no_rule_triggered]
        return y

    def filter_by_confidence(self, confidence_threshold: float) -> "Rules":
        """
        Filter rules according to a confidence threshold.

        :param confidence_threshold: Minimum confidence value.
        :return: Filtered rules.
        """
        rules = [rule for rule in self._rules if rule.stats.conf > confidence_threshold]
        self._log.debug("Filtered rules by confidence >= %.3f: %d -> %d",
                        confidence_threshold, len(self._rules), len(rules))
        return Rules(rules, self._origin_config)

    def filter_by_support(self, support_threshold: int) -> "Rules":
        """
        Filter rules according to a support threshold.

        :param support_threshold: Minimum support value.
        :return: Filtered rules.
        """
        rules = [rule for rule in self._rules if rule.stats.support > support_threshold]
        self._log.debug("Filtered rules by support >= %d: %d -> %d",
                        support_threshold, len(self._rules), len(rules))
        return Rules(rules, self._origin_config)

    def generate_classification_report(self, X: csr_matrix, y: numpy.ndarray, dataset_type: str,
                                       target_names: Sequence[str]) -> None:
        """
        Calculate and store classification report with quality metrics for given dataset.

        :param X: Features matrix.
        :param y: target vector.
        :param dataset_type: Can be set to "test" or "train" only. Marks passing data as train or \
                             test.
        :param target_names: Classes names in y.
        """
        # TODO(zurk): multi-language support.
        assert dataset_type in {"test", "train"}, "Unknown dataset_type='%s'. Known are 'test' " \
                                                  "and 'train'" % dataset_type
        y_pred = self.apply(X)
        self._classification_report[dataset_type] = get_classification_report(
            y_pred, y, target_names)

    @staticmethod
    def _get_composite(feature_extractor: FeatureExtractor, labels: Tuple[int, ...]) -> int:
        if labels in feature_extractor.class_sequences_to_labels:
            return feature_extractor.class_sequences_to_labels[labels]
        feature_extractor.class_sequences_to_labels[labels] = \
            len(feature_extractor.class_sequences_to_labels)
        feature_extractor.labels_to_class_sequences.append(labels)
        return len(feature_extractor.labels_to_class_sequences) - 1

    def _group_quote_predictions(self, vnodes_y: Sequence[VirtualNode],
                                 vnodes: Sequence[VirtualNode]) -> QuotedNodeTripleMapping:
        quotes_classes = frozenset((CLASS_INDEX[CLS_DOUBLE_QUOTE], CLASS_INDEX[CLS_SINGLE_QUOTE]))
        y_indices = {id(vnode): i for i, vnode in enumerate(vnodes_y)}
        grouped_predictions = OrderedDict()
        for vnode1, vnode2, vnode3 in zip(vnodes, islice(vnodes, 1, None),
                                          islice(vnodes, 2, None)):
            if (id(vnode1) not in y_indices or id(vnode3) not in y_indices or vnode2.node is None
                    or vnode1.y[-1] not in quotes_classes or vnode3.y[0] != vnode1.y[-1]):
                continue
            vnode2_roles = frozenset(role_name(role_id) for role_id in vnode2.node.roles)
            if "STRING" in vnode2_roles:
                grouped_predictions[id(vnode1)] = vnode1, vnode2, vnode3
                grouped_predictions[id(vnode3)] = None
        return grouped_predictions

    def harmonize_quotes(self, y_pred: numpy.ndarray, vnodes_y: Sequence[VirtualNode],
                         vnodes: Sequence[VirtualNode], winners: numpy.ndarray,
                         feature_extractor: FeatureExtractor,
                         grouped_quote_predictions: QuotedNodeTripleMapping,
                         ) -> Tuple[numpy.ndarray, numpy.ndarray, "Rules"]:
        """
        Post-process predictions to correct mis-matched quotes.

        To do so, we consider only the tuples (', STRING, ') or (", STRING, ") in the input. We
        then create fake rules as needed (because a rule going from the input to the corrected
        quote might not exist in the trained rules).

        :param y_pred: Predictions to correct.
        :param vnodes_y: Sequence of the predicted virtual nodes.
        :param vnodes: Sequence of virtual nodes representing the input.
        :param winners: Indices of the rules that were used to compute the predictions.
        :param feature_extractor: FeatureExtractor used to extract features.
        :param grouped_quote_predictions: Quotes predictions (handled differenlty from the rest).
        :return: Updated y, winners and new rules.
        """
        quotes_classes = {CLASS_INDEX[CLS_DOUBLE_QUOTE], CLASS_INDEX[CLS_SINGLE_QUOTE]}
        processed_rules = list(self.rules)
        processed_y = y_pred.copy()
        processed_winners = winners.copy()
        new_rules = {}

        def append_new_rule(labels: Tuple[int, ...], y_i: int, conf: float, support: int) -> None:
            rule_id = (labels, conf, support)
            if rule_id in new_rules:
                rule_index = new_rules[rule_id]
            else:
                processed_rules.append(
                    Rule(attrs=tuple(),
                         stats=RuleStats(cls=Rules._get_composite(feature_extractor, labels),
                                         conf=conf, support=support),
                         artificial=True))
                rule_index = len(processed_rules) - 1
                new_rules[rule_id] = rule_index
            processed_winners[y_i] = rule_index
            processed_y[y_i] = processed_rules[rule_index].stats.cls

        y_indices = {id(vnode): i for i, vnode in enumerate(vnodes_y)}
        for group in grouped_quote_predictions.values():
            if group is None:
                continue
            vnode1, vnode2, vnode3 = group
            y_i_1 = y_indices[id(vnode1)]
            y_i_3 = y_indices[id(vnode3)]
            stats_vnode1 = processed_rules[winners[y_i_1]].stats
            stats_vnode3 = processed_rules[winners[y_i_3]].stats
            labels1 = list(feature_extractor.labels_to_class_sequences[y_pred[y_i_1]])
            labels3 = list(feature_extractor.labels_to_class_sequences[y_pred[y_i_3]])
            if labels1[-1] not in quotes_classes or labels3[0] not in quotes_classes:
                append_new_rule(vnode1.y, y_i_1, 1., 1)
                append_new_rule(vnode3.y, y_i_3, 1., 1)
            elif labels1[-1] != labels3[0]:
                quote = labels1[-1] if stats_vnode1.conf >= stats_vnode3.conf else labels3[0]
                if labels1[-1] != quote:
                    labels1[-1] = quote
                    append_new_rule(tuple(labels1), y_i_1, stats_vnode3.conf, stats_vnode3.support)
                else:
                    labels3[0] = quote
                    append_new_rule(tuple(labels3), y_i_3, stats_vnode1.conf, stats_vnode1.support)
        return processed_y, processed_winners, Rules(processed_rules, self._origin_config)

    @property
    def rules(self) -> List[Rule]:
        """Return the list of rules."""
        return self._rules

    @property
    def origin_config(self) -> Mapping[str, Any]:
        """Return the configuration used for the model training."""
        return self._origin_config

    @property
    def avg_rule_len(self) -> float:
        """Compute the average length of the rules."""
        if not self._rules:
            return 0
        return sum(len(r.attrs) for r in self._rules) / len(self._rules)

    @classmethod
    def _compile(cls, rules: Sequence[Rule]) -> CompiledRulesType:
        cls._log.debug("compiling %d rules", len(rules))
        attrs = defaultdict(lambda: defaultdict(lambda: [[], []]))
        for i, (branch, _, _) in enumerate(rules):
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
                           rules: Sequence[Rule], x: numpy.ndarray,
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


LabelScore = NamedTuple("LabelScore", (
    ("accuracy", float), ("precision", float), ("recall", float), ("f", float), ("support", int)))


class TrainableRules(BaseEstimator, ClassifierMixin):
    """Trainable rules model based on a decision tree or a random forest."""

    _log = logging.getLogger("TrainableRules")

    def __init__(self, *, base_model_name: str = "sklearn.tree.DecisionTreeClassifier",
                 prune_branches_algorithms=("reduced-error", "top-down-greedy"),
                 top_down_greedy_budget: Tuple[bool, Union[float, int]] = (False, 1.0),
                 prune_attributes=True, confidence_threshold=0.8,
                 attribute_similarity_threshold=0.98, prune_dataset_ratio=.2, n_estimators=10,
                 max_depth=None, max_features=None, min_samples_leaf=1, min_samples_split=2,
                 random_state=42, origin_config=None):
        """
        Initialize a new instance of Rules class.

        :param base_model_name: fully qualified type name of the base model to train. \
                                Must be either "sklearn.tree.DecisionTreeClassifier" or \
                                "sklearn.ensemble.RandomForestClassifier".
        :param prune_branches_algorithms: branch pruning algorithms to use.
        :param top_down_greedy_budget: tuple describing the budget of the top down algorithm: \
                                       boolean to indicate if it's absolute (True) or not \
                                       (False). If the first value is True (absolute budget), the \
                                       second  should be an integer describing the maximum number \
                                       of rules to keep. If it is False (relative budget), it \
                                       should be a float between 0 and 1 to specify the \
                                       proportion of rules to keep.
        :param prune_attributes: indicates whether to remove useless parts of rules.
        :param confidence_threshold: confidence threshold to filter the rules.
        :param attribute_similarity_threshold: remove attribute comparisons which trigger on \
                                               similar samples.
        :param prune_dataset_ratio: Ratio of the dataset to use for pruning during training.
        :param n_estimators: n_estimators parameter of the base model.
        :param max_depth: max_depth parameter of the base model.
        :param max_features: max_features parameter of the base model.
        :param min_samples_leaf: min_samples parameter of the base model.
        :param min_samples_split: min_samples_split parameter of the base model.
        :param random_state: random_state parameter of the base model.
        :param origin_config: all parameters that are used for the model training.
        """
        super().__init__()

        self.base_model_name = base_model_name
        self.prune_branches_algorithms = prune_branches_algorithms
        self.top_down_greedy_budget = top_down_greedy_budget
        self.prune_attributes = prune_attributes
        self.confidence_threshold = confidence_threshold
        self.attribute_similarity_threshold = attribute_similarity_threshold
        self.prune_dataset_ratio = prune_dataset_ratio
        # Parameters for base_model must be named the same as in the base_model class
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self._rules = None  # type: Rules
        self._origin_config = origin_config

    def fit(self, X: csr_matrix, y: numpy.ndarray) -> "TrainableRules":
        """
        Train the rules using the base tree model and the samples (X, y).

        If `base_model` is already fitted, the samples may be different from the ones that were
        used.

        :param X: input features.
        :param y: input labels - the same length as X.
        :return: self
        """
        self._log.debug("fitting rules with params %s", self.get_params(False))
        models_params = {name: val for name, val in self.get_params().items()
                         if name in self._base_param_names}
        base_model = self._base_model_class(**models_params)

        if self.prune_branches_algorithms or self.prune_attributes:
            X_train, X_prune, y_train, y_prune = train_test_split(
                X, y, test_size=self.prune_dataset_ratio, random_state=42)
        else:
            X_train, y_train = X, y
        base_model.fit(X_train, y_train)
        self.feature_importances_ = base_model.feature_importances_

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
            for estimator in base_model.estimators_:
                if "reduced-error" in self.prune_branches_algorithms:
                    old += (count_nonzero(estimator.tree_.children_left == Tree.TREE_LEAF)
                            + count_nonzero(estimator.tree_.children_right == Tree.TREE_LEAF))
                    estimator = self._prune_reduced_error(estimator, X_prune, y_prune)
                    new += (count_nonzero(estimator.tree_.children_left == Tree.TREE_LEAF)
                            + count_nonzero(estimator.tree_.children_right == Tree.TREE_LEAF))
                rules_partial, leaf2rule_partial = self._tree_to_rules(
                    estimator, offset=offset, class_mapping=base_model.classes_)
                offset += len(rules_partial)
                leaf2rule.append(leaf2rule_partial)
                rules.extend(rules_partial)
            if "reduced-error" in self.prune_branches_algorithms:
                self._log.debug("pruned %d/%d branches with reduced error pruning", old - new, old)

        def count_attrs():
            return sum(len(r.attrs) for r in rules)

        old = count_attrs()
        rules = [rule for rule in rules if rule.stats.conf > self.confidence_threshold]
        rules = self._merge_rules(rules)
        self._log.debug("merged %d/%d attributes", old - count_attrs(), old)
        if "top-down-greedy" in self.prune_branches_algorithms:
            old = len(rules)
            rules = self._prune_branches_top_down_greedy(
                base_model, rules, X_prune, y_prune, leaf2rule, self.top_down_greedy_budget)
            self._log.debug("pruned %d/%d rules w/ greedy pruning", old - len(rules), old)
        if self.prune_attributes:
            old_attrs = count_attrs()
            old_rules_len = len(rules)
            rules = self._prune_attributes(
                rules, X_prune, y_prune, self.attribute_similarity_threshold)
            self._log.debug("pruned %d/%d attributes (%d/%d rules)",
                            old_attrs - count_attrs(), old_attrs, old_rules_len - len(rules),
                            old_rules_len)
        self._rules = Rules(rules, self._origin_config)
        return self

    def prune_categorical_attributes(self, feature_extractor: FeatureExtractor) -> None:
        """
        Remove "not in" categorical assertions which are overridden by strict equalities.

        :param feature_extractor: FeatureExtractor which created the train samples.
        :return: Nothing
        """
        new_rules = []
        known_rules = set()
        pruned_count = 0
        attr_count = 0
        for rule in self.rules.rules:
            attr_count += len(rule.attrs)
            excluded = []
            if not rule.artificial:
                for feature, _, splits, _, _ in rule.group_features(feature_extractor):
                    if isinstance(feature, CategoricalFeature):
                        if len(splits) <= 1:
                            continue
                        has_included = False
                        for _, cmp, _ in splits:
                            if cmp:
                                has_included = True
                                break
                        if not has_included:
                            continue
                        for attr in splits:
                            if not attr.cmp:
                                excluded.append(attr)
            if excluded:
                pruned_count += len(excluded)
                attrs = tuple(a for a in rule.attrs if RuleAttribute(
                    feature_extractor.index_to_feature[a.feature][3], a.cmp, a.threshold)
                              not in excluded)
            else:
                attrs = rule.attrs
            if attrs not in known_rules:
                new_rules.append(Rule(attrs, rule.stats, rule.artificial))
                known_rules.add(attrs)
        self._log.debug("pruned %d/%d categorical attributes (%d/%d rules)",
                        pruned_count, attr_count, len(self.rules) - len(new_rules),
                        len(self.rules))
        self._rules = Rules(new_rules, self.rules.origin_config)

    @property
    def base_model_name(self) -> str:
        """Return the name of the base model used for training."""
        return self._base_model_name

    @base_model_name.setter
    def base_model_name(self, value: Union[str, Type[DecisionTreeClassifier],
                                           Type[RandomForestClassifier]]):
        """Set the name of the base model used for training."""
        if isinstance(value, str):
            self._base_model_name = value
            base_model_module_name, base_model_class_name = value.rsplit(".", 1)
            base_model_module = import_module(base_model_module_name)
            value = getattr(base_model_module, base_model_class_name)
        else:
            self._base_model_name = "%s.%s" % (value.__module__, value.__name__)
        if not issubclass(value, (DecisionTreeClassifier, RandomForestClassifier)):
            raise TypeError("%s base model type is not allowed" % value)
        self._base_model_class = value
        self._base_param_names = set(self._base_model_class().get_params())

    @property
    def fitted(self):
        """Return whether the model is fitted or not."""
        return self._rules is not None

    def _check_fitted(func):
        @functools.wraps(func)
        def wrapped_check_fitted(self: "TrainableRules", *args, **kwargs):
            if not self.fitted:
                raise NotFittedError
            return func(self, *args, **kwargs)

        return wrapped_check_fitted

    @_check_fitted
    def predict(self, X: csr_matrix) -> numpy.ndarray:
        """
        Evaluate the rules against the given features.

        :param X: Input features.
        :return: Array of the same length as X with predictions.
        """
        return self._rules.apply(X)

    @_check_fitted
    def full_score(self, X: csr_matrix, y: numpy.ndarray) -> Dict[int, LabelScore]:
        """
        Evaluate the trained rules and return the metrics.

        :param X: Input data.
        :param y: Output labels.
        :return: Mapping from labels to `ClassScore`-s.
        """
        y_pred = self.predict(X)
        labels = numpy.unique(y)
        cm = confusion_matrix(y_true=y, y_pred=y_pred, labels=labels)
        accuracy = (cm / cm.sum(axis=1)[:, numpy.newaxis]).diagonal()
        precision, recall, fs, support = precision_recall_fscore_support(
            y_true=y, y_pred=y_pred, labels=labels)
        return {cls: LabelScore(accuracy=acc, precision=prec, recall=rec, f=f, support=sup)
                for (cls, acc, prec, rec, f, sup)
                in zip(labels, accuracy, precision, recall, fs, support)}

    _check_fitted = staticmethod(_check_fitted)

    @property
    def rules(self) -> Rules:
        """Return the list of rules."""
        return self._rules

    @classmethod
    def _tree_to_rules(cls, tree: DecisionTreeClassifier, offset: int = 0,
                       class_mapping: Optional[numpy.ndarray] = None,
                       ) -> Tuple[List[Rule], Mapping[int, int]]:
        """
        Convert an sklearn decision tree to a set of rules.

        Each rule is a branch in the tree.

        :param tree: input decision tree.
        :param offset: offset for the rules' identifiers - used when there are several trees.
        :param class_mapping: mapping for rules' classes - used when there are several trees.
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
                # why -0.5? Read R. Quinlan's paper about production rules.
                support = freqs.sum()
                conf = (freqs.max() - 0.5) / support
                leaf2rule[node] = len(rules) + offset
                prediction = int(tree.classes_[numpy.argmax(freqs)])
                if class_mapping is not None:
                    prediction = class_mapping[prediction]
                rules.append(Rule(attrs=path, stats=RuleStats(prediction, conf, support),
                                  artificial=False))
        return rules, leaf2rule

    @classmethod
    def _merge_rules(cls, rules: List[Rule]) -> List[Rule]:
        new_rules = []
        for rule, stats, artificial in rules:
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
            new_rules.append(Rule(attrs=tuple(new_rule), stats=stats, artificial=artificial))
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
            leaf2rule: Sequence[Mapping[int, int]], budget: Tuple[bool, Union[float, int]],
            ) -> List[Rule]:
        """
        Prune branches using a greedy top down algorithm.

        :param base_model: Sklearn decision tree or random forest base model.
        :param rules: Rules extracted from the base model.
        :param X: Samples to use to evaluate the quality of subsets of branches.
        :param Y: Labels to use to evaluate the quality of subsets of branches.
        :param leaf2rule: Mapping from leaves in the base model to rules.
        :param budget: Tuple describing the budget: boolean to indicate if it's absolute (True) \
                       or not (False). If the first value is True (absolute budget), the second \
                       should be an integer describing the maximum number of rules to keep. If it \
                       is False (relative budget), it should be a float between 0 and 1 to \
                       specify the proportion of rules to keep.
        :return: Pruned list of rules.
        """
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
        for _ in range(n_budget):
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
                          X: csr_matrix, Y: numpy.ndarray,
                          sim_threshold: float) -> List[Rule]:
        """
        Remove the attribute comparisons which do not influence the rule decision.

        We treat two attribute comparisons as similar if the samples on which they trigger and \
        mistake are similar by Jaccard metric.

        :param rules: List of rules to simplify.
        :param X: Input features, used to exclude the irrelevant attributes.
        :param Y: Input labels.
        :param sim_threshold: how many attributes to prune. Must be between 0 and 1. \
                              The closer to 0, the fewer attributes are left.
        :return: New list of simplified rules.
        """
        new_rules_set = set()
        new_rules = []
        pseudo_progress = False
        if cls._log.isEnabledFor(logging.DEBUG) and not slogging.logs_are_structured:
            if sys.stderr.isatty():
                rules = tqdm(rules)
            else:
                pseudo_progress = True
        x_array = X.toarray()
        not_cs = {}
        for i, (rule, stats, artificial) in enumerate(rules):
            if pseudo_progress and ((i + 1) % 100) == 0:
                cls._log.debug("attributes pruning status: %d/%d", i + 1, len(rules))
            if artificial:
                new_rules.append(rule)
                continue
            c = stats.cls
            not_c = not_cs.setdefault(c, Y != c)
            errs = []
            for feature, cmp, thr in rule:
                if cmp:
                    errs.append(frozenset(numpy.nonzero((x_array[:, feature] > thr) & not_c)[0]))
                else:
                    errs.append(frozenset(numpy.nonzero((x_array[:, feature] <= thr) & not_c)[0]))
            graph = Graph()
            graph.add_vertices(len(rule))
            for x, tx in enumerate(errs):
                for y, ty in enumerate(errs[x + 1:]):
                    y += x + 1
                    sim = len(tx.intersection(ty)) / len(tx.union(ty))
                    if sim > sim_threshold:
                        graph.add_edge(x, y)
            communities = graph.community_multilevel()
            saved = set()
            clusters = {}
            for i, m in enumerate(communities.membership):
                if not clusters.get(m, False):
                    saved.add(i)
                    clusters[m] = True
            if len(saved) == len(rule):
                new_rule = Rule(rule, stats, artificial)
            else:
                new_rule = Rule(
                    tuple(r for i, r in enumerate(rule) if i in saved), stats, artificial)
            if new_rule.attrs not in new_rules_set:
                new_rules.append(new_rule)
                new_rules_set.add(new_rule.attrs)
        return new_rules

    @staticmethod
    def _sanitize_params(params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize the parameters from get_params() so that they are suitable for serialization.

        :param params: Dictionary obtained from get_params().
        :return: Normalized dictionary.
        """
        sanitized = {}
        for k, v in params.items():
            if isinstance(v, tuple):
                # fix namedtuple-s in ASDF
                v = list(v)
            sanitized[k] = v
        return sanitized

    @classmethod
    def _get_param_names(cls):
        names = super()._get_param_names()
        names.remove("origin_config")
        return names
