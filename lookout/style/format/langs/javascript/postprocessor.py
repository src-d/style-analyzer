"""Prediction post processing module."""
from itertools import islice
from typing import MutableMapping, Sequence, Tuple

from bblfsh import role_name
import numpy

from lookout.style.format.feature_extractor import FeatureExtractor
from lookout.style.format.virtual_node import (
    VirtualNode)
from lookout.style.format.classes import CLS_SINGLE_QUOTE, CLS_DOUBLE_QUOTE, CLASS_INDEX
from lookout.style.format.rules import Rule, Rules, RuleStats


def _get_composite(feature_extractor: FeatureExtractor, labels: Tuple[int, ...]) -> int:
    if labels in feature_extractor.class_sequences_to_labels:
        return feature_extractor.class_sequences_to_labels[labels]
    feature_extractor.class_sequences_to_labels[labels] = \
        len(feature_extractor.class_sequences_to_labels)
    feature_extractor.labels_to_class_sequences.append(labels)
    return len(feature_extractor.labels_to_class_sequences) - 1


def _get_new_rule(feature_extractor: FeatureExtractor, labels: Tuple[int, ...], rules: Rules,
                  new_rules: MutableMapping[Tuple[int, ...], int]) -> int:
    if labels in new_rules:
        return new_rules[labels]
    rules._rules.append(Rule(tuple(), RuleStats(cls=_get_composite(feature_extractor, labels),
                                                conf=1., support=1)))
    rule_i = len(rules._rules) - 1
    new_rules[labels] = rule_i
    return rule_i


def _set_new_rule(feature_extractor: FeatureExtractor, labels: Tuple[int, ...], rules: Rules,
                  new_rules: MutableMapping[Tuple[int, ...], int], winners: numpy.ndarray,
                  y: numpy.ndarray, y_i: int) -> None:
    rule = _get_new_rule(feature_extractor, tuple(labels), rules, new_rules)
    winners[y_i] = rule
    y[y_i] = rules.rules[rule].stats.cls


def postprocess(X: numpy.ndarray, y_pred: numpy.ndarray, vnodes_y: Sequence[VirtualNode],
                vnodes: Sequence[VirtualNode], winners: numpy.ndarray, rules: Rules,
                feature_extractor: FeatureExtractor) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """
    Post-process predictions to account for domain constraints.

    :param X: Feature matrix.
    :param y_pred: Predictions to correct.
    :param vnodes_y: Sequence of the predicted virtual nodes.
    :param vnodes: Sequence of virtual nodes representing the input.
    :param winners: Indices of the rules that were used to compute the predictions.
    :param rules: Rules used to perform the prediction.
    :param feature_extractor: FeatureExtractor used to extract features.
    :return: Updated y and winners.
    """
    quotes_classes = {CLASS_INDEX[CLS_DOUBLE_QUOTE], CLASS_INDEX[CLS_SINGLE_QUOTE]}
    processed_y = y_pred.copy()
    processed_winners = winners.copy()
    y_indices = {id(vnode): i for i, vnode in enumerate(vnodes_y)}
    new_rules = {}
    for vnode1, vnode2, vnode3 in zip(vnodes, islice(vnodes, 1, None), islice(vnodes, 2, None)):
        if (id(vnode1) not in y_indices or id(vnode3) not in y_indices or vnode2.node is None
                or vnode1.y[-1] not in quotes_classes or vnode3.y[0] != vnode1.y[-1]):
            continue
        vnode2_roles = frozenset(role_name(role_id) for role_id in vnode2.node.roles)
        if "STRING" not in vnode2_roles:
            continue
        y_i_1 = y_indices[id(vnode1)]
        y_i_3 = y_indices[id(vnode3)]
        conf_vnode1 = rules.rules[winners[y_i_1]].stats.conf
        conf_vnode3 = rules.rules[winners[y_i_3]].stats.conf
        labels1 = list(feature_extractor.labels_to_class_sequences[y_pred[y_i_1]])
        labels3 = list(feature_extractor.labels_to_class_sequences[y_pred[y_i_3]])
        if labels1[-1] not in quotes_classes or labels3[0] not in quotes_classes:
            _set_new_rule(feature_extractor, vnode1.y, rules, new_rules, processed_winners,
                          processed_y, y_i_1)
            _set_new_rule(feature_extractor, vnode3.y, rules, new_rules, processed_winners,
                          processed_y, y_i_3)
        elif labels1[-1] != labels3[0]:
            quote = labels1[-1] if conf_vnode1 >= conf_vnode3 else labels3[0]
            if labels1[-1] != quote:
                labels1[-1] = quote
                _set_new_rule(feature_extractor, tuple(labels1), rules, new_rules,
                              processed_winners, processed_y, y_i_1)
            else:
                labels3[0] = quote
                _set_new_rule(feature_extractor, tuple(labels3), rules, new_rules,
                              processed_winners, processed_y, y_i_3)
    return processed_y, processed_winners
