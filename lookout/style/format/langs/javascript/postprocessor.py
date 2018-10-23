"""Prediction post processing module."""
from typing import Sequence, Tuple

from bblfsh import role_name
import numpy

from lookout.style.format.feature_utils import (CLASS_INDEX, CLS_DOUBLE_QUOTE, CLS_SINGLE_QUOTE,
                                                VirtualNode)
from lookout.style.format.rules import Rules


def postprocess(X: numpy.ndarray, y: numpy.ndarray, vnodes_y: Sequence[VirtualNode],
                vnodes: Sequence[VirtualNode], winners: numpy.ndarray, rules: Rules
                ) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """
    Post-process predictions to account for domain constraints.

    :param X: Feature matrix.
    :param y: Predictions to correct.
    :param vnodes_y: Sequence of the predicted virtual nodes.
    :param vnodes: Sequence of virtual nodes representing the input.
    :param winners: Indices of the rules that were used to compute the predictions.
    :param rules: Rules used to perform the prediction.
    :return: Updated y and winners.
    """
    quotes_classes = {CLASS_INDEX[CLS_DOUBLE_QUOTE], CLASS_INDEX[CLS_SINGLE_QUOTE]}
    processed_y = y.copy()
    processed_winners = winners.copy()
    vnodes_y_set = set(id(vnode) for vnode in vnodes_y)
    y_indices = {}
    for i, vnode in enumerate(vnodes):
        if id(vnode) in vnodes_y_set:
            y_indices[i] = len(y_indices)
    for i, y_i in y_indices.items():
        if not i + 2 in y_indices:
            continue
        y_i_paired = y_indices[i + 2]
        vnode, vnode_in_between, vnode_paired = vnodes[i], vnodes[i + 1], vnodes[i + 2]
        if vnode.y not in quotes_classes or vnode_paired.y not in quotes_classes:
            continue
        if vnode_in_between.node is None:
            continue
        vnode_in_between_roles = {role_name(role_id) for role_id in vnode_in_between.node.roles}
        if "STRING" in vnode_in_between_roles and y[y_i] != y[y_i_paired]:
            conf_vnode = rules.rules[winners[y_i]].stats.conf
            conf_vnode_paired = rules.rules[winners[y_i_paired]].stats.conf
            winner, loser = ((y_i, y_i_paired)
                             if conf_vnode > conf_vnode_paired
                             else (y_i_paired, y_i))
            processed_y[loser] = y[winner]
            processed_winners[loser] = winners[winner]
    return processed_y, processed_winners
