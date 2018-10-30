"""Prediction post processing module."""
from typing import MutableMapping, Sequence, Tuple

from bblfsh.client import BblfshClient
import bblfsh
import numpy

from lookout.style.format.feature_extractor import FeatureExtractor
from lookout.style.format.feature_utils import (CLASS_INDEX,
                                                INDEX_CHARACTER, VirtualNode)


def equal_uasts(uast1: bblfsh.Node, uast2: bblfsh.Node) -> bool:
    parents = {}
    queue1 = [uast1]
    queue2 = [uast2]
    while queue1 or queue2:
        try:
            node1 = queue1.pop()
            node2 = queue2.pop()
        except IndexError:
            return False
        for child1, child2 in zip(node1.children, node2.children):
            if child1.roles != child2.roles or child1.internal_type != child2.internal_type:
                return False
        queue1.extend(node1.children)
        queue2.extend(node2.children)
    return True


def filter_corrupting_preds(y: numpy.ndarray, y_pred: numpy.ndarray, vnodes_y: Sequence[VirtualNode],
                            content: str, uast_before: bblfsh.Node) -> Tuple[numpy.ndarray, numpy.ndarray]:
    for i, (gt, pred, vn_y) in enumerate(zip(y, y_pred, vnodes_y)):
        if gt != pred:
            try:
                content_after = content[:vn_y.start.offset] + INDEX_CHARACTER[pred].encode() + content[vn_y.start.offset+1:]
                bblfsh_client = BblfshClient("0.0.0.0:9432")
                uast_after = bblfsh_client.parse(filename="fake/path", contents=content_after, language="javascript").uast
                if not equal_uasts(uast_before, uast_after):
                    y_pred[i] *= -1
            except KeyError:
                continue
    return y_pred
