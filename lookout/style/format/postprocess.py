"""Prediction post processing module."""
from typing import Mapping, Sequence

from bblfsh.client import BblfshClient
import bblfsh
import numpy

from lookout.core.api.service_data_pb2 import File
from lookout.style.format.feature_extractor import FeatureExtractor
from lookout.style.format.feature_utils import (
    CLS_SINGLE_QUOTE, CLS_DOUBLE_QUOTE, INDEX_CLS_TO_STR, VirtualNode)


def check_uasts_are_equal(uast1: bblfsh.Node, uast2: bblfsh.Node) -> bool:
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


def filter_uast_breaking_preds(y: numpy.ndarray, y_pred: numpy.ndarray,
                               vnodes_y: Sequence[VirtualNode], files: Mapping[str, File],
                               feature_extractor: FeatureExtractor, client: BblfshClient
                               ) -> numpy.ndarray:
    for i, (gt, pred, vn_y) in enumerate(zip(y, y_pred, vnodes_y)):
        if gt != pred:
            content_before = files[vn_y.path].content
            if INDEX_CLS_TO_STR[pred] not in (CLS_SINGLE_QUOTE, CLS_DOUBLE_QUOTE):
                content_after = content_before[:vn_y.start.offset] \
                                + INDEX_CLS_TO_STR[pred].encode() \
                                + content_before[vn_y.start.offset+1:]
                uast_before = files[vn_y.path].uast
                uast_after = client.parse(filename="", contents=content_after,
                                          language=feature_extractor.language).uast
                if not check_uasts_are_equal(uast_before, uast_after):
                    y_pred[i] = gt
    return y_pred
