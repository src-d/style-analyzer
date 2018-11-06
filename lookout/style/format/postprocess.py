"""Postprocess predictions of the model."""
from typing import Mapping, Sequence, Tuple

import bblfsh
from bblfsh.client import BblfshClient
import numpy

from lookout.core.api.service_data_pb2 import File
from lookout.style.format.feature_extractor import FeatureExtractor
from lookout.style.format.feature_utils import CLS_DOUBLE_QUOTE, CLS_SINGLE_QUOTE, \
    INDEX_CLS_TO_STR, VirtualNode


def check_uasts_are_equal(uast1: bblfsh.Node, uast2: bblfsh.Node) -> bool:
    """
    Check if 2 UASTs are identical or not in terms of nodes `roles`, `internal_type` and `token`.

    :param uast1: The bblfsh.Node of the first UAST to compare.
    :param uast2: The bblfsh.Node of the second UAST to compare.
    :return: A boolean equals to True if the 2 input UASTs are identical and False otherwise.
    """
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


def node2lines(uast: bblfsh.Node) -> Tuple[int]:
    offsets = set()
    queue = [uast]
    while queue:
        node = queue.pop()
        for child in node.children:
            offsets.update({child.START_POSITION_FIELD_NUMBER, child.END_POSITION_FIELD_NUMBER})
        queue.extend(node.children)
    return min(offsets), max(offsets)


def filter_uast_breaking_preds(y: numpy.ndarray, y_pred: numpy.ndarray,
                               vnodes_y: Sequence[VirtualNode], vnodes: Sequence[VirtualNode], files: Mapping[str, File],
                               feature_extractor: FeatureExtractor, client: BblfshClient, vnodes_trace, parents
                               ) -> numpy.ndarray:
    """
    Filter and drop the model's predictions that modify the UAST apart from positioning.

    :param y: Numpy 1-dimensional array of labels.
    :param y_pred: Numpy 1-dimensional array of predicted labels by the model.
    :param vnodes_y: Sequence of the labeled `VirtualNode`-s corresponding to labeled samples.
    :param files: Dictionary of File-s with content, uast and path.
    :param feature_extractor: FeatureExtractor used to extract features.
    :param client: Babelfish client.
    :return: Numpy 1-dimensional array of predictions that do not modify the uast.
    """
    for i, (gt, pred, vn_y) in enumerate(zip(y, y_pred, vnodes_y)):
        if gt != pred:
            import pdb;pdb.set_trace()
            index = [vn.start for vn in vnodes].index(vn_y.start)
            #index = vnodes.index(vn_y)
            for vn in vnodes[:index + 1][::-1]:
                if vn.node:
                    closest_left_node_id = id(vn.node)
                    break
            else:
                closest_left_node_id = None
            parent = feature_extractor._find_parent(index, vnodes, parents[vn_y.path], closest_left_node_id)
            if parent is None:
                parent = files[vn_y.path].uast
            #import pdb;pdb.set_trace() # test lintervalle doffset de parent
            # get the right content
            #try:
            start, end = vnodes_trace[id(vn_y)]
            #except KeyError:
            #    if vn_y.start == (9, 1, 10) and vn_y.path == "/home/waren/sourced/data/tmp/noisy_repos/jquery_test/src/attributes/attr.js":
            #        print("Keyerror")
            #        print("vn :", vn_y)
            #        print("id(vn) :", id(vn_y))
            #import pdb;pdb.set_trace()


            #print("i :", i)
            content_before = files[vn_y.path].content
            if INDEX_CLS_TO_STR[pred] not in (CLS_SINGLE_QUOTE, CLS_DOUBLE_QUOTE):
                content_after = content_before[:vn_y.start.offset] \
                                + INDEX_CLS_TO_STR[pred].encode() \
                                + content_before[vn_y.start.offset+1:]
                content_after = content_after[start:end+1]
            #    uast_before = files[vn_y.path].uast
                parent_after = client.parse(filename="", contents=content_after,
                                          language=feature_extractor.language).uast
                if not check_uasts_are_equal(parent, parent_after):
                    print("X")
                    y_pred[i] = gt
                else:
                    print("O")
    return y_pred