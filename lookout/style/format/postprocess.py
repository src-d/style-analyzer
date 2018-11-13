"""Postprocess predictions of the model."""
from typing import Iterable, Mapping, Sequence

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
            if (child1.roles != child2.roles or child1.internal_type != child2.internal_type
                    or child1.token != child2.token):
                return False
        queue1.extend(node1.children)
        queue2.extend(node2.children)
    return True


def filter_uast_breaking_preds(y: numpy.ndarray, y_pred: numpy.ndarray,
                               vnodes_y: Sequence[VirtualNode], files: Mapping[str, File],
                               feature_extractor: FeatureExtractor, client: BblfshClient,
                               vnodes_parents: Mapping[int, bblfsh.Node],
                               parents: Mapping[str, bblfsh.Node]) -> Iterable[int]:
    """
    Filter the model's predictions that modify the UAST apart from positioning.

    :param y: Numpy 1-dimensional array of labels.
    :param y_pred: Numpy 1-dimensional array of predicted labels by the model.
    :param vnodes_y: Sequence of the labeled `VirtualNode`-s corresponding to labeled samples.
    :param files: Dictionary of File-s with content, uast and path.
    :param feature_extractor: FeatureExtractor used to extract features.
    :param client: Babelfish client.
    :param vnodes_parents: `VirtualNode`-s' parents mapping as the LCA of the closest
                           left and right babelfish nodes.
    :param parents: Parents mapping of the input UASTs.
    :return: List of predictions indices that are considered valid i.e. that are not breaking
             the UAST.
    """
    CLS_QUOTES = {CLS_SINGLE_QUOTE, CLS_DOUBLE_QUOTE}
    safe_preds = []
    for i, (gt, pred, vn_y) in enumerate(zip(y, y_pred, vnodes_y)):
        if gt != pred:
            content_before = files[vn_y.path].content
            parent = vnodes_parents[id(vn_y)]
            pred_string = "".join(INDEX_CLS_TO_STR[j] for j in
                                  feature_extractor.labels_to_class_sequences[pred])
            if not (CLS_QUOTES.intersection(vn_y.value) and CLS_QUOTES.intersection(pred_string)):
                errors_parsing = True
                while errors_parsing:
                    start, end = parent.start_position.offset, parent.end_position.offset
                    parse_response_before = client.parse(filename="",
                                                         contents=content_before[start:end],
                                                         language=feature_extractor.language)
                    if not parse_response_before.errors:
                        errors_parsing = parse_response_before.errors
                        diff_pred_offset = len(pred_string) - len(vn_y.value)
                        content_after = content_before[:vn_y.start.offset] \
                            + pred_string.encode() \
                            + content_before[vn_y.start.offset - diff_pred_offset + 1:]
                        content_after = content_after[start:end + diff_pred_offset]
                        parse_response_after = client.parse(filename="",
                                                            contents=content_after,
                                                            language=feature_extractor.language)
                        if not parse_response_after.errors:
                            parent_after = parse_response_after.uast
                            parent_before = parse_response_before.uast
                            if check_uasts_are_equal(parent_before, parent_after):
                                safe_preds.append(i)
                    else:
                        parent = parents[id(parent)]
            else:
                safe_preds.append(i)
        else:
            safe_preds.append(i)
    return safe_preds
