"""Postprocess predictions of the model."""
from logging import getLogger
from typing import Dict, Mapping, MutableMapping, Optional, Sequence, Tuple  # noqa: F401

import bblfsh
from lookout.core.api.service_data_pb2 import File
from lookout.core.data_requests import parse_uast
import numpy

from lookout.style.format.feature_extractor import FeatureExtractor
from lookout.style.format.rules import QuotedNodeTripleMapping
from lookout.style.format.virtual_node import VirtualNode


_log = getLogger("Postprocess")


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


def _parse_code(parent: bblfsh.Node, content: str, stub: "bblfsh.aliases.ProtocolServiceStub",
                parsing_cache: MutableMapping[int, Optional[Tuple[bblfsh.Node, int, int]]],
                language: str, node_parents: Mapping[int, bblfsh.Node], path: str,
                ) -> Optional[Tuple[bblfsh.Node, int, int]]:
    """
    Find a parent node that Babelfish can parse and parse it.

    Iterates over the parents of the current virtual node until it is parseable and returns the
    parsed UAST or None if it reaches the root without finding a parseable parent.

    The cache will be used to avoid recomputations for parents that have already been considered.

    :param parent: First virtual node to try to parse. Will go up in the tree if it fails.
    :param content: Content of the file.
    :param stub: Babelfish GRPC service stub.
    :param parsing_cache: Cache to avoid the recomputation of the results for already seen nodes.
    :param language: language to use for Babelfish.
    :param node_parents: Parents mapping of the input UASTs.
    :param path: Path of the file being parsed.
    :return: Optional tuple of the parsed UAST and the corresponding starting and ending offsets.
    """
    descendants = []
    current_ancestor = parent
    while current_ancestor is not None:
        if id(current_ancestor) in parsing_cache:
            result = parsing_cache[id(current_ancestor)]
            break
        descendants.append(current_ancestor)
        start, end = (current_ancestor.start_position.offset,
                      current_ancestor.end_position.offset)
        uast, errors = parse_uast(stub, content[start:end], filename="",
                                  language=language)
        if not errors:
            result = uast, start, end
            break
        current_ancestor = node_parents.get(id(current_ancestor), None)
    else:
        result = None
        _log.warning("skipped file %s, due to errors in parsing the whole content", path)
    for descendant in descendants:
        parsing_cache[id(descendant)] = result
    return result


def filter_uast_breaking_preds(
        y: numpy.ndarray, y_pred: numpy.ndarray, vnodes_y: Sequence[VirtualNode],
        vnodes: Sequence[VirtualNode], files: Mapping[str, File],
        feature_extractor: FeatureExtractor, stub: "bblfsh.aliases.ProtocolServiceStub",
        vnode_parents: Mapping[int, bblfsh.Node], node_parents: Mapping[int, bblfsh.Node],
        rule_winners: numpy.ndarray, grouped_quote_predictions: QuotedNodeTripleMapping,
        ) -> Tuple[numpy.ndarray, numpy.ndarray, Sequence[VirtualNode], numpy.ndarray,
                   Sequence[int]]:
    """
    Filter the model's predictions that modify the UAST apart from changing positions.

    :param y: Numpy 1-dimensional array of labels.
    :param y_pred: Numpy 1-dimensional array of predicted labels by the model.
    :param vnodes_y: Sequence of the labeled `VirtualNode`-s corresponding to labeled samples.
    :param vnodes: Sequence of all the `VirtualNode`-s corresponding to the input.
    :param files: Dictionary of File-s with content, uast and path.
    :param feature_extractor: FeatureExtractor used to extract features.
    :param stub: Babelfish GRPC service stub.
    :param vnode_parents: `VirtualNode`-s' parents mapping as the LCA of the closest
                           left and right babelfish nodes.
    :param node_parents: Parents mapping of the input UASTs.
    :param rule_winners: Numpy array of the index of the winning rule for each sample.
    :param grouped_quote_predictions: Quotes predictions (handled differenlty from the rest).
    :return: List of predictions indices that are considered valid i.e. that are not breaking
             the UAST.
    """
    safe_preds = []
    current_path = None  # type: Optional[str]
    parsing_cache = {}  # type: Dict[int, Optional[Tuple[bblfsh.Node, int, int]]]
    file_content = None  # type: Optional[str]
    cur_i = 0
    for i, (gt, pred, vn_y) in enumerate(zip(y, y_pred, vnodes_y)):
        if vn_y.path != current_path:
            parsing_cache = {}
            current_path = vn_y.path
            file_content = files[vn_y.path].content.decode("utf-8", "replace")
        while vn_y is not vnodes[cur_i]:
            cur_i += 1
            if cur_i >= len(vnodes):
                raise AssertionError("vnodes_y and vnodes are not consistent.")
        # quote types are special cased
        if id(vn_y) in grouped_quote_predictions:
            pred_string = feature_extractor.label_to_str(pred)
            group = grouped_quote_predictions[id(vn_y)]
            # already handled with the previous vnode
            if group is None:
                continue
            vnode1, vnode2, vnode3 = group
            content_before = file_content[vnode1.start.offset:vnode3.end.offset]
            content_after = (feature_extractor.label_to_str(y_pred[i]) + vnode2.value
                             + feature_extractor.label_to_str(y_pred[i + 1]))
            parsed_before, errors = parse_uast(stub, content_before, filename="",
                                               language=feature_extractor.language)
            if not errors:
                parsed_after, errors = parse_uast(stub, content_after, filename="",
                                                  language=feature_extractor.language)
                if check_uasts_are_equal(parsed_before, parsed_after):
                    safe_preds.extend((i, i + 1))
            continue
        if gt == pred:
            safe_preds.append(i)
            continue
        pred_string = feature_extractor.label_to_str(pred)
        parsed_before = _parse_code(vnode_parents[id(vn_y)], file_content, stub, parsing_cache,
                                    feature_extractor.language, node_parents, vn_y.path)
        if parsed_before is None:
            continue
        parent_before, start, end = parsed_before
        # when the input node value is NOOP i.e. an empty string, the replacement is restricted
        # to the first occurence
        output_pred = "".join(n.value for n in vnodes[cur_i:cur_i+2]).replace(vn_y.value,
                                                                              pred_string, 1)
        diff_pred_offset = len(pred_string) - len(vn_y.value)
        try:
            # to handle mixed indentations, we include the `VirtualNode` following the predicted
            # one in the output predicted string, and start the rest of the sequence one
            # `VirtualNode` further to avoid its repetitions
            start_next_vnode = vn_y.start.offset + len(vn_y.value) + len(vnodes[cur_i + 1].value)
            content_after = (file_content[:vn_y.start.offset]
                             + output_pred
                             + file_content[start_next_vnode:])
        # in case the prediction to check corresponds to the last label of a file
        except IndexError:
            content_after = file_content[:vn_y.start.offset] \
                + output_pred
        content_after = content_after[start:end + diff_pred_offset]
        parent_after, errors_after = parse_uast(
            stub, content_after, filename="", language=feature_extractor.language)
        if not errors_after:
            if check_uasts_are_equal(parent_before, parent_after):
                safe_preds.append(i)
    _log.info("Non UAST breaking predictions: %d selected out of %d",
              len(safe_preds), y_pred.shape[0])
    vnodes_y = [vn for i, vn in enumerate(list(vnodes_y)) if i in safe_preds]
    return y[safe_preds], y_pred[safe_preds], vnodes_y, rule_winners[safe_preds], safe_preds
