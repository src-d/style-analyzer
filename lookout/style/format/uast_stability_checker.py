"""Module to check if predictions change the UAST structure of the processed files."""
import difflib
from itertools import chain
from logging import getLogger
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union  # noqa: F401

import bblfsh
from lookout.core.api.service_data_pb2 import File
from lookout.core.data_requests import parse_uast
import numpy

from lookout.style.format.code_generator import CodeGenerationBaseError, CodeGenerator
from lookout.style.format.feature_extractor import FeatureExtractor
from lookout.style.format.rules import QuotedNodeTripleMapping
from lookout.style.format.virtual_node import VirtualNode


class UASTStabilityChecker:
    """
    Check if predictions change the UAST structure of the processed files.

    See `check()` or `file_check()` for more info.
    """

    _log = getLogger("UASTStabilityChecker")
    _check_return_type = Tuple[numpy.ndarray, numpy.ndarray, Sequence[VirtualNode], numpy.ndarray,
                               numpy.ndarray]

    def __init__(self, feature_extractor: FeatureExtractor, debug: bool=False):
        """
        Construct a UASTStabilityChecker.

        :param feature_extractor: Feature extraction class that was used to generate data for \
                                  the check.
        :param debug: Logs code diff for unsafe predictions with debug level.
        """
        self._feature_extractor = feature_extractor
        self._code_generator = CodeGenerator(self._feature_extractor, skip_errors=False)
        self._parsing_cache = {}  # type: Dict[int, Optional[Tuple[bblfsh.Node, int, int]]]
        self._debug = debug

    def _parse_code(self, vnode: VirtualNode, parent: bblfsh.Node, content: str,
                    stub: "bblfsh.aliases.ProtocolServiceStub",
                    node_parents: Mapping[int, bblfsh.Node], path: str,
                    ) -> Optional[Tuple[bblfsh.Node, int, int]]:
        """
        Find a parent node that Babelfish can parse and parse it.

        Iterates over the parents of the current virtual node until it is parsable and returns the
        parsed UAST or None if it reaches the root without finding a parsable parent.

        The cache will be used to avoid recomputations for parents that have already been
        considered.

        :param vnode: Vnode that is modified. Used to check that we retrieve the correct parent.
        :param parent: First virtual node to try to parse. Will go up in the tree if it fails.
        :param content: Content of the file.
        :param stub: Babelfish GRPC service stub.
        :param node_parents: Parents mapping of the input UASTs.
        :param path: Path of the file being parsed.
        :return: tuple of the parsed UAST and the corresponding starting and ending offsets. \
                 None if Babelfish failed to parse the whole file.
        """
        descendants = []
        current_ancestor = parent
        while current_ancestor is not None:
            if id(current_ancestor) in self._parsing_cache:
                result = self._parsing_cache[id(current_ancestor)]
                break
            descendants.append(current_ancestor)
            start, end = (current_ancestor.start_position.offset,
                          current_ancestor.end_position.offset)
            if start <= vnode.start.offset and end > vnode.end.offset:
                uast, errors = parse_uast(stub, content[start:end], filename="", unicode=True,
                                          language=self._feature_extractor.language)
                if not errors:
                    result = uast, start, end
                    break
            current_ancestor = node_parents.get(id(current_ancestor), None)
        else:
            result = None
            self._log.warning("skipped file %s, due to errors in parsing the whole content", path)
        for descendant in descendants:
            self._parsing_cache[id(descendant)] = result
        return result

    def _check_file(
            self, y: numpy.ndarray, y_pred: numpy.ndarray, vnodes_y: Sequence[VirtualNode],
            vnodes: Sequence[VirtualNode], file: File, stub: "bblfsh.aliases.ProtocolServiceStub",
            vnode_parents: Mapping[int, bblfsh.Node], node_parents: Mapping[int, bblfsh.Node],
            rule_winners: numpy.ndarray, grouped_quote_predictions: QuotedNodeTripleMapping,
    ) -> _check_return_type:
        # TODO(warenlg): Add current algorithm description.
        # TODO(vmarkovtsev): Apply ML to not parse all the parents.
        self._parsing_cache = {}
        unsafe_preds = set()
        file_content = file.content
        vnodes_i = 0
        changes = numpy.where((y_pred != -1) & (y != y_pred))[0]
        start_offset_to_vnodes = {}
        end_offset_to_vnodes = {}
        for i, vnode in enumerate(vnodes):
            if vnode.start.offset not in start_offset_to_vnodes:
                # NOOP always included
                start_offset_to_vnodes[vnode.start.offset] = i
        for i, vnode in enumerate(vnodes[::-1]):
            if vnode.end.offset not in end_offset_to_vnodes:
                # NOOP always included that is why we have reverse order in this loop
                end_offset_to_vnodes[vnode.end.offset] = len(vnodes) - i
        for i in changes:
            vnode_y = vnodes_y[i]
            while vnode_y is not vnodes[vnodes_i]:
                vnodes_i += 1
                if vnodes_i >= len(vnodes):
                    raise AssertionError("vnodes_y and vnodes are not consistent.")
            if id(vnode_y) in grouped_quote_predictions:
                # quote types are special case
                group = grouped_quote_predictions[id(vnode_y)]
                if group is None:
                    # already handled with the previous vnode
                    continue
                vnode1, vnode2, vnode3 = group
                content_before = file_content[vnode1.start.offset:vnode3.end.offset]
                content_after = (self._feature_extractor.label_to_str(y_pred[i]) + vnode2.value +
                                 self._feature_extractor.label_to_str(y_pred[i + 1]))
                parsed_before, errors = parse_uast(
                    stub, content_before, filename="", unicode=True,
                    language=self._feature_extractor.language)
                if not errors:
                    parsed_after, errors = parse_uast(
                        stub, content_after, filename="", unicode=True,
                        language=self._feature_extractor.language)
                    if not self.check_uasts_equivalent(parsed_before, parsed_after):
                        unsafe_preds.add(i)
                        unsafe_preds.add(i + 1)  # Second quote
                continue

            parsed_before = self._parse_code(vnode_y, vnode_parents[id(vnode_y)], file_content,
                                             stub, node_parents, vnode_y.path)
            if parsed_before is None:
                continue
            parent_before, start, end = parsed_before
            vnode_start_index = start_offset_to_vnodes[start]
            vnode_end_index = end_offset_to_vnodes[end]

            assert vnode_start_index <= vnodes_i < vnode_end_index, \
                "Tried to fix vnode %d by using vnodes %d to %d of %d total vnodes" % (
                    vnodes_i, vnode_start_index, vnode_end_index, len(vnodes))

            try:
                content_after = self._code_generator.generate_one_change(
                    vnodes[vnode_start_index:vnode_end_index],
                    vnodes_i - vnode_start_index, y_pred[i])
            except CodeGenerationBaseError as e:
                self._log.debug("Code generator can't generate code: %s", repr(e.args))
                unsafe_preds.add(i)
                continue
            parent_after, errors_after = parse_uast(
                stub, content_after, filename="", language=self._feature_extractor.language,
                unicode=True)
            if errors_after:
                unsafe_preds.add(i)
                continue
            if not self.check_uasts_equivalent(parent_before, parent_after):
                if self._debug:
                    self._log.debug(
                        "Bad prediction\nfile:%s\nDiff:\n%s\n\n", vnode_y.path,
                        "\n".join(line for line in difflib.unified_diff(
                            file_content[start:end].splitlines(), content_after.splitlines(),
                            fromfile="original", tofile="suggested", lineterm="")))
                unsafe_preds.add(i)
        self._log.info("%d filtered out of %d with changes", len(unsafe_preds), changes.shape[0])
        safe_preds = numpy.array([i for i in range(len(y)) if i not in unsafe_preds],
                                 dtype=numpy.int32)
        vnodes_y = [vn for i, vn in enumerate(list(vnodes_y)) if i not in unsafe_preds]
        return y[safe_preds], y_pred[safe_preds], vnodes_y, rule_winners[safe_preds], safe_preds

    def check(
            self, y: numpy.ndarray, y_pred: numpy.ndarray, vnodes_y: Sequence[VirtualNode],
            vnodes: Sequence[VirtualNode], files: Sequence[File],
            stub: "bblfsh.aliases.ProtocolServiceStub", vnode_parents: Mapping[int, bblfsh.Node],
            node_parents: Mapping[int, bblfsh.Node], rule_winners: numpy.ndarray,
            grouped_quote_predictions: QuotedNodeTripleMapping,
    ) -> _check_return_type:
        """
        Filter the model's predictions that modify the UAST apart from changing Node positions.

        :param y: Numpy 1-dimensional array of labels.
        :param y_pred: Numpy 1-dimensional array of predicted labels by the model.
        :param vnodes_y: Sequence of the labeled `VirtualNode`-s corresponding to labeled samples.
        :param vnodes: Sequence of all the `VirtualNode`-s corresponding to the input.
        :param files: File or Sequence of File-s with content, uast and path.
        :param stub: Babelfish GRPC service stub.
        :param vnode_parents: `VirtualNode`-s' parents mapping as the LCA of the closest \
                               left and right babelfish nodes.
        :param node_parents: Parents mapping of the input UASTs.
        :param rule_winners: Numpy array with the indexes of the winning rules for each sample.
        :param grouped_quote_predictions: Quotes predictions (handled differenlty from the rest).
        :return: List of predictions indices that are considered valid i.e. that are not breaking \
                 the UAST.
        """
        if len(files) == 1:
            return self._check_file(y, y_pred, vnodes_y, vnodes, files[0], stub, vnode_parents,
                                    node_parents, rule_winners, grouped_quote_predictions)
        # There is more then one file in data and data splitting is required.
        # The logic of the next code is about splitting mostly.
        current_path = vnodes_y[0].path
        file_vnodes_indexes = {current_path: [0, None]}
        for i, vnode in enumerate(vnodes):
            if vnode.path != current_path:
                file_vnodes_indexes[current_path][1] = i
                file_vnodes_indexes[vnode.path] = [i, None]
                current_path = vnode.path
        file_vnodes_indexes[current_path][1] = len(vnodes)
        files = {file.path: file for file in files}
        current_path = vnodes_y[0].path
        i_start = 0
        check_result = []
        for i, vnode_y in enumerate(vnodes_y):
            if vnode_y.path != current_path or i + 1 == len(vnodes_y):
                file_vnodes = vnodes[file_vnodes_indexes[current_path][0]:
                                     file_vnodes_indexes[current_path][1]]
                check_result.append(list(self._check_file(
                    y[i_start:i], y_pred[i_start:i], vnodes_y[i_start:i], file_vnodes,
                    files[current_path], stub, vnode_parents, node_parents,
                    rule_winners[i_start:i], grouped_quote_predictions)))
                check_result[-1][-1] += i_start
                current_path = vnode_y.path
                i_start = i

        res = []
        for res_i in zip(*check_result):
            if isinstance(res_i[0], list):
                res.append(list(chain(*res_i)))
            else:
                res.append(numpy.concatenate(res_i))
            pass
        return tuple(res)

    @staticmethod
    def check_uasts_equivalent(uast1: bblfsh.Node, uast2: bblfsh.Node) -> bool:
        """
        Check if 2 UAST nodes are identical regarding `roles`, `internal_type` and `token` of \
        their subtree members.

        :param uast1: The bblfsh.Node of the first UAST to compare.
        :param uast2: The bblfsh.Node of the second UAST to compare.
        :return: True if the 2 input UASTs are identical and False otherwise.
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
