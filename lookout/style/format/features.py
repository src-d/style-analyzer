import importlib
from typing import Dict, Iterable, List, Mapping, NamedTuple, Optional, Sequence, Tuple

import bblfsh
import numpy

from lookout.core.api.service_data_pb2 import File

Position = NamedTuple("Position", (("offset", int), ("line", int), ("col", int)))
"""
`line` and `col` are 1-based to match UAST!
"""


class VirtualNode:
    def __init__(self,  value: str, start: Position, end: Position,
                 *, node: bblfsh.Node = None, y: int = None):
        """
        This represents either a real UAST node or an imaginary token.

        :param value: text of the token.
        :param start: starting position of the token (0-based).
        :param end: ending position of the token (0-based).
        :param node: corresponding UAST node (if exists).
        """
        self.value = value
        assert start.line >= 1 and start.col >= 1
        assert end.line >= 1 and end.col >= 1
        self.start = start
        self.end = end
        self.node = node
        self.y = y

    def __str__(self):
        return self.value

    def __repr__(self):
        return "VirtualNode(\"%s\", start=%s, end=%s, node=%s)" % (
            self.value, tuple(self.start), tuple(self.end),
            id(self.node) if self.node is not None else "None")

    @staticmethod
    def from_node(node: bblfsh.Node, file: str) -> Iterable["VirtualNode"]:
        """
        Initializes the VirtualNode from a UAST node. Takes into account prefixes and suffixes.

        :param node: UAST node
        :return: new VirtualNode-s
        """

        outer_token = file[node.start_position.offset:node.end_position.offset]
        start_offset = outer_token.find(node.token)
        start_pos = node.start_position.offset + start_offset
        if start_offset:
            yield VirtualNode(outer_token[:start_offset],
                              Position(offset=node.start_position.offset,
                                       line=node.start_position.line,
                                       col=node.start_position.col),
                              Position(offset=start_pos,
                                       line=node.start_position.line,
                                       col=node.start_position.col + start_offset))
        end_offset = start_offset + len(node.token)
        end_pos = start_pos + len(node.token)
        yield VirtualNode(node.token,
                          Position(offset=start_pos,
                                   line=node.start_position.line,
                                   col=node.start_position.col + start_offset),
                          Position(offset=end_pos,
                                   line=node.end_position.line,
                                   col=node.start_position.col + end_offset),
                          node=node)
        if end_offset:
            yield VirtualNode(outer_token[end_offset:],
                              Position(offset=end_pos,
                                       line=node.end_position.line,
                                       col=node.start_position.col + end_offset),
                              Position(offset=node.end_position.offset,
                                       line=node.end_position.line,
                                       col=node.end_position.col))


CLS_NONE = "<none>"
CLS_SPACE = "<space>"
CLS_SPACE_INC = "<+space>"
CLS_SPACE_DEC = "<-space>"
CLS_TAB = "<tab>"
CLS_TAB_INC = "<+tab>"
CLS_TAB_DEC = "<-tab>"
CLS_NEWLINE = "<newline>"
CLS_SINGLE_QUOTE = "'"
CLS_DOUBLE_QUOTE = '"'
CLASSES = (CLS_NONE, CLS_SPACE, CLS_TAB, CLS_NEWLINE, CLS_SPACE_INC, CLS_SPACE_DEC,
           CLS_TAB_INC, CLS_TAB_DEC, CLS_SINGLE_QUOTE, CLS_DOUBLE_QUOTE)
CLASS_INDEX = {cls: i for i, cls in enumerate(CLASSES)}


class FeatureExtractor:

    feature_names = ["start_offset", "start_line", "start_col", "end_offset", "end_line",
                     "end_col", "internal_role"]

    def __init__(self, language: str, siblings_window: int = 5, parents_depth: int = 2):
        """
        Construct a `FeatureExtractor`.

        :param parents_depth: how many parents to use for each node.
        :param siblings_window: how many siblings to use for each node (both left and right).
        """
        self.siblings_window = siblings_window
        self.parents_depth = parents_depth
        language = language.lower()
        self.tokens = importlib.import_module("lookout.style.format.langs.%s.tokens" % language)
        self.roles = importlib.import_module("lookout.style.format.langs.%s.roles" % language)

    def extract_features(self, files: List[File]) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """
        Given a list of `File`-s, compute the features and labels required for the training of
        downstream models.

        :param files: the list of `File`-s (see service_data.proto) of the same language.
        :return: tuple of numpy.ndarray (2 and 1 dimensional respectively): features and labels.
        """
        parsed_files = []
        labels = []
        for file in files:
            contents = file.content.decode("utf-8", "replace")
            uast = file.uast
            vnodes, parents = self._parse_file(contents, uast)
            vnodes = self._classify_vnodes(vnodes)
            parsed_files.append((vnodes, parents))
            labels.append([vnode.y for vnode in vnodes if vnode.y])

        y = numpy.concatenate(labels)
        X = numpy.full(
            (y.shape[0],
             (1 + self.siblings_window * 2 + self.parents_depth) * len(self.feature_names)),
            -1)
        line_offset = 0
        for (vnodes, parents), partial_labels in zip(parsed_files, labels):
            self._inplace_write_vnodes_features(vnodes, parents, line_offset, X)
            line_offset += len(labels)
        return X, y

    def _classify_vnodes(self, nodes: Iterable[VirtualNode]) -> List[VirtualNode]:
        """
        This function fills "y" attribute in the VirtualNode-s from _parse_file().
        It is the index of the corresponding class to predict.
        We detect indentation changes so several whitespace nodes are merged together.

        :param nodes: sequence of VirtualNodes.
        :return: new list of VirtualNodes, the size is different from the original.
        """
        indentation = []
        result = []
        for node in nodes:
            if node.node is not None:
                result.append(node)
                continue
            if not node.value.isspace():
                for cls in (CLS_SINGLE_QUOTE, CLS_DOUBLE_QUOTE):
                    if node.value == cls:
                        node.y = CLASS_INDEX[cls]
                        break
                result.append(node)
                continue
            lines = node.value.split("\r\n")
            if len(lines) > 1:
                sep = "\r\n"
            else:
                lines = node.value.split("\n")
                sep = "\n"
            if len(lines) == 1:
                # only tabs and spaces are possible
                for i, char in enumerate(node.value):
                    if char == "\t":
                        cls = CLASS_INDEX[CLS_TAB]
                    else:
                        cls = CLASS_INDEX[CLS_SPACE]
                    offset, line, col = node.start
                    result.append(VirtualNode(
                        char,
                        Position(offset + i, line, col + i),
                        Position(offset + i + 1, line, col + i + 1),
                        y=cls))
                continue
            line_offset = 0
            for i, line in enumerate(lines):
                if i < len(lines) - 1:
                    # `line` contains trailing whitespaces, we add it to the newline node
                    newline = line + sep
                    start_offset = node.start.offset + line_offset
                    start_col = node.start.col if i == 0 else 1
                    lineno = node.start.line + i
                    result.append(VirtualNode(
                        newline,
                        Position(start_offset, lineno, start_col),
                        Position(start_offset + len(newline), lineno, start_col + len(newline)),
                        y=CLASS_INDEX[CLS_NEWLINE]))
                    line_offset += len(line) + len(sep)
                    continue
                my_indent = list(line)
                offset, lineno, col = node.end
                try:
                    for ws in indentation:
                        my_indent.remove(ws)
                except ValueError:
                    if my_indent:
                        # mixed tabs and spaces, do not classify
                        result.append(VirtualNode(
                            line,
                            Position(offset - len(line), lineno, col - len(line)),
                            node.end))
                        continue
                    # indentation decrease
                    offset -= len(line)
                    col -= len(line)
                    for char in indentation[len(line):]:
                        if char == "\t":
                            cls = CLASS_INDEX[CLS_TAB_DEC]
                        else:
                            cls = CLASS_INDEX[CLS_SPACE_DEC]
                        result.append(VirtualNode(
                            "",
                            Position(offset, lineno, col),
                            Position(offset, lineno, col),
                            y=cls))
                    indentation = indentation[:len(line)]
                    result.append(VirtualNode(
                        "".join(indentation),
                        Position(offset, lineno, col),
                        node.end))
                else:
                    result.append(VirtualNode(
                        "".join(indentation),
                        Position(offset - len(line), lineno, col - len(line)),
                        Position(offset - len(line) + len(indentation),
                                 lineno,
                                 col - len(line) + len(indentation))))
                    offset += - len(line) + len(indentation)
                    col += - len(line) + len(indentation)
                    if not my_indent:
                        # indentation is the same
                        continue
                    # indentation increase
                    for i, char in enumerate(my_indent):
                        indentation.append(char)
                        if char == "\t":
                            cls = CLASS_INDEX[CLS_TAB_INC]
                        else:
                            cls = CLASS_INDEX[CLS_SPACE_INC]
                        result.append(VirtualNode(
                            char,
                            Position(offset + i, lineno, col + i),
                            Position(offset + i + 1, lineno, col + i + 1),
                            y=cls))
        return result

    def _inplace_write_vnode_features(self, vnode: VirtualNode, vnode_focused: VirtualNode,
                                      row: int, vnode_index: int, features: numpy.ndarray) -> None:
        """
        Given a feature `numpy.ndarray`, a `VirtualNode`, a row and a feature index, write the
        features of the vnode at the correct location in the feature matrix.

        :param vnode: `VirtualNode` we want to write the features of.
        :param vnode_focused: the focused `VirtualNode` we'll compare our characteristics with
        :param row: on which row to write the features.
        :param vnode_index: the column at which we'll start writing the features is \
                            vnode_index * n_features.
        :param features: `numpy.ndarray` the features matrix.
        """
        vnode_focused_pos = (vnode_focused.start.line, vnode_focused.end.line,
                             vnode_focused.start.col, vnode_focused.end.col)
        if vnode is vnode_focused:
            pos = vnode_focused_pos
        else:
            vnode_pos = (vnode.start.line, vnode.end.line, vnode.start.col, vnode.end.col)
            pos = tuple(map(lambda a, b: abs(a - b), vnode_focused_pos, vnode_pos))
        role_index = -1
        if vnode.node:
            role = vnode.node.internal_type
            if role in self.roles.ROLE_INDEX:
                role_index = self.roles.ROLE_INDEX[role]
        col = vnode_index * len(self.feature_names)
        features[row, col:col + 4] = pos
        features[row, col + 4] = role_index

    @staticmethod
    def _find_parent(vnode_index: int, vnodes: Sequence[VirtualNode],
                     parents: Mapping[int, bblfsh.Node], closest_left_parent: bblfsh.Node
                     ) -> Optional[bblfsh.Node]:
        """
        Compute current vnode parent. If both left and right closest parent are the same then we
        use it else we use no parent feature (set them to -1 later on)

        :param vnode_index: the index of the current node
        :param vnodes: the sequence of `VirtualNode`-s being transformed into features
        :param parents: the id of bblfsh node to parent bblfsh node mapping
        :oaram closest_left_parent: bblfsh node of the closest parent already gone through
        """
        next_right_parent = None
        for future_vnode in vnodes[vnode_index + 1:]:
            if future_vnode.node:
                next_right_parent = parents[id(future_vnode.node)]
                break
        return closest_left_parent if closest_left_parent is next_right_parent else None

    def _inplace_write_vnodes_features(self, vnodes: Sequence[VirtualNode],
                                       parents: Mapping[int, bblfsh.Node], line_offset: int,
                                       X: numpy.ndarray) -> None:
        """
        Given a sequence of `VirtualNode`-s and relevant info, compute the input matrix and label
        vector.

        :param vnodes: sequence of `VirtualNode`-s
        :param parents: dictionnary of node id to parent node
        :param line_offset: at which line of the input ndarrays should we start writing
        :param X: features matrix
        """
        closest_left_parent = None
        i = 0
        for vnode in vnodes:
            if vnode.node:
                closest_left_parent = parents[id(vnode.node)]
            if not vnode.y:
                continue
            parent = self._find_parent(i, vnodes, parents, closest_left_parent)

            # compute how many dummy features we'll need to account for the lack of left and right
            # siblings and the lack of parents
            n_dummies_left = abs(min(i - self.siblings_window, 0))
            parents_list = []
            if parent:
                current_vnode = vnode
                for j in range(self.parents_depth):
                    node_id = id(current_vnode)
                    if node_id not in parents:
                        break
                    parent = parents[node_id]
                    parents_list.append(parent)
                    current_vnode = parent

            # complete the feature ndarray by taking the dummies into account in 4 steps:
            # 1. write the node itself's features
            self._inplace_write_vnode_features(vnode, vnode, i + line_offset, 0, X)
            # 2. write the node's left siblings features
            # The offset is now 1 (the node) + n_dummies_left (if there are not enough siblings on
            # the left to obtain siblings_window features)
            for j, left_vnode in enumerate(vnodes[max(0, i - self.siblings_window):i]):
                self._inplace_write_vnode_features(left_vnode, vnode, i + line_offset,
                                                   1 + n_dummies_left + j, X)
            # 3. write the node's right siblings features
            # The offset is now 1 (the node) + siblings_window (the node's siblings on the left)
            for j, right_vnode in enumerate(vnodes[i + 1:i + 1 + self.siblings_window]):
                self._inplace_write_vnode_features(right_vnode, vnode, i + line_offset,
                                                   1 + self.siblings_window + j, X)
            # 4. write the node's parents features.
            # The offset is now 1 (the node) + 2 * siblings (the node's siblings on both sides)
            for j, parent_vnode in enumerate(parents_list):
                self._inplace_write_vnode_features(parent_vnode, vnode, i + line_offset,
                                                   1 + self.siblings_window * 2 + j, X)
            i += 1

    def _parse_file(self, contents: str, root: bblfsh.Node) -> \
            Tuple[List[VirtualNode], Dict[int, bblfsh.Node]]:
        """
        Given the source text and the corresponding UAST this function compiles the list of
        `VirtualNode`-s and the parents mapping. That list of nodes equals to the original
        source text bit-to-bit after `"".join(n.value for n in nodes)`. `parents` map from
        `id(node)` to its parent `bblfsh.Node`.

        :param contents: source file text
        :param root: UAST root node
        :return: list of `VirtualNode`-s and the parents.
        """
        # build the line mapping
        lines = contents.split("\n")
        line_offsets = numpy.zeros(len(lines) + 1, dtype=numpy.int32)
        pos = 0
        for i, line in enumerate(lines):
            line_offsets[i] = pos
            pos += len(line) + 1
        line_offsets[-1] = pos

        # walk the tree: collect nodes with assigned tokens and build the parents map
        node_tokens = []
        parents = {}
        queue = [root]
        while queue:
            node = queue.pop()
            for child in node.children:
                parents[id(child)] = node
            queue.extend(node.children)
            if node.token:
                node_tokens.append(node)
        node_tokens.sort(key=lambda n: n.start_position.offset)
        sentinel = bblfsh.Node()
        sentinel.start_position.offset = len(contents)
        sentinel.start_position.line = len(lines)
        node_tokens.append(sentinel)

        # scan `node_tokens` and fill the gaps with imaginary nodes
        result = []
        pos = 0
        parser = self.tokens.PARSER
        searchsorted = numpy.searchsorted
        for node in node_tokens:
            if node.start_position.offset > pos:
                sumlen = 0
                diff = contents[pos:node.start_position.offset]
                for match in parser.finditer(diff):
                    positions = []
                    for suboff in (match.start(), match.end()):
                        offset = pos + suboff
                        line = searchsorted(line_offsets, offset, side="right")
                        col = offset - line_offsets[line - 1] + 1
                        positions.append(Position(offset, line, col))
                    token = match.group()
                    sumlen += len(token)
                    result.append(VirtualNode(token, *positions))
                assert sumlen == node.start_position.offset - pos, \
                    "missed some imaginary tokens: \"%s\"" % diff
            if node is sentinel:
                break
            result.extend(VirtualNode.from_node(node, contents))
            pos = node.end_position.offset
        return result, parents
