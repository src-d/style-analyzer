import importlib
import logging
from typing import Dict, Iterable, List, Mapping, NamedTuple, Optional, Sequence, Tuple, Set

import bblfsh
import numpy

from lookout.core.api.service_analyzer_pb2 import Comment
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

    def __eq__(self, other: "VirtualNode") -> bool:
        return self.value == other.value and \
               self.start == other.start and \
               self.end == other.end and \
               self.node == other.node and \
               self.y == other.y

    @staticmethod
    def from_node(node: bblfsh.Node, file: str) -> Iterable["VirtualNode"]:
        """
        Initializes the VirtualNode from a UAST node. Takes into account prefixes and suffixes.

        :param node: UAST node
        :param file: the file contents.
        :return: new VirtualNode-s
        """

        outer_token = file[node.start_position.offset:node.end_position.offset]
        if not node.token:
            yield VirtualNode(outer_token,
                              Position(*[f[1] for f in node.start_position.ListFields()]),
                              Position(*[f[1] for f in node.end_position.ListFields()]),
                              node=node)
            return
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

    def to_comment(self, correct_y: int) -> Comment:
        """
        Writes the comment with regard to the correct node class.
        :param correct_y: the index of the correct node class.
        :return: Lookout Comment object.
        """
        comment = Comment()
        comment.line = self.start.line
        if correct_y == CLASS_INDEX[CLS_NOOP]:
            comment.text = "format: %s at column %d should be removed" % (
                CLASSES[self.y], self.start.col)
        elif self.y == CLASS_INDEX[CLS_NOOP]:
            comment.text = "format: %s should be inserted at column %d" % (
                CLASSES[correct_y], self.start.col)
        else:
            comment.text = "format: replace %s with %s at column %d" % (
                CLASSES[self.y], CLASSES[correct_y], self.start.col)
        return comment


CLS_SPACE = "<space>"
CLS_SPACE_INC = "<+space>"
CLS_SPACE_DEC = "<-space>"
CLS_TAB = "<tab>"
CLS_TAB_INC = "<+tab>"
CLS_TAB_DEC = "<-tab>"
CLS_NEWLINE = "<newline>"
CLS_SINGLE_QUOTE = "'"
CLS_DOUBLE_QUOTE = '"'
CLS_NOOP = "<noop>"
CLASSES = (CLS_SPACE, CLS_TAB, CLS_NEWLINE, CLS_SPACE_INC, CLS_SPACE_DEC,
           CLS_TAB_INC, CLS_TAB_DEC, CLS_SINGLE_QUOTE, CLS_DOUBLE_QUOTE, CLS_NOOP)
CLASS_INDEX = {cls: i for i, cls in enumerate(CLASSES)}


class FeatureExtractor:

    self_features = ["start_line", "end_line", "start_col", "end_col"]
    left_siblings_features = ["start_line_diff", "end_line_diff", "start_col_diff", "end_col_diff",
                              "role_id"]
    right_siblings_features = ["role_id"]
    parents_features = ["role_id"]

    _log = logging.getLogger("FeaturesExtractor")

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

    def extract_features(self, files: Iterable[File], lines: List[List[int]]=None
                         ) -> Tuple[numpy.ndarray, numpy.ndarray, List[VirtualNode]]:
        """
        Given a list of `File`-s, compute the features and labels required for the training of
        downstream models.

        :param files: the list of `File`-s (see service_data.proto) of the same language.
        :param lines: the list of enabled line numbers per file. The lines which are not \
                      mentioned will not be extracted.
        :return: tuple of numpy.ndarray (2 and 1 dimensional respectively): features and labels \
                 and the corresponding `VirtualNode`-s.
        """
        parsed_files = []
        labels = []
        for i, file in enumerate(files):
            contents = file.content.decode("utf-8", "replace")
            uast = file.uast
            try:
                vnodes, parents = self._parse_file(contents, uast)
            except AssertionError as e:
                self._log.warning("could not parse file %s, skipped it", file.path,
                                  extra={"file": file.path, "unhandled": str(e)})
                continue
            vnodes = self._classify_vnodes(vnodes)
            vnodes = self._add_noops(vnodes)
            file_lines = set(lines[i]) if lines is not None else None
            parsed_files.append((vnodes, parents, file_lines))
            labels.append([vnode.y for vnode in vnodes if vnode.y and
                           (vnode.start.line in file_lines if file_lines is not None else True)])

        y = numpy.concatenate(labels)
        X = numpy.full(
            (y.shape[0],
             len(self.self_features)
             + self.siblings_window * len(self.left_siblings_features)
             + self.siblings_window * len(self.right_siblings_features)
             + self.parents_depth * len(self.parents_features)),
            -1)
        vn = [None] * y.shape[0]
        offset = 0
        for (vnodes, parents, file_lines), partial_labels in zip(parsed_files, labels):
            offset = self._inplace_write_vnode_features(vnodes, parents, file_lines, offset, X, vn)
        return X, y, vn

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

    @staticmethod
    def _add_noops(vnodes: Sequence[VirtualNode]) -> List[VirtualNode]:
        """
        Add CLS_NOOP nodes in between each node in the input sequence.

        :param vnodes: The sequence of `VirtualNode`-s to augment with noop nodes.
        :return: The augmented `VirtualNode`-s sequence.
        """
        result = [VirtualNode(value="", start=Position(0, 1, 1), end=Position(0, 1, 1),
                              y=CLASS_INDEX[CLS_NOOP])]
        for vnode in vnodes:
            result.append(vnode)
            result.append(VirtualNode(value="", start=vnode.end, end=vnode.end,
                                      y=CLASS_INDEX[CLS_NOOP]))
        return result

    def _get_role_index(self, vnode: VirtualNode) -> int:
        role_index = -1
        if vnode.node:
            role = vnode.node.internal_type
            if role in self.roles.ROLE_INDEX:
                role_index = self.roles.ROLE_INDEX[role]
        elif vnode.value in self.tokens.RESERVED_INDEX:
            role_index = len(self.roles.ROLE_INDEX) + self.tokens.RESERVED_INDEX[vnode.value]
        return role_index

    def _get_self_features(self, vnode: VirtualNode) -> Sequence[int]:
        return vnode.start.line, vnode.end.line, vnode.start.col, vnode.end.col

    def _get_left_sibling_features(self, left_sibling_vnode: VirtualNode, vnode: VirtualNode
                                   ) -> Sequence[int]:
        return (abs(left_sibling_vnode.start.line - vnode.start.line),
                abs(left_sibling_vnode.end.line - vnode.end.line),
                left_sibling_vnode.start.col - vnode.start.col,
                left_sibling_vnode.end.col - vnode.end.col,
                self._get_role_index(left_sibling_vnode))

    def _get_right_sibling_features(self, right_sibling_vnode: VirtualNode) -> Sequence[int]:
        return self._get_role_index(right_sibling_vnode),

    def _get_parent_features(self, parent_node: bblfsh.Node) -> Sequence[int]:
        return self.roles.ROLE_INDEX.get(parent_node.internal_type, -1),

    @staticmethod
    def _find_parent(vnode_index: int, vnodes: Sequence[VirtualNode],
                     parents: Mapping[int, bblfsh.Node], closest_left_node_id: int
                     ) -> Optional[bblfsh.Node]:
        """
        Compute current vnode parent. If both left and right closest parent are the same then we
        use it else we use no parent feature (set them to -1 later on)

        :param vnode_index: the index of the current node
        :param vnodes: the sequence of `VirtualNode`-s being transformed into features
        :param parents: the id of bblfsh node to parent bblfsh node mapping
        :param closest_left_parent: bblfsh node of the closest parent already gone through
        """
        left_ancestors = set()
        current_left_ancestor_id = closest_left_node_id
        while current_left_ancestor_id in parents:
            left_ancestors.add(id(parents[current_left_ancestor_id]))
            current_left_ancestor_id = id(parents[current_left_ancestor_id])

        for future_vnode in vnodes[vnode_index + 1:]:
            if future_vnode.node:
                break
        else:
            return None
        current_right_ancestor_id = id(future_vnode.node)
        while current_right_ancestor_id in parents:
            if id(parents[current_right_ancestor_id]) in left_ancestors:
                return parents[current_right_ancestor_id]
            current_right_ancestor_id = id(parents[current_right_ancestor_id])
        return None

    @staticmethod
    def _inplace_write_features(features: Sequence[int], row: int, col: int, X: numpy.ndarray
                                ) -> int:
        """
        Write features in X at the given location (row, col) and return the number of features
        written.

        :param features: the features
        :param row: the row where we should write
        :param col: the column where we should write
        :param X: the feature matrix
        :return: the number of features written
        """
        X[row, col:col + len(features)] = features
        return len(features)

    def _inplace_write_vnode_features(
            self, vnodes: Sequence[VirtualNode], parents: Mapping[int, bblfsh.Node],
            lines: Set[int], index_offset: int, X: numpy.ndarray, vn: List[VirtualNode]) -> int:
        """
        Given a sequence of `VirtualNode`-s and relevant info, compute the input matrix and label
        vector.

        :param vnodes: input sequence of `VirtualNode`s
        :param parents: dictionnary of node id to parent node
        :param lines: indices of lines to consider. 1-based.
        :param index_offset: at which index in the input ndarrays we should start writing
        :param X: features matrix, row per sample
        :param vn: list of the corresponding `VirtualNode`s, the length is the same as `X.shape[0]`
        :return: the new offset
        """
        closest_left_node_id = None
        position = index_offset
        for i, vnode in enumerate(vnodes):
            if vnode.node:
                closest_left_node_id = id(vnode.node)
            if not vnode.y or (lines is not None and vnode.start.line not in lines):
                continue
            if self.parents_depth:
                parent = self._find_parent(i, vnodes, parents, closest_left_node_id)
            else:
                parent = None

            parents_list = []
            if parent:
                current_ancestor = parent
                for j in range(self.parents_depth):
                    parents_list.append(current_ancestor)
                    current_ancestor_id = id(current_ancestor)
                    if current_ancestor_id not in parents:
                        break
                    current_ancestor = parents[current_ancestor_id]

            col_offset = 0

            # 1. write features of the current node
            col_offset += self._inplace_write_features(self._get_self_features(vnode),
                                                       position, col_offset, X)
            # 2. account for the possible lack of siblings by adjusting offset and write features
            # of the left siblings of the current node
            col_offset += abs(min(i - self.siblings_window, 0)) * len(self.left_siblings_features)
            for left_vnode in vnodes[max(0, i - self.siblings_window):i]:
                col_offset += self._inplace_write_features(
                    self._get_left_sibling_features(left_vnode, vnode), position, col_offset, X)

            # 3. write features of the right siblings of the current node and account for the
            # possible lack of siblings by adjusting offset
            for right_vnode in vnodes[i + 1:i + 1 + self.siblings_window]:
                col_offset += self._inplace_write_features(
                    self._get_right_sibling_features(right_vnode), position, col_offset, X)
            col_offset += (max(i + self.siblings_window - len(vnodes) + 1, 0)
                           * len(self.right_siblings_features))

            # 4. write features of the parents of the current node
            for parent_vnode in parents_list:
                col_offset += self._inplace_write_features(
                    self._get_parent_features(parent_vnode), position, col_offset, X)

            vn[position] = vnode
            position += 1
        return position

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
            if node.token or node.start_position and node.end_position and not node.children:
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
