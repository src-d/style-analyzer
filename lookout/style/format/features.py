import importlib
from typing import List, Dict, Tuple, NamedTuple, Iterable

import bblfsh
import numpy

from lookout.core.api.service_data_pb2 import File


Position = NamedTuple("Position", (("offset", int), ("line", int), ("col", int)))
"""
`line` and `col` are 1-based to match UAST!
"""


class VirtualNode:
    def __init__(self,  value: str, start: Position, end: Position,
                 *, node: bblfsh.Node=None, y: int=None):
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
    def __init__(self, language: str):
        self.tokens = importlib.import_module("lookout.style.format.langs.%s.tokens" % language)
        self.roles = importlib.import_module("lookout.style.format.langs.%s.roles" % language)

    def extract_features(self, files: List[File]):
        """
        This is the dream interface for the feature extraction.

        :param files: the list of `File`-s (see service_data.proto) of the same language.
        :param language: the name of the language.
        :param config: feature extraction parameters.
        :return:
        """
        pass

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

    def _parse_file(self, contents: str, root: bblfsh.Node) -> \
            Tuple[List[VirtualNode], Dict[int, bblfsh.Node]]:
        """
        Given the source text and the corresponding UAST this function compiles the list of
        `VirtualNode`-s and the parents mapping. That list of nodes equals to the original
        source text bit-to-bit after `"".join(n.value for n in nodes)`. `parents` map from
        `id(node)` to it's parent `bblfsh.Node`.

        :param contents: source file text
        :param root: UAST root node
        :param language: programming language of the file
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
