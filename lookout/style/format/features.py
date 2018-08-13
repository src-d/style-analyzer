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
                 *, node: bblfsh.Node=None):
        """
        This represents either a real UAST node or an imaginary token.

        :param value: text of the token.
        :param start: starting position of the token (0-based).
        :param end: ending position of the token (0-based).
        :param node: corresponding UAST node (if exists).
        """
        self.value = value
        self.start = start
        self.end = end
        self.node = node

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
        if start_offset:
            yield VirtualNode(outer_token[:start_offset],
                              Position(offset=node.start_position.offset,
                                       line=node.start_position.line,
                                       col=node.start_position.col),
                              Position(offset=node.start_position.offset + start_offset,
                                       line=node.start_position.line,
                                       col=node.start_position.col + start_offset))
        start_pos = start_offset + node.start_position.offset
        end_pos = start_offset + len(node.token)
        end_offset = node.end_position.offset - end_pos
        yield VirtualNode(node.token,
                          Position(offset=start_pos,
                                   line=node.start_position.line,
                                   col=node.start_position.col + start_offset),
                          Position(offset=end_pos,
                                   line=node.end_position.line,
                                   col=node.end_position.offset - end_offset),
                          node=node)
        if end_offset:
            yield VirtualNode(outer_token[end_pos:],
                              Position(offset=end_pos,
                                       line=node.end_position.line,
                                       col=node.end_position.col - end_offset),
                              Position(offset=node.end_position.offset,
                                       line=node.end_position.line,
                                       col=node.end_position.col))


CLS_NONE = "none"
CLS_SPACE = "space"
CLS_SPACE_BACK = "space-"
CLS_TAB = "tab"
CLS_TAB_BACK = "tab-"
CLS_NEWLINE = "newline"
CLS_SINGLE_QUOTE = "'"
CLS_DOUBLE_QUOTE = '"'
CLASSES = (CLS_NONE, CLS_SPACE, CLS_TAB, CLS_NEWLINE, CLS_SPACE_BACK, CLS_TAB_BACK,
           CLS_SINGLE_QUOTE, CLS_DOUBLE_QUOTE)
CLASS_INDEX = {cls: i for i, cls in enumerate(CLASSES)}


def parse_file(contents: str, root: bblfsh.Node, language: str) -> \
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
    tokens = importlib.import_module("lookout.style.format.langs.%s.tokens" % language)
    for node in node_tokens:
        if node.start_position.offset > pos:
            sumlen = 0
            diff = contents[pos:node.start_position.offset]
            for match in tokens.PARSER.finditer(diff):
                positions = []
                line = None
                for suboff in (match.start(), match.end()):
                    offset = pos + suboff
                    if line is None:
                        line = numpy.searchsorted(line_offsets, offset)
                    col = offset - line_offsets[line - 1]
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


def extract_features(files: List[File], language: str, config: dict):
    """
    This is the dream interface for the feature extraction.

    :param files: the list of `File`-s (see service_data.proto) of the same language.
    :param language: the name of the language.
    :param config: feature extraction parameters.
    :return:
    """
    pass
