"""Facilities to create and use features."""
from typing import Callable, Iterable, Mapping, NamedTuple, Tuple

import bblfsh


Position = NamedTuple("Position", (("offset", int), ("line", int), ("col", int)))
"""
`line` and `col` are 1-based to match UAST!
"""


class VirtualNode:
    """Represent either a real UAST node or an imaginary token."""

    def __init__(self, value: str, start: Position, end: Position,
                 *, node: bblfsh.Node = None, y: int = None, path: str = None) -> None:
        """
        Construct a VirtualNode.

        :param value: Text of the token.
        :param start: Starting position of the token (0-based).
        :param end: Ending position of the token (0-based).
        :param node: Corresponding UAST node (if exists).
        :param y: The label of the node.
        :param path: Path to related file. Useful for debugging.
        """
        self.value = value
        assert start.line >= 1 and start.col >= 1, "start line and column are 1-based like UASTs"
        assert end.line >= 1 and end.col >= 1, "end line and column are 1-based like UASTs"
        assert y in EMPTY_CLS or start.offset < end.offset, "illegal empty node"
        self.start = start
        self.end = end
        self.node = node
        self.y = y
        self.path = path

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return ("VirtualNode(\"%s\", y=%s, start=%s, end=%s, node=%s, path=\"%s\")" % (
                    self.value.replace("\n", "\\n"),
                    "None" if self.y is None else self.y,
                    tuple(self.start),
                    tuple(self.end),
                    id(self.node) if self.node is not None else "None",
                    self.path))

    def __eq__(self, other: "VirtualNode") -> bool:
        return (self.value == other.value
                and self.start == other.start
                and self.end == other.end
                and self.node == other.node
                and self.y == other.y
                and self.path == other.path)

    @staticmethod
    def from_node(node: bblfsh.Node, file: str, path: str,
                  token_unwrappers: Mapping[str, Callable[[str], Tuple[str, str]]]
                  ) -> Iterable["VirtualNode"]:
        """
        Initialize the VirtualNode from a UAST node. Takes into account prefixes and suffixes.

        :param node: UAST node.
        :param file: File contents.
        :param path: File path.
        :param token_unwrappers: Mapping from bblfsh internal types to functions to unwrap tokens.
        :return: New VirtualNode-s.
        """
        outer_token = file[node.start_position.offset:node.end_position.offset]
        if not node.token:
            yield VirtualNode(outer_token,
                              Position(*[f[1] for f in node.start_position.ListFields()]),
                              Position(*[f[1] for f in node.end_position.ListFields()]),
                              node=node, path=path)
            return
        node_token = node.token
        if node.internal_type in token_unwrappers:
            node_token, outer_token = token_unwrappers[node.internal_type](outer_token)
        start_offset = outer_token.find(node_token)
        assert start_offset >= 0, (
            "Couldn't find the token in the specified position:\nNode role: %s\nParsed form: “%s”"
            "\nRaw form: “%s”\nStart position: %d, %d, %d\nEnd position: %d, %d, %d" % (
                node.internal_type, node_token, outer_token, node.start_position.offset,
                node.start_position.line, node.start_position.col, node.end_position.offset,
                node.end_position.line, node.end_position.col))
        start_pos = node.start_position.offset + start_offset
        if start_offset:
            yield VirtualNode(outer_token[:start_offset],
                              Position(offset=node.start_position.offset,
                                       line=node.start_position.line,
                                       col=node.start_position.col),
                              Position(offset=start_pos,
                                       line=node.start_position.line,
                                       col=node.start_position.col + start_offset),
                              path=path)
        end_offset = start_offset + len(node_token)
        end_pos = start_pos + len(node_token)
        yield VirtualNode(node_token,
                          Position(offset=start_pos,
                                   line=node.start_position.line,
                                   col=node.start_position.col + start_offset),
                          Position(offset=end_pos,
                                   line=node.end_position.line,
                                   col=node.start_position.col + end_offset),
                          node=node, path=path)
        if end_pos < node.end_position.offset:
            yield VirtualNode(outer_token[end_offset:],
                              Position(offset=end_pos,
                                       line=node.end_position.line,
                                       col=node.start_position.col + end_offset),
                              Position(offset=node.end_position.offset,
                                       line=node.end_position.line,
                                       col=node.end_position.col),
                              path=path)


CLS_SPACE = "<space>"
CLS_TAB = "<tab>"
CLS_NEWLINE = "<newline>"
CLS_SPACE_INC = "<+space>"
CLS_SPACE_DEC = "<-space>"
CLS_TAB_INC = "<+tab>"
CLS_TAB_DEC = "<-tab>"
CLS_SINGLE_QUOTE = "'"
CLS_DOUBLE_QUOTE = '"'
CLS_NOOP = "<noop>"
CLASSES = (CLS_SPACE, CLS_TAB, CLS_NEWLINE, CLS_SPACE_INC, CLS_SPACE_DEC,
           CLS_TAB_INC, CLS_TAB_DEC, CLS_SINGLE_QUOTE, CLS_DOUBLE_QUOTE, CLS_NOOP)
CLASS_INDEX = {cls: i for i, cls in enumerate(CLASSES)}
EMPTY_CLS = frozenset([CLASS_INDEX[CLS_TAB_DEC], CLASS_INDEX[CLS_SPACE_DEC],
                       CLASS_INDEX[CLS_NOOP]])
