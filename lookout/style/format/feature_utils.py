from typing import Callable, Iterable, Mapping, NamedTuple, Optional, Tuple

import bblfsh

from lookout.core.api.service_analyzer_pb2 import Comment

Position = NamedTuple("Position", (("offset", int), ("line", int), ("col", int)))
"""
`line` and `col` are 1-based to match UAST!
"""


class VirtualNode:
    def __init__(self, value: str, start: Position, end: Position,
                 *, node: bblfsh.Node = None, y: int = None, path: str = None,
                 global_index: Optional[int] = None, labeled_index: Optional[int] = None) -> None:
        """
        This represents either a real UAST node or an imaginary token.

        :param value: text of the token.
        :param start: starting position of the token (0-based).
        :param end: ending position of the token (0-based).
        :param node: corresponding UAST node (if exists).
        :param path: path to related file. Useful for debugging.
        :param global_index: position in the list of all nodes. Useful for debugging.
        :param labeled_index: position in the list of labeled nodes. Useful for debugging.
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
        self.global_index = global_index
        self.labeled_index = labeled_index

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return ("VirtualNode(\"%s\", y=%s, start=%s, end=%s, node=%s, path=\"%s\", "
                "global_index=%s, labeled_index=%s)" % (
                    self.value.replace("\n", "\\n"),
                    "None" if self.y is None else self.y,
                    tuple(self.start),
                    tuple(self.end),
                    id(self.node) if self.node is not None else "None",
                    self.path,
                    self.global_index,
                    self.labeled_index))

    def __eq__(self, other: "VirtualNode") -> bool:
        return (self.value == other.value
                and self.start == other.start
                and self.end == other.end
                and self.node == other.node
                and self.y == other.y
                and self.path == other.path
                and self.global_index == other.global_index
                and self.labeled_index == other.labeled_index)

    @staticmethod
    def from_node(node: bblfsh.Node, file: str, path: str,
                  token_unwrappers: Mapping[str, Callable[[str], Tuple[str, str]]]
                  ) -> Iterable["VirtualNode"]:
        """
        Initializes the VirtualNode from a UAST node. Takes into account prefixes and suffixes.

        :param node: UAST node
        :param file: the file contents
        :param path: the file path
        :param token_unwrappers: mapping from bblfsh internal types to functions to unwrap tokens
        :return: new VirtualNode-s
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
        comment.text = comment.text.replace("<", "`").replace(">", "`")
        return comment


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
