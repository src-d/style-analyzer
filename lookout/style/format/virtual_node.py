"""Defines VirtualNode - a class which backs any source code token."""
from typing import Callable, Iterable, Mapping, NamedTuple, Optional, Set, Tuple, Union

import bblfsh

from lookout.style.format.annotations.annotations import Annotation, RawTokenAnnotation, \
    UASTNodeAnnotation
from lookout.style.format.classes import CLASS_REPRESENTATIONS, EMPTY_CLS


class Position(NamedTuple("Position", (("offset", int), ("line", int), ("col", int)))):
    """
    Data class to hold position information of virtual nodes.

    `line` and `col` are 1-based to match UAST, `offset` is 0-based.
    """

    @staticmethod
    def from_bblfsh_position(position: bblfsh.Position) -> "Position":
        """Create a lookout.style.format.virtual_node.Position from a bblfsh.Position."""
        line = position.line
        col = position.col
        if hasattr(position, "offset"):
            offset = position.offset
        elif line == 1 and col == 1:
            offset = 0
        else:
            raise ValueError("Offset missing.")
        return Position(offset, line, col)


class VirtualNode:
    """
    Represent either a real UAST node or an imaginary token.

    The class instance should be considered as read-only.
    """

    def __init__(self, value: str, start: Position, end: Position,
                 *, node: bblfsh.Node = None, y: Optional[Tuple[int, ...]] = None,
                 is_accumulated_indentation: bool=False,  path: str = None) -> None:
        """
        Construct a VirtualNode.

        :param value: Text of the token.
        :param start: Starting position of the token.
        :param end: Ending position of the token.
        :param node: Corresponding UAST node (if exists).
        :param y: The label of the node. It can be either a predicted token class from CLASSES \
                  or a composite sequence of such classes. It is guaranteed that the final type \
                  is Tuple[int]; the plain integer is an intermediate "unmerged" value which \
                  is replaced during the class composition.
        :param is_accumulated_indentation: Marks virtual node as node with accumulated indentation.
        :param path: Path to related file. Useful for debugging.
        """
        self.value = value
        assert start.line >= 1 and start.col >= 1, "start line and column are 1-based like UASTs"
        assert end.line >= 1 and end.col >= 1, "end line and column are 1-based like UASTs"
        assert y is None or set(y) <= EMPTY_CLS or start.offset < end.offset, \
            "illegal empty node y=%s, start.offset=%d, end.offset=%d" % (
                str(y), start.offset, end.offset)
        assert not is_accumulated_indentation or y is None, "y can not be set for accumulated " \
                                                            "indentation node."
        self.start = start
        self.end = end
        self.node = node
        self.y = y
        self.is_accumulated_indentation = is_accumulated_indentation
        self.path = path

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return ("VirtualNode(%s, y=%s, start=%s, end=%s, node=%s, path=\"%s\")" % (
                    repr(self.value),
                    "None" if self.y is None else "".join(CLASS_REPRESENTATIONS[c]
                                                          for c in self.y),
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
                and self.path == other.path
                and self.is_accumulated_indentation == other.is_accumulated_indentation)

    def copy(self) -> "VirtualNode":
        """Produce a full clone of the node."""
        return VirtualNode(
            self.value, self.start, self.end, node=self.node, y=self.y,
            is_accumulated_indentation=self.is_accumulated_indentation, path=self.path)

    def is_labeled_on_lines(self, lines: Optional[Set[int]]) -> bool:
        """
        Return true for labeled VirtualNode instance that located on specified lines.

        :param lines: list of lines or None. None is considered as all possible lines.
        :return: Condition value
        """
        if self.y is None:
            return False
        return lines is None or bool(lines.intersection(range(self.start.line, self.end.line + 1)))

    @staticmethod
    def from_node(node: bblfsh.Node, file: str,
                  token_unwrappers: Mapping[str, Callable[[str], Tuple[str, str]]],
                  ) -> Iterable[Annotation]:
        """
        Initialize the VirtualNode from a UAST node. Takes into account prefixes and suffixes.

        :param node: UAST node.
        :param file: File contents.
        :param token_unwrappers: Mapping from bblfsh internal types to functions to unwrap tokens.
        :return: New VirtualNode-s.
        """
        # TODO(zurk): Move out from this class
        outer_token = file[node.start_position.offset:node.end_position.offset]
        if outer_token == "''" or outer_token == '""':
            assert node.token == "" or node.token == "''" or node.token == '""'
            middle = node.start_position.offset + 1
            yield RawTokenAnnotation(node.start_position.offset, middle)
            # In the future we can use only yield UASTNodeAnnotation.from_node(node)
            uast_annotation = UASTNodeAnnotation(middle, middle, node)
            yield RawTokenAnnotation(middle, middle, uast_annotation)
            yield uast_annotation
            yield RawTokenAnnotation(middle, node.end_position.offset)
            return
        if not node.token:
            uast_annotation = UASTNodeAnnotation.from_node(node)
            yield RawTokenAnnotation(node.start_position.offset, node.end_position.offset,
                                     uast_annotation)
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
            yield RawTokenAnnotation(node.start_position.offset, start_pos)
        end_pos = start_pos + len(node_token)
        uast_annotation = UASTNodeAnnotation(start_pos, end_pos, node)
        yield RawTokenAnnotation(start_pos, end_pos, uast_annotation)
        yield uast_annotation
        if end_pos < node.end_position.offset:
            yield RawTokenAnnotation(end_pos, node.end_position.offset)


AnyNode = Union[VirtualNode, bblfsh.Node]
