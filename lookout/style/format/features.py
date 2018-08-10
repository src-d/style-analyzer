import bblfsh
from typing import List

from lookout.core.api.service_data_pb2 import File


class WrappedNode:
    def __init__(self, start: int, end: int, node: bblfsh.Node, parent: bblfsh.Node,
                 start_line: int, start_col: int, end_line: int, end_col: int):
        """
        :param start: start offset of node.
        :param end: end offset of node.
        :param node: UAST node or string with operator/reserved keywords/etc.
        :param start_line: start line of node (1-based index).
        :param start_col: start column of node (1-based index). Useful for calculation of
                          indentation.
        :param end_line: end line of node (line of newline character is counted as the same line as
                      start line) (1-based index).
        :param end_col: end column of node (1-based index).
        :param parent: parent node.
        """
        self.start = start
        if end != 0:
            self.end = end
        else:
            self.end = start + len(node.token) + 1

        # start & end line/col are useful for nodes not from UAST
        self.start_line = start_line
        self.start_col = start_col
        self.end_line = end_line
        self.end_col = end_col
        self.node = node
        self.parent = parent

    @staticmethod
    def wrap(node: bblfsh.Node, parent: bblfsh.Node):
        """
        Convert node from UAST to common format.

        :param node: UAST node
        :param parent: parent of this node
        :return: new node
        """
        return WrappedNode(start=node.start_position.offset, end=node.end_position.offset,
                           node=node, parent=parent,
                           start_line=node.start_position.line, start_col=node.start_position.col,
                           end_line=node.end_position.line, end_col=node.end_position.col)


def extract_features(files: List[File], language: str, config: dict):
    """
    This is the dream interface for the feature extraction.

    :param files: the list of `File`-s (see service_data.proto) of the same language.
    :param language: the name of the language.
    :param config: feature extraction parameters.
    :return:
    """
    pass
