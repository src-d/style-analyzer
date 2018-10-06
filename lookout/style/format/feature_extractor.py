"""Feature extraction module."""
from collections import OrderedDict
import importlib
import logging
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union

import bblfsh
import numpy
from sklearn.feature_selection import SelectKBest, VarianceThreshold

from lookout.core.api.service_data_pb2 import File
from lookout.style.format.feature_utils import (
    CLASS_INDEX, CLS_DOUBLE_QUOTE, CLS_NEWLINE, CLS_NOOP, CLS_SINGLE_QUOTE, CLS_SPACE,
    CLS_SPACE_DEC, CLS_SPACE_INC, CLS_TAB, CLS_TAB_DEC, CLS_TAB_INC, Position, VirtualNode)
from lookout.style.format.features import FeatureGroup, get_features


class FeatureExtractor:
    _log = logging.getLogger("FeaturesExtractor")

    def __init__(self, *, language: str, left_siblings_window: int, right_siblings_window: int,
                 parents_depth: int, node_features: Sequence[str], left_features: Sequence[str],
                 right_features: Sequence[str], parent_features: Sequence[str],
                 no_labels_on_right: bool, select_features_number: Optional[int],
                 remove_constant_features: bool, insert_noops: bool, debug_parsing: bool,
                 index_nodes: bool, selected_features: Optional[numpy.ndarray] = None) -> None:
        """
        Construct a `FeatureExtractor`.

        :param language: Which language to extract features for.
        :param left_siblings_window: How many siblings to use on the left of the current node.
        :param right_siblings_window: How many siblings to use on the right of the current node.
        :param parents_depth: How many parents to use for each node.
        :param node_features: Which features to compute for the current node.
        :param left_features: Which features to compute for the current node's left siblings.
        :param right_features: Which features to compute for the current node's right siblings.
        :param parent_features: Which features to compute for the current node's parents.
        :param no_labels_on_right: Whether to avoid using nodes with labels in right siblings.
        :param select_features_number: How many features to keep during feature selection.
        :param remove_constant_features: Whether to remove constant features
        :param insert_noops: Whether to insert noop nodes or not.
        :param debug_parsing: Whether to pause on parsing exceptions instead of skipping.
        :param index_nodes: Whether to compute the index of VirtualNodes for debugging.
        :param selected_features: Features to use. Skips further feature selection.
        """
        self.language = language.lower()
        self.left_siblings_window = left_siblings_window
        self.right_siblings_window = right_siblings_window
        self.parents_depth = parents_depth
        self.node_features = sorted(node_features)
        self.left_features = sorted(left_features)
        self.right_features = sorted(right_features)
        self.parent_features = sorted(parent_features)
        self.no_labels_on_right = no_labels_on_right
        self.select_features_number = select_features_number
        self.remove_constant_features = remove_constant_features
        self.selected_features = selected_features
        self.insert_noops = insert_noops
        self.debug_parsing = debug_parsing
        self.index_nodes = index_nodes
        self.tokens = importlib.import_module("lookout.style.format.langs.%s.tokens" % language)
        self.roles = importlib.import_module("lookout.style.format.langs.%s.roles" % language)
        try:
            self.token_unwrappers = importlib.import_module(
                "lookout.style.format.langs.%s.token_unwrappers" % language).TOKEN_UNWRAPPERS
        except ImportError:
            # It's normal for some languages not to have a token_unwrappers module.
            self.token_unwrappers = {}

        try:
            self.node_fixtures = importlib.import_module(
                "lookout.style.format.langs.%s.uast_fixers" % language).NODE_FIXTURES
        except ImportError:
            # It's normal for some languages not to have a uast_fixes module.
            self.node_fixtures = {}
        self._features = get_features(self.language)
        self._compute_feature_info()

    def _compute_feature_info(self) -> None:
        def populate_layout(feature_group: FeatureGroup, features: Sequence[str],
                            repetition: Optional[int]) -> None:
            if repetition is None:
                self._feature_layout[feature_group] = tuple(
                    "%s_%s" % (feature_group.name, feature_name)
                    for feature in features
                    for feature_name in self._features[feature].names)
            else:
                self._feature_layout[feature_group] = tuple(
                    "%s_%d_%s" % (feature_group.name, i, feature_name)
                    for i in range(repetition)
                    for feature in features
                    for feature_name in self._features[feature].names)

        self._feature_layout = OrderedDict()
        populate_layout(FeatureGroup.node, self.node_features, None)
        populate_layout(FeatureGroup.left, self.left_features, self.left_siblings_window)
        populate_layout(FeatureGroup.right, self.right_features, self.right_siblings_window)
        populate_layout(FeatureGroup.parents, self.parent_features, self.parents_depth)

        self._feature_names = tuple(name for names in self.feature_layout.values()
                                    for name in names)
        self._selected_feature_names = (
            self.feature_names if self.selected_features is None
            else [self._feature_names[x] for x in self.selected_features])
        self._feature2index = {name: i for i, name in enumerate(self.feature_names)}

    @property
    def feature_layout(self):
        return self._feature_layout

    @property
    def feature_names(self) -> Tuple[str]:
        return self._feature_names

    @property
    def selected_feature_names(self) -> Sequence[str]:
        return self._selected_feature_names

    @property
    def feature2index(self) -> Dict[str, int]:
        return self._feature2index

    def count_features(self, feature_group: FeatureGroup) -> int:
        """
        Return the number of features belonging to a specific group.

        `FeatureGroup.all` returns the overall number of features.
        """
        if feature_group == FeatureGroup.all:
            return len(self.feature_names)
        return len(self.feature_layout[feature_group])

    def extract_features(self, files: Iterable[File], lines: List[List[int]]=None
                         ) -> Optional[Tuple[numpy.ndarray, numpy.ndarray, List[VirtualNode]]]:
        """
        Given a list of `File`-s, compute the features and labels required for the training of
        downstream models.

        :param files: the list of `File`-s (see service_data.proto) of the same language.
        :param lines: the list of enabled line numbers per file. The lines which are not \
                      mentioned will not be extracted.
        :return: tuple of numpy.ndarray (2 and 1 dimensional respectively): features and labels \
                 and the corresponding `VirtualNode`-s or None in case not extracting features.
        """
        parsed_files = []
        labels = []
        for i, file in enumerate(files):
            contents = file.content.decode("utf-8", "replace")
            uast = file.uast
            try:
                vnodes, parents = self._parse_file(contents, uast, file.path)
            except AssertionError as e:
                self._log.warning("could not parse file %s with error '%s', skipping",
                                  file.path, e)
                if self.debug_parsing:
                    import traceback
                    traceback.print_exc()
                    input("Press Enter to continue…")
                continue
            vnodes = self._classify_vnodes(vnodes, file.path)
            if self.insert_noops:
                vnodes = self._add_noops(vnodes, file.path)
            else:
                vnodes = list(vnodes)
            if self.index_nodes:
                self._index_nodes(vnodes)
            file_lines = set(lines[i]) if lines is not None else None
            parsed_files.append((vnodes, parents, file_lines))
            labels.append([vnode.y for vnode in vnodes if vnode.y is not None and
                           (vnode.start.line in file_lines if file_lines is not None else True)])

        if not labels:
            # nothing was extracted
            return None

        y = numpy.concatenate(labels)
        X = numpy.zeros((y.shape[0], self.count_features(FeatureGroup.all)), dtype=numpy.uint8)
        vnodes_y = [None] * y.shape[0]
        offset = 0
        for vnodes, parents, file_lines in parsed_files:
            offset = self._inplace_write_vnode_features(vnodes, parents, file_lines, offset, X,
                                                        vnodes_y)
        self._log.debug("Features shape: %s" % (X.shape,))
        return X, y, vnodes_y, vnodes

    def select_features(self, X: numpy.ndarray, y: numpy.ndarray) -> Tuple[numpy.ndarray,
                                                                           numpy.ndarray]:
        """
        Select the most useful features based on sklearn's univariate feature selection.

        :param X: Numpy 2-dimensional array of features to select.
        :param y: Numpy 1-dimensional array of labels.
        :return: Tuple of numpy arrays: X with only the selected features (columns) kept and an \
                                        array of the indices of the kept features for later \
                                        reapplication.
        """
        if self.selected_features is not None:
            selected_features = self.selected_features
            X = X[:, selected_features]
        else:
            selected_features = None
            if self.remove_constant_features:
                feature_selector = VarianceThreshold()
                X = feature_selector.fit_transform(X)
                selected_features = feature_selector.get_support(indices=True)
            if self.select_features_number and self.select_features_number < X.shape[1]:
                feature_selector = SelectKBest(k=self.select_features_number)
                X = feature_selector.fit_transform(X, y)
                if selected_features is not None:
                    selected_features = selected_features[feature_selector.get_support(
                        indices=True)]
                else:
                    selected_features = feature_selector.get_support(indices=True)
        self._log.debug("Features shape after selection: %s" % (X.shape,))
        if selected_features is None:
            selected_features = numpy.arange(X.shape[1])
        return X, selected_features

    def _classify_vnodes(self, nodes: Iterable[VirtualNode], path: str) -> Iterable[VirtualNode]:
        """
        This function fills "y" attribute in the VirtualNode-s from _parse_file().
        It is the index of the corresponding class to predict.
        We detect indentation changes so several whitespace nodes are merged together.

        :param nodes: sequence of VirtualNodes.
        :param path: path to file.
        :return: new list of VirtualNodes, the size is different from the original.
        """
        indentation = []
        for node in nodes:
            if node.node is not None:
                yield node
                continue
            if not node.value.isspace():
                if node.value == "'":
                    node.y = CLASS_INDEX[CLS_SINGLE_QUOTE]
                elif node.value == '"':
                    node.y = CLASS_INDEX[CLS_DOUBLE_QUOTE]
                yield node
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
                    offset, lineno, col = node.start
                    yield VirtualNode(
                        char,
                        Position(offset + i, lineno, col + i),
                        Position(offset + i + 1, lineno, col + i + 1),
                        y=cls, path=path)
                continue
            line_offset = 0
            for i, line in enumerate(lines[:-1]):
                # `line` contains trailing whitespaces, we add it to the newline node
                newline = line + sep
                start_offset = node.start.offset + line_offset
                start_col = node.start.col if i == 0 else 1
                lineno = node.start.line + i
                yield VirtualNode(
                    newline,
                    Position(start_offset, lineno, start_col),
                    Position(start_offset + len(newline), lineno, start_col + len(newline)),
                    y=CLASS_INDEX[CLS_NEWLINE], path=path)
                line_offset += len(line) + len(sep)
            line = lines[-1]
            my_indent = list(line)
            offset, lineno, col = node.end
            try:
                for ws in indentation:
                    my_indent.remove(ws)
            except ValueError:
                if my_indent:
                    # mixed tabs and spaces, do not classify
                    yield VirtualNode(
                        line,
                        Position(offset - len(line), lineno, col - len(line)),
                        node.end, path=path)
                    continue
                # indentation decrease
                offset -= len(line)
                col -= len(line)
                for char in indentation[len(line):]:
                    if char == "\t":
                        cls = CLASS_INDEX[CLS_TAB_DEC]
                    else:
                        cls = CLASS_INDEX[CLS_SPACE_DEC]
                    yield VirtualNode(
                        "",
                        Position(offset, lineno, col),
                        Position(offset, lineno, col),
                        y=cls, path=path)
                indentation = indentation[:len(line)]
                if indentation:
                    yield VirtualNode(
                        "".join(indentation),
                        Position(offset, lineno, col),
                        node.end, path=path)
            else:
                if indentation:
                    yield VirtualNode(
                        "".join(indentation),
                        Position(offset - len(line), lineno, col - len(line)),
                        Position(offset - len(line) + len(indentation),
                                 lineno,
                                 col - len(line) + len(indentation)),
                        path=path)
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
                    yield VirtualNode(
                        char,
                        Position(offset + i, lineno, col + i),
                        Position(offset + i + 1, lineno, col + i + 1),
                        y=cls, path=path)

    @staticmethod
    def _add_noops(vnodes: Sequence[VirtualNode], path: str) -> List[VirtualNode]:
        """
        Add CLS_NOOP nodes in between each node in the input sequence.

        :param vnodes: The sequence of `VirtualNode`-s to augment with noop nodes.
        :param path: path to file.
        :return: The augmented `VirtualNode`-s sequence.
        """
        result = [VirtualNode(value="", start=Position(0, 1, 1), end=Position(0, 1, 1),
                              y=CLASS_INDEX[CLS_NOOP], path=path)]
        for vnode in vnodes:
            result.append(vnode)
            result.append(VirtualNode(value="", start=vnode.end, end=vnode.end,
                                      y=CLASS_INDEX[CLS_NOOP], path=path))
        return result

    @staticmethod
    def _index_nodes(vnodes: Sequence[VirtualNode]) -> None:
        global_index = 0
        labeled_index = 0
        for vnode in vnodes:
            vnode.global_index = global_index
            global_index += 1
            if vnode.y is not None:
                vnode.labeled_index = labeled_index
                labeled_index += 1

    def _get_features(self, features: Sequence[str], sibling: Union[VirtualNode, bblfsh.Node],
                      node: VirtualNode) -> List[int]:
        for feature in features:
            yield from self._features[feature](sibling, node)

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
        Write features starting at X[row, col] and return the number of features written.

        :param features: the features
        :param row: the row where we should write
        :param col: the column where we should write
        :param X: the feature matrix
        :return: the number of features written
        """
        to_write = [min(0xff, feature) for feature in features]
        X[row, col:col + len(to_write)] = to_write
        return len(to_write)

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
            if vnode.y is None or (lines is not None and vnode.start.line not in lines):
                continue
            if self.parents_depth:
                parent = self._find_parent(i, vnodes, parents, closest_left_node_id)
            else:
                parent = None

            parents_list = []
            if parent:
                current_ancestor = parent
                for _ in range(self.parents_depth):
                    parents_list.append(current_ancestor)
                    current_ancestor_id = id(current_ancestor)
                    if current_ancestor_id not in parents:
                        break
                    current_ancestor = parents[current_ancestor_id]

            if self.insert_noops:
                # First we define the ranges into which we find siblings when the current node is
                # NOOP. If the current node is NOOP, then its direct neighbours are interesting
                # (read non-NOOP) nodes.
                start_left = i - 1
                end_left = start_left - self.left_siblings_window * 2
                # For non-NOOP nodes, the first interesting nodes are further from the current node
                # since there are NOOP nodes in-between.
                if vnode.y != CLASS_INDEX[CLS_NOOP]:
                    start_left -= 1
                    end_left -= 1
                # We go two by two to avoid NOOP nodes.
                left_siblings = vnodes[max(start_left, 0):max(end_left, 0):-2]

                # If we don't compute right siblings without labeled nodes, we do the same for
                # them.
                if not self.no_labels_on_right:
                    start_right = i + 1
                    end_right = start_right + self.right_siblings_window * 2
                    if vnode.y != CLASS_INDEX[CLS_NOOP]:
                        start_right += 1
                        end_right += 1
                    right_siblings = vnodes[start_right:end_right:2]
            else:
                left_siblings = vnodes[max(i - 1, 0):max(i - self.left_siblings_window - 1, 0):-1]
                right_siblings = vnodes[i + 1:i + self.right_siblings_window + 1]

            if self.no_labels_on_right:
                right_siblings = []
                for j in range(i + 1, len(vnodes)):
                    if len(right_siblings) >= self.right_siblings_window:
                        break
                    if vnodes[j].y is None and not vnodes[j].value.isspace():
                        right_siblings.append(vnodes[j])

            col_offset = 0
            # 1. write features of the current node
            col_offset += self._inplace_write_features(
                self._get_features(self.node_features, vnode, None), position, col_offset, X)

            for left_vnode in left_siblings:
                col_offset += self._inplace_write_features(
                    self._get_features(self.left_features, left_vnode, vnode),
                    position, col_offset, X)
            if self.left_siblings_window > 0:
                col_offset += ((self.left_siblings_window - len(left_siblings))
                               * self.count_features(FeatureGroup.left)
                               // self.left_siblings_window)

            # 3. write features of the right siblings of the current node and account for the
            # possible lack of siblings by adjusting offset
            for right_vnode in right_siblings:
                col_offset += self._inplace_write_features(
                    self._get_features(self.right_features, right_vnode, vnode),
                    position, col_offset, X)
            if self.right_siblings_window > 0:
                col_offset += ((self.right_siblings_window - len(right_siblings))
                               * self.count_features(FeatureGroup.right)
                               // self.right_siblings_window)

            # 4. write features of the parents of the current node
            for parent_vnode in parents_list:
                col_offset += self._inplace_write_features(
                    self._get_features(self.parent_features, parent_vnode, vnode),
                    position, col_offset, X)

            vn[position] = vnode
            position += 1
        return position

    def _parse_file(self, contents: str, root: bblfsh.Node, path: str) -> \
            Tuple[List[VirtualNode], Dict[int, bblfsh.Node]]:
        """
        Given the source text and the corresponding UAST this function compiles the list of
        `VirtualNode`-s and the parents mapping. That list of nodes equals to the original
        source text bit-to-bit after `"".join(n.value for n in nodes)`. `parents` map from
        `id(node)` to its parent `bblfsh.Node`.

        :param contents: source file text
        :param root: UAST root node
        :param path: path to the file, used for debugging
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
            if node.internal_type in self.node_fixtures:
                node = self.node_fixtures[node.internal_type](node)
            for child in node.children:
                parents[id(child)] = node
            queue.extend(node.children)
            if (node.token or node.start_position and node.end_position
                    and node.start_position != node.end_position and not node.children):
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
            if node.start_position.offset < pos:
                continue
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
                    result.append(VirtualNode(token, *positions, path=path))
                assert sumlen == node.start_position.offset - pos, \
                    "missed some imaginary tokens: \"%s\"" % diff
            if node is sentinel:
                break
            result.extend(VirtualNode.from_node(node, contents, path, self.token_unwrappers))
            pos = node.end_position.offset
        return result, parents
