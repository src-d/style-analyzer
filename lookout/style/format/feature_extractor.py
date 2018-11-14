"""Feature extraction module."""
from collections import OrderedDict
from enum import Enum, unique
import importlib
from itertools import islice
import logging
from typing import (Dict, Iterable, List, Mapping, MutableMapping, Optional,
                    Sequence, Set, Tuple, Union)

import bblfsh
import numpy
from sklearn.feature_selection import SelectKBest, VarianceThreshold

from lookout.core.api.service_data_pb2 import File
from lookout.style.format.feature_utils import (
    CLASS_INDEX, CLS_DOUBLE_QUOTE, CLS_NEWLINE, CLS_NOOP, CLS_SINGLE_QUOTE, CLS_SPACE,
    CLS_SPACE_DEC, CLS_SPACE_INC, CLS_TAB, CLS_TAB_DEC, CLS_TAB_INC, Position, VirtualNode)
from lookout.style.format.features import get_features, MultipleValuesFeature


@unique
class FeatureGroup(Enum):
    """
    Feature groups.

    Each feature belongs to one and only one of these classes.
    """

    node = 1
    parents = 2
    left = 3
    right = 4

    def format(self, value) -> str:
        """
        Represent the feature group for user interfaces. The trailing dot is appended as needed.

        :param value: The feature group parameter. E.g., the index of the node for "left" \
                      and "right".
        :return: pretty-printed string, the trailing dot is appended as needed.
        """
        if self == FeatureGroup.node:
            return "•••"
        if self == FeatureGroup.parents:
            return "^%d." % (value + 1)  # ↑ is displayed like shit in Ubuntu
        if self == FeatureGroup.left:
            return str(-value - 1) + "."
        if self == FeatureGroup.right:
            return "+%s." % (value + 1)
        return "%s.%s." % (self.name, value)


FeatureToIndex = MutableMapping[FeatureGroup, List[MutableMapping[str, List[int]]]]
IndexToFeature = List[Tuple[FeatureGroup, int, str, int]]

FEATURES_NUMPY_TYPE = numpy.uint8
FEATURES_MIN = numpy.iinfo(FEATURES_NUMPY_TYPE).min
FEATURES_MAX = numpy.iinfo(FEATURES_NUMPY_TYPE).max


class FeatureExtractor:
    """Extract features for downstream models."""

    _log = logging.getLogger("FeaturesExtractor")

    def __init__(self, *, language: str, left_siblings_window: int, right_siblings_window: int,
                 parents_depth: int, node_features: Sequence[str], left_features: Sequence[str],
                 right_features: Sequence[str], parent_features: Sequence[str],
                 no_labels_on_right: bool, select_features_number: Optional[int],
                 remove_constant_features: bool, insert_noops: bool, debug_parsing: bool,
                 return_sibling_indices: bool, selected_features: Optional[numpy.ndarray] = None,
                 label_composites: Optional[List[Tuple[int, ...]]] = None) -> None:
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
        :param return_sibling_indices: Whether to return the indices of siblings of the predicted \
                                       nodes.
        :param selected_features: Feature indexes to use. If None ("train" stage), we learn them \
                                  in select_features(). Otherwise, we are in "analyze" stage.
        :param label_composites: Maps composite output classes to the corresponding sequences of \
                                 "atomic" classes. If None or empty ("train" stage), we build \
                                 this mapping inside `extract_features()`. Otherwise, we are in \
                                 "analyze" stage.
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
        self.insert_noops = insert_noops
        self.debug_parsing = debug_parsing
        self.return_sibling_indices = return_sibling_indices
        self.selected_features = selected_features
        self.labels_to_class_sequences = label_composites if label_composites is not None else []
        self.class_sequences_to_labels = {
            tuple(l): i for i, l in enumerate(self.labels_to_class_sequences)}
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
        if self.labels_to_class_sequences:
            self._features = get_features(self.language, self.labels_to_class_sequences)
            self._compute_feature_info()

    def _compute_feature_info(self) -> None:
        if self.selected_features is not None:
            selected_features_set = set(self.selected_features)

        def populate_indices(feature_group: FeatureGroup, feature_names: Sequence[str],
                             nodes_number: int, total_index: int) -> int:
            self._feature_to_indices[feature_group] = []
            self._feature_to_indices_set[feature_group] = []
            for node_index in range(nodes_number):
                self._feature_to_indices[feature_group].append(OrderedDict())
                self._feature_to_indices_set[feature_group].append(OrderedDict())
                for feature_name in feature_names:
                    feature = self._features[feature_name]
                    names = list(feature.names)
                    self._feature_to_indices[feature_group][node_index][feature_name] = []
                    self._feature_to_indices_set[feature_group][node_index][feature_name] = set()
                    for index in range(len(list(self._features[feature_name].names))):
                        if (self.selected_features is None
                                or total_index in selected_features_set):
                            if isinstance(feature, MultipleValuesFeature):
                                name = "%s.%s" % (feature.id.name, names[index])
                            else:
                                name = feature.id.name
                            self._feature_names.append("%s.%d.%s" % (feature_group.name,
                                                                     node_index, name))
                            self._feature_to_indices[feature_group][node_index][feature_name] \
                                .append(total_index)
                            self._feature_to_indices_set[feature_group][node_index][feature_name] \
                                .add(index)
                            self._index_to_feature.append((feature_group, node_index, feature_name,
                                                           index))
                        total_index += 1
            return total_index
        self._index_to_feature = []  # type: IndexToFeature
        self._feature_to_indices = OrderedDict()  # type: FeatureToIndex
        self._feature_names = []  # type: List[str]
        # Not exposed through properties, only used during feature extraction.
        self._feature_to_indices_set = OrderedDict()
        total_index = 0
        total_index = populate_indices(FeatureGroup.node, self.node_features, 1, total_index)
        total_index = populate_indices(FeatureGroup.left, self.left_features,
                                       self.left_siblings_window, total_index)
        total_index = populate_indices(FeatureGroup.right, self.right_features,
                                       self.right_siblings_window, total_index)
        total_index = populate_indices(FeatureGroup.parents, self.parent_features,
                                       self.parents_depth, total_index)
        # we don't need the last `total_index`

        self._feature_node_counts = {group: [sum(len(feature) for feature in node_index.values())
                                             for node_index in self.feature_to_indices[group]]
                                     for group in FeatureGroup}
        self._feature_group_counts = {group: sum(counts)
                                      for group, counts in self._feature_node_counts.items()}
        self._feature_count = sum(self._feature_group_counts.values())

    @property
    def index_to_feature(self) -> IndexToFeature:
        """Return the mapping from integer indices to the corresponding feature names."""
        return self._index_to_feature

    @property
    def feature_to_indices(self) -> FeatureToIndex:
        """Return the mapping from feature names to the corresponding integer indices."""
        return self._feature_to_indices

    @property
    def feature_names(self) -> List[str]:
        """
        Return the names of the features.

        A feature name uniquely identifies a feature. It reflects the feature structure: it is
        comprised of the feature group the feature belongs to, its sibling identifier, its feature
        name and its index if applicable.

        Those names, as well as the feature layout, depend on the configuration used to launch the
        analyzer.
        """
        return self._feature_names

    def count_features(self, feature_group: Optional[FeatureGroup] = None,
                       neighbour_index: Optional[int] = None) -> int:
        """Return the feature count of a given subset of features."""
        if feature_group is None:
            return self._feature_count
        if neighbour_index is None:
            return self._feature_group_counts[feature_group]
        return self._feature_node_counts[feature_group][neighbour_index]

    def extract_features(self, files: Iterable[File], lines: List[List[int]]=None
                         ) -> Optional[Union[
                             Tuple[numpy.ndarray, numpy.ndarray,
                                   List[VirtualNode], List[VirtualNode]],
                             Tuple[numpy.ndarray, numpy.ndarray,
                                   List[VirtualNode], List[VirtualNode], List[List[int]]]]]:
        """
        Compute features and labels required by downstream models given a list of `File`-s.

        :param files: the list of `File`-s (see service_data.proto) of the same language.
        :param lines: the list of enabled line numbers per file. The lines which are not \
                      mentioned will not be extracted.
        :return: tuple of numpy.ndarray (2 and 1 dimensional respectively): features and labels \
                 and the corresponding `VirtualNode`-s or None in case not extracting features.
        """
        parsed_files = []
        index_labels = not self.labels_to_class_sequences
        for i, file in enumerate(files):
            contents = file.content.decode("utf-8", "replace")
            uast = file.uast
            try:
                file_vnodes, file_parents = self._parse_file(contents, uast, file.path)
            except AssertionError as e:
                self._log.warning("could not parse file %s with error '%s', skipping",
                                  file.path, e)
                if self.debug_parsing:
                    import traceback
                    traceback.print_exc()
                    input("Press Enter to continue…")
                continue
            file_vnodes = self._classify_vnodes(file_vnodes, file.path)
            file_vnodes = self._merge_classes_to_composite_labels(
                file_vnodes, file.path, index_labels=index_labels)
            if self.insert_noops:
                file_vnodes = self._add_noops(list(file_vnodes), file.path,
                                              index_labels=index_labels)
            else:
                file_vnodes = list(file_vnodes)
            file_lines = set(lines[i]) if lines is not None and lines[i] is not None else None
            parsed_files.append((file_vnodes, file_parents, file_lines))

        labels = [[self.class_sequences_to_labels[vnode.y]
                   for vnode in file_vnodes if vnode.is_labeled_on_lines(file_lines)]
                  for file_vnodes, file_parents, file_lines in parsed_files]

        if not labels:
            # nothing was extracted
            return None

        if index_labels:
            self._features = get_features(self.language, self.labels_to_class_sequences)
            self._compute_feature_info()

        y = numpy.concatenate(labels)
        X = numpy.zeros((y.shape[0], self.count_features()), dtype=FEATURES_NUMPY_TYPE)
        vnodes = []
        vnodes_y = [None] * y.shape[0]
        offset = 0
        if self.return_sibling_indices:
            sibling_indices_list = []
        for file_vnodes, file_parents, file_lines in parsed_files:
            vnodes.extend(file_vnodes)
            offset, sibling_indices = self._inplace_write_vnode_features(
                file_vnodes, file_parents, file_lines, offset, X, vnodes_y)
            if self.return_sibling_indices:
                sibling_indices_list.extend(sibling_indices)
        self._log.debug("Features shape: %s" % (X.shape,))
        if self.return_sibling_indices:
            return X, y, vnodes_y, vnodes, sibling_indices_list
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
            return X, self.selected_features
        if self.remove_constant_features:
            feature_selector = VarianceThreshold()
            X = feature_selector.fit_transform(X)
            self.selected_features = feature_selector.get_support(indices=True)
        if self.select_features_number and self.select_features_number < X.shape[1]:
            feature_selector = SelectKBest(k=self.select_features_number)
            X = feature_selector.fit_transform(X, y)
            if self.selected_features is not None:
                self.selected_features = self.selected_features[feature_selector.get_support(
                    indices=True)]
            else:
                self.selected_features = feature_selector.get_support(indices=True)
        self._log.debug("Features shape after selection: %s" % (X.shape,))
        if self.selected_features is None:
            self.selected_features = numpy.arange(X.shape[1])
        self._compute_feature_info()
        return X, self.selected_features

    def _classify_vnodes(self, nodes: Iterable[VirtualNode], path: str) -> Iterable[VirtualNode]:
        """
        Fill "y" attribute in the VirtualNode-s extracted from _parse_file().

        It is the index of the corresponding class to predict. We detect indentation changes so
        several whitespace nodes are merged together.

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
            offset -= len(line)
            col -= len(line)
            try:
                for ws in indentation:
                    my_indent.remove(ws)
            except ValueError:
                if my_indent:
                    # mixed tabs and spaces, do not classify
                    yield VirtualNode(
                        line,
                        Position(offset, lineno, col),
                        node.end, path=path)
                    continue
                # indentation decreases
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
                # indentation is stable or increases
                for i, char in enumerate(my_indent):
                    if char == "\t":
                        cls = CLASS_INDEX[CLS_TAB_INC]
                    else:
                        cls = CLASS_INDEX[CLS_SPACE_INC]
                    yield VirtualNode(
                        char,
                        Position(offset + i, lineno, col + i),
                        Position(offset + i + 1, lineno, col + i + 1),
                        y=cls, path=path)
                offset += len(my_indent)
                col += len(my_indent)
                if indentation:
                    yield VirtualNode(
                        "".join(indentation),
                        Position(offset, lineno, col),
                        Position(offset + len(indentation), lineno, col + len(indentation)),
                        path=path)
                for char in my_indent:
                    indentation.append(char)

    def _merge_classes_to_composite_labels(
            self, vnodes: Iterable[VirtualNode], path: str, index_labels: bool = False
            ) -> Iterable[VirtualNode]:
        """
        Pack successive predictable nodes into single "composite" labels.

        :param vnodes: Iterable of `VirtualNode`-s to process.
        :param path: Path to the file from which we are currently extracting features.
        :param index_labels: Whether to index labels to define output classes or not.
        :yield: The sequence of `VirtualNode`-s which is identical to the input but \
                the successive Y-nodes are merged together.
        """
        start, end, value, current_class_seq = None, None, "", []
        for vnode in vnodes:
            if vnode.y is None:
                if current_class_seq:
                    class_seq = tuple(current_class_seq)
                    if class_seq not in self.class_sequences_to_labels:
                        if index_labels:
                            self.class_sequences_to_labels[class_seq] = \
                                len(self.class_sequences_to_labels)
                            self.labels_to_class_sequences.append(class_seq)
                        else:
                            class_seq = None
                    yield VirtualNode(value=value, start=start, end=end,
                                      y=class_seq, path=path)
                    start, end, value, current_class_seq = None, None, "", []
                yield vnode
            else:
                if not current_class_seq:
                    start = vnode.start
                end = vnode.end
                value += vnode.value
                current_class_seq.append(vnode.y)
        if value or current_class_seq:
            yield VirtualNode(
                value=value, start=start, end=end, y=tuple(current_class_seq), path=path)

    def _add_noops(self, vnodes: Sequence[VirtualNode], path: str, index_labels: bool = False
                   ) -> List[VirtualNode]:
        """
        Add CLS_NOOP nodes in between tokens without labeled nodes to allow for insertions.

        :param vnodes: The sequence of `VirtualNode`-s to augment with noop nodes.
        :param path: path to file.
        :param index_labels: Whether to index labels to define output classes or not.
        :return: The augmented `VirtualNode`-s sequence.
        """
        augmented_vnodes = []
        noop_label = (CLASS_INDEX[CLS_NOOP],)
        assert index_labels or noop_label in self.class_sequences_to_labels
        if index_labels and noop_label not in self.class_sequences_to_labels:
            self.class_sequences_to_labels[noop_label] = \
                len(self.class_sequences_to_labels)
            self.labels_to_class_sequences.append(noop_label)
        if not len(vnodes):
            return augmented_vnodes
        if vnodes[0].y is None:
            augmented_vnodes.append(VirtualNode(value="", start=Position(0, 1, 1),
                                                end=Position(0, 1, 1), y=noop_label, path=path))
        for vnode, next_vnode in zip(vnodes, islice(vnodes, 1, None)):
            augmented_vnodes.append(vnode)
            if vnode.y is None and next_vnode.y is None:
                augmented_vnodes.append(VirtualNode(value="", start=vnode.end, end=vnode.end,
                                                    y=noop_label, path=path))
        augmented_vnodes.append(next_vnode)
        if augmented_vnodes[-1].y is None:
            augmented_vnodes.append(VirtualNode(value="", start=vnodes[-1].end, end=vnodes[-1].end,
                                                y=noop_label, path=path))
        return augmented_vnodes

    def _get_features(self, feature_group: FeatureGroup, node_index: int,
                      sibling: Union[VirtualNode, bblfsh.Node], node: VirtualNode
                      ) -> Iterable[int]:
        for feature_name, indices in self._feature_to_indices_set[feature_group][node_index] \
                .items():
            if not len(indices):
                continue
            yield from (value
                        for i, value in enumerate(self._features[feature_name](sibling, node))
                        if i in indices)

    @staticmethod
    def _find_parent(vnode_index: int, vnodes: Sequence[VirtualNode],
                     parents: Mapping[int, bblfsh.Node], closest_left_node_id: int
                     ) -> Optional[bblfsh.Node]:
        """
        Compute vnode parent as the LCA of the closest left and right babelfish nodes.

        :param vnode_index: the index of the current node
        :param vnodes: the sequence of `VirtualNode`-s being transformed into features
        :param parents: the id of bblfsh node to parent bblfsh node mapping
        :param closest_left_node_id: bblfsh node of the closest parent already gone through
        :return: The bblfsh.Node of the found parent or None if no parent was found.
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
    def _inplace_write_features(features: Iterable[int], row: int, col: int, X: numpy.ndarray
                                ) -> int:
        """
        Write features starting at X[row, col] and return the number of features written.

        :param features: the features
        :param row: the row where we should write
        :param col: the column where we should write
        :param X: the feature matrix
        :return: the number of features written
        """
        to_write = [max(FEATURES_MIN, min(FEATURES_MAX, feature)) for feature in features]
        X[row, col:col + len(to_write)] = to_write
        return len(to_write)

    def _keep_sibling(self, sibling: VirtualNode, vnode: VirtualNode, include_labeled: bool
                      ) -> bool:
        if not include_labeled and (
                sibling.y is not None or sibling.node is None and sibling.value.isspace()):
            return False
        if sibling.y == (CLASS_INDEX[CLS_NOOP],):
            return False
        if sibling.y is None:
            return True
        quote_classes = set([CLASS_INDEX[CLS_DOUBLE_QUOTE], CLASS_INDEX[CLS_SINGLE_QUOTE]])
        return not (quote_classes & set(vnode.y) and quote_classes & set(sibling.y))

    def _inplace_write_vnode_features(
            self, vnodes: Sequence[VirtualNode], parents: Mapping[int, bblfsh.Node],
            lines: Set[int], index_offset: int, X: numpy.ndarray, vn: List[VirtualNode]
            ) -> Tuple[int, Optional[List[int]]]:
        """
        Write features in the input matrix given a sequence of `VirtualNode`-s and relevant info.

        :param vnodes: input sequence of `VirtualNode`s
        :param parents: dictionnary of node id to parent node
        :param lines: indices of lines to consider. 1-based.
        :param index_offset: at which index in the input ndarrays we should start writing
        :param X: features matrix, row per sample
        :param vn: list of the corresponding `VirtualNode`s, the length is the same as `X.shape[0]`
        :return: the new offset, list of neighbours if return_neighbours is True else None
        """
        closest_left_node_id = None
        position = index_offset
        if self.return_sibling_indices:
            sibling_indices_list = []
        for i, vnode in enumerate(vnodes):
            if vnode.node:
                closest_left_node_id = id(vnode.node)
            if not vnode.is_labeled_on_lines(lines):
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

            left_sibling_indices = []
            for j in range(i - 1, 0, -1):
                if len(left_sibling_indices) >= self.left_siblings_window:
                    break
                if not self._keep_sibling(vnodes[j], vnode, include_labeled=True):
                    continue
                left_sibling_indices.append(j)
            right_sibling_indices = []
            for j in range(i + 1, len(vnodes)):
                if len(right_sibling_indices) >= self.right_siblings_window:
                    break
                if not self._keep_sibling(vnodes[j], vnode,
                                          include_labeled=not self.no_labels_on_right):
                    continue
                right_sibling_indices.append(j)

            col_offset = 0
            # 1. write features of the current node
            col_offset += self._inplace_write_features(
                self._get_features(FeatureGroup.node, 0, vnode, vnode), position, col_offset, X)

            for j, node_index in enumerate(left_sibling_indices):
                col_offset += self._inplace_write_features(
                    self._get_features(FeatureGroup.left, j, vnodes[node_index], vnode),
                    position, col_offset, X)
            for j in range(self.left_siblings_window - 1, len(left_sibling_indices) - 1, -1):
                col_offset += self.count_features(FeatureGroup.left, j)

            # 3. write features of the right siblings of the current node and account for the
            # possible lack of siblings by adjusting offset
            for j, node_index in enumerate(right_sibling_indices):
                col_offset += self._inplace_write_features(
                    self._get_features(FeatureGroup.right, j, vnodes[node_index], vnode),
                    position, col_offset, X)
            for j in range(self.right_siblings_window - 1, len(right_sibling_indices) - 1, -1):
                col_offset += self.count_features(FeatureGroup.right, j)

            # 4. write features of the parents of the current node
            for j, parent_vnode in enumerate(parents_list):
                col_offset += self._inplace_write_features(
                    self._get_features(FeatureGroup.parents, j, parent_vnode, vnode),
                    position, col_offset, X)

            vn[position] = vnode
            position += 1
            if self.return_sibling_indices:
                sibling_indices_list.append(list(left_sibling_indices)
                                            + list(right_sibling_indices))
        return position, sibling_indices_list if self.return_sibling_indices else None

    def _parse_file(self, contents: str, root: bblfsh.Node, path: str) -> \
            Tuple[List[VirtualNode], Dict[int, bblfsh.Node]]:
        """
        Parse a file into a sequence of `VirtuaNode`-s and a mapping from VirtualNode to parent.

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
