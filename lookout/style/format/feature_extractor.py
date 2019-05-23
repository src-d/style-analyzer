"""Feature extraction module."""
from collections import defaultdict, OrderedDict
import importlib
from itertools import chain, zip_longest
import logging
from operator import itemgetter
from typing import (Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union)

import bblfsh
from lookout.core.analyzer import UnicodeFile
import numpy
from scipy.sparse import csr_matrix, hstack, vstack
from sklearn.exceptions import NotFittedError
from sklearn.feature_selection import SelectKBest, VarianceThreshold

from lookout.style.format.annotations.annotated_data import AnnotationManager
from lookout.style.format.annotations.annotations import AccumulatedIndentationAnnotation, \
    AtomicTokenAnnotation, ClassAnnotation, LabelAnnotation, LinesToCheckAnnotation, \
    PathAnnotation, RawTokenAnnotation, TokenAnnotation, TokenParentAnnotation, UASTAnnotation
from lookout.style.format.classes import (
    CLASS_INDEX, CLASS_PRINTABLES, CLASS_REPRESENTATIONS, CLS_DOUBLE_QUOTE, CLS_NOOP,
    CLS_SINGLE_QUOTE, CLS_SPACE, CLS_SPACE_DEC, CLS_SPACE_INC, CLS_TAB, CLS_TAB_DEC, CLS_TAB_INC,
    INDEX_CLS_TO_STR, NEWLINE_INDEX, QUOTES_INDEX)
from lookout.style.format.features import (  # noqa: F401
    Feature, FEATURE_CLASSES, FeatureGroup, FeatureId, FeatureLayout, Layout,
    MultipleValuesFeature, MutableFeatureLayout, MutableLayout)
from lookout.style.format.virtual_node import AnyNode, Position, VirtualNode


IndexToFeature = List[Tuple[FeatureGroup, int, FeatureId, int]]

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
                 debug_parsing: bool, return_sibling_indices: bool,
                 selected_features: Optional[numpy.ndarray] = None,
                 label_composites: Optional[List[Tuple[int, ...]]] = None,
                 cutoff_label_support: int = 0) -> None:
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
        :param debug_parsing: Whether to pause on parsing exceptions instead of skipping.
        :param return_sibling_indices: Whether to return the indices of siblings of the predicted \
                                       nodes.
        :param selected_features: Feature indexes to use. If None ("train" stage), we learn them \
                                  in select_features(). Otherwise, we are in "analyze" stage.
        :param label_composites: Maps composite output classes to the corresponding sequences of \
                                 "atomic" classes. If None or empty ("train" stage), we build \
                                 this mapping inside `extract_features()`. Otherwise, we are in \
                                 "analyze" stage.
        :param cutoff_label_support: Minimum number of samples for each class to be included.
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
        self.debug_parsing = debug_parsing
        self.return_sibling_indices = return_sibling_indices
        self.selected_features = selected_features
        self.cutoff_label_support = cutoff_label_support
        self.labels_to_class_sequences = list(map(tuple, label_composites)) \
            if label_composites is not None else []
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
            self._compute_feature_info()

    @property
    def index_to_feature(self) -> IndexToFeature:
        """Return the mapping from integer indices to the corresponding feature names."""
        if not hasattr(self, "_index_to_feature"):
            raise NotFittedError()
        return self._index_to_feature

    @property
    def feature_to_indices(self) -> FeatureLayout[Sequence[int]]:
        """Return the mapping from feature names to the corresponding integer indices."""
        if not hasattr(self, "_feature_to_indices"):
            raise NotFittedError()
        return self._feature_to_indices

    @property
    def features(self) -> FeatureLayout[Feature]:
        """Return the `Feature`-s used by this feature extractor."""
        if not hasattr(self, "_features"):
            raise NotFittedError()
        return self._features

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
        if not hasattr(self, "_feature_names"):
            raise NotFittedError()
        return self._feature_names

    @property
    def composite_class_representations(self) -> List[str]:
        """
        Return the class representations of composite classes.

        :return: Strings representing the composite classes.
        """
        return ["".join(CLASS_REPRESENTATIONS[label] for label in labels)
                for labels in self.labels_to_class_sequences]

    @property
    def composite_class_printables(self) -> List[str]:
        """
        Return the class printables of composite classes.

        :return: Strings that can be printed to represent the composite classes.
        """
        return ["".join(CLASS_PRINTABLES[label] for label in labels)
                for labels in self.labels_to_class_sequences]

    def count_features(self, feature_group: Optional[FeatureGroup] = None,
                       neighbour_index: Optional[int] = None) -> int:
        """Return the feature count of a given subset of features."""
        if not hasattr(self, "_feature_count"):
            raise NotFittedError()
        if feature_group is None:
            return self._feature_count
        if neighbour_index is None:
            return self._feature_group_counts[feature_group]
        return self._feature_node_counts[feature_group][neighbour_index]

    def extract_features(self, files: Iterable[UnicodeFile],
                         lines: Optional[List[List[int]]] = None) \
            -> Optional[Union[Tuple[csr_matrix, numpy.ndarray,
                                    Tuple[List[VirtualNode], List[VirtualNode],
                                          Dict[int, bblfsh.Node], Dict[int, bblfsh.Node]]],
                              Tuple[csr_matrix, numpy.ndarray,
                                    Tuple[List[VirtualNode], List[VirtualNode],
                                          Dict[int, bblfsh.Node], Dict[int, bblfsh.Node],
                                          List[List[int]]]]]]:
        """
        Compute features and labels required by downstream models given a list of `File`-s.

        :param files: the list of `File`-s (see service_data.proto) of the same language.
        :param lines: the list of enabled line numbers per file. The lines which are not \
                      mentioned will not be extracted.
        :return: tuple of numpy.ndarray (2 and 1 dimensional respectively): features and labels, \
                 the corresponding `VirtualNode`-s and the parents mapping \
                 or None in case no features were extracted.
        """
        files = self._annotate_files(files, lines)
        parsed_files, node_parents, vnode_parents = files_to_old_parse_vnodes_format(files)
        xy = self._convert_files_to_xy(parsed_files)
        if xy is None:
            return None
        X, y, vnodes_y, vnodes, sibling_indices_list = xy
        self._log.debug("Features shape: %s", X.shape)
        if self.return_sibling_indices:
            return X, y, (vnodes_y, vnodes, vnode_parents, node_parents, sibling_indices_list)
        return X, y, (vnodes_y, vnodes, vnode_parents, node_parents)

    def select_features(self, X: csr_matrix, y: numpy.ndarray) -> Tuple[csr_matrix, numpy.ndarray]:
        """
        Select the most useful features based on sklearn's univariate feature selection.

        :param X: Scipy CSR 2-dimensional matrix of features to select.
        :param y: Numpy 1-dimensional array of labels.
        :return: Tuple of a CSR matrix with only the selected features (columns) kept and an \
                 array of the indices of the kept features for later reapplication.
        """
        if self.selected_features is not None:
            return X, self.selected_features
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

    def label_to_str(self, label: int) -> str:
        """Convert a label to string."""
        return "".join(INDEX_CLS_TO_STR[cls]
                       for cls in self.labels_to_class_sequences[label])

    def _compute_feature_info(self) -> None:
        if self.selected_features is not None:
            selected_features_set = set(self.selected_features)

        def populate_indices(feature_group: FeatureGroup, feature_names: Sequence[str],
                             nodes_number: int, total_index: int) -> int:
            self._feature_to_indices[feature_group] = []
            self._features[feature_group] = []
            for node_index in range(nodes_number):
                self._feature_to_indices[feature_group].append(OrderedDict())
                self._features[feature_group].append(OrderedDict())
                for feature_name in feature_names:
                    feature_id = FeatureId[feature_name]
                    self._feature_to_indices[feature_group][node_index][feature_id] = []
                    feature_class = FEATURE_CLASSES[FeatureId[feature_name]]
                    feature_pre_selection = feature_class(
                        language=self.language,
                        labels_to_class_sequences=self.labels_to_class_sequences,
                        selected_indices=None,
                        neighbour_group=feature_group,
                        neighbour_index=node_index)
                    selected_indices = []
                    names = feature_pre_selection.names
                    for index, name in enumerate(names):
                        if self.selected_features is None or total_index in selected_features_set:
                            selected_indices.append(index)
                            if issubclass(feature_class, MultipleValuesFeature):
                                last_name_part = "%s.%s" % (feature_pre_selection.id.name, name)
                            else:
                                last_name_part = feature_pre_selection.id.name
                            self._feature_names.append("%s.%d.%s" % (feature_group.name,
                                                                     node_index,
                                                                     last_name_part))
                            self._feature_to_indices[feature_group][node_index][feature_id] \
                                .append(total_index)
                            self._index_to_feature.append((feature_group, node_index, feature_id,
                                                           index))
                        total_index += 1
                    self._features[feature_group][node_index][feature_id] = feature_class(
                        language=self.language,
                        labels_to_class_sequences=self.labels_to_class_sequences,
                        selected_indices=selected_indices,
                        neighbour_group=feature_group,
                        neighbour_index=node_index)
            return total_index
        self._index_to_feature = []  # type: IndexToFeature
        self._feature_to_indices = OrderedDict()  # type: MutableFeatureLayout[List[int]]
        self._feature_names = []  # type: List[str]
        self._features = OrderedDict()  # type: MutableFeatureLayout[Feature]
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
                                             for node_index in self._feature_to_indices[group]]
                                     for group in FeatureGroup}
        self._feature_group_counts = {group: sum(counts)
                                      for group, counts in self._feature_node_counts.items()}
        self._feature_count = sum(self._feature_group_counts.values())

    def _annotate_files(self, files: Iterable[UnicodeFile],
                        lines: Optional[List[List[int]]] = None) -> List[AnnotationManager]:
        parsed_files = []
        for i, file in enumerate(files):
            path = file.path
            file = AnnotationManager.from_file(file)
            if lines is not None and lines[i] is not None:
                file.add(LinesToCheckAnnotation(0, len(file), frozenset(lines[i])))
            try:
                self._parse_file(file)
            except AssertionError as e:
                self._log.warning("could not parse %s: error '%s', skipping", path, e)
                if self.debug_parsing:
                    import traceback
                    traceback.print_exc()
                    input("Press Enter to continue…")
                continue
            self._classify_vnodes(file)
            self._merge_classes_to_composite_labels(file)
            self._add_noops(file)
            parsed_files.append(file)
            self._fill_vnode_parents(file)
        vnodes_parsed_number = sum(file.count(TokenAnnotation) for file in parsed_files)
        self._log.debug("Parsed %d vnodes", vnodes_parsed_number)
        return parsed_files

    def _convert_files_to_xy(
            self, parsed_files: List[Tuple[List[VirtualNode], Dict[int, bblfsh.Node], Set[int]]],
            ) -> Optional[Tuple[csr_matrix, numpy.ndarray, List[VirtualNode], List[VirtualNode],
                          List[List[int]]]]:
        vnodes_parsed_number = sum(len(vn) for vn, _, _ in parsed_files)
        index_labels = not self.labels_to_class_sequences
        # filter composite labels by support
        if index_labels:
            self._compute_labels_mappings(chain.from_iterable(vn for vn, _, _ in parsed_files))
        files_vnodes_y = [[vnode for vnode in file_vnodes
                           if vnode.is_labeled_on_lines(file_lines) and
                           vnode.y in self.class_sequences_to_labels]
                          for file_vnodes, _, file_lines in parsed_files]
        labels = [[self.class_sequences_to_labels[vnode.y]
                   for vnode in file_vnodes_y]
                  for file_vnodes_y in files_vnodes_y]
        if not labels:
            # nothing was extracted
            return None
        y = numpy.concatenate(labels)
        self._log.debug("%d out of %d are labeled and saved after filtering", y.shape[0],
                        vnodes_parsed_number)
        if index_labels:
            self._compute_feature_info()
        features = [feature for group_features in self._features.values()
                    for node_features in group_features for feature in node_features.values()]
        xs = []
        vnodes = []
        vnodes_y = []
        sibling_indices_list = []
        assert len(parsed_files) == len(files_vnodes_y)
        for (file_vnodes, file_parents, _), file_vnodes_y in zip(parsed_files, files_vnodes_y):
            vnodes.extend(file_vnodes)
            vnodes_y.extend(file_vnodes_y)
            neighbours, sibling_indices = self._create_neighbours(
                file_vnodes, file_vnodes_y, file_parents, self.return_sibling_indices)
            xs.append(hstack([feature(neighbours) for feature in features]))
            if self.return_sibling_indices:
                sibling_indices_list.extend(sibling_indices)
        assert len(y) == len(vnodes_y)
        X = vstack(xs)
        return X, y, vnodes_y, vnodes, sibling_indices_list

    def _create_neighbours(self, vnodes: Sequence[VirtualNode], vnodes_y: Sequence[VirtualNode],
                           parents: Mapping[int, bblfsh.Node],
                           return_sibling_indices: bool = False,
                           ) -> Tuple[Layout[Sequence[Optional[AnyNode]]],
                                      Optional[List[List[int]]]]:
        if self.return_sibling_indices:
            sibling_indices_list = []
        neighbours = OrderedDict()  # type: MutableLayout[List[Optional[AnyNode]]]
        for group_id, size in zip([FeatureGroup.node, FeatureGroup.left, FeatureGroup.right,
                                   FeatureGroup.parents],
                                  [1, self.left_siblings_window, self.right_siblings_window,
                                   self.parents_depth]):
            neighbours[group_id] = [[] for _ in range(size)]

        vnodes_y_set = frozenset(id(vnode_y) for vnode_y in vnodes_y)
        closest_left_node_id = None

        for i, vnode in enumerate(vnodes):
            if vnode.node:
                closest_left_node_id = id(vnode.node)
            if id(vnode) not in vnodes_y_set:
                continue

            # Current node
            neighbours[FeatureGroup.node][0].append(vnode)

            # Current node's parents
            parent = (self._find_parent_old(i, vnodes, parents, closest_left_node_id)
                      if self.parents_depth else None)
            parents_list = []
            if parent:
                current_ancestor = parent
                for _ in range(self.parents_depth):
                    parents_list.append(current_ancestor)
                    current_ancestor_id = id(current_ancestor)
                    if current_ancestor_id not in parents:
                        break
                    current_ancestor = parents[current_ancestor_id]
            for j, node in zip_longest(range(self.parents_depth), parents_list):
                neighbours[FeatureGroup.parents][j].append(node)

            # Current node's left siblings
            left_sibling_indices = []
            for j in range(i - 1, 0, -1):
                if len(left_sibling_indices) >= self.left_siblings_window:
                    break
                if not self._keep_sibling(vnodes[j], vnode, include_labeled=True):
                    continue
                left_sibling_indices.append(j)
            for j, node_index in zip_longest(range(self.left_siblings_window),
                                             left_sibling_indices):
                neighbours[FeatureGroup.left][j].append(None if node_index is None
                                                        else vnodes[node_index])

            # Current node's right siblings
            right_sibling_indices = []
            for j in range(i + 1, len(vnodes)):
                if len(right_sibling_indices) >= self.right_siblings_window:
                    break
                if not self._keep_sibling(vnodes[j], vnode,
                                          include_labeled=not self.no_labels_on_right):
                    continue
                right_sibling_indices.append(j)
            for j, node_index in zip_longest(range(self.right_siblings_window),
                                             right_sibling_indices):
                neighbours[FeatureGroup.right][j].append(None if node_index is None
                                                         else vnodes[node_index])

            if return_sibling_indices:
                sibling_indices_list.append(left_sibling_indices + right_sibling_indices)
        return neighbours, sibling_indices_list if return_sibling_indices else None

    def _classify_vnodes(self, file: AnnotationManager) -> None:
        """
        Annotate source code with `AtomicTokenAnnotation`, `ClassAnnotation` and \
        `AccumulatedIndentationAnnotation`.

        `ClassAnnotation` contains the index of the corresponding class to predict.
        We detect indentation changes, so several whitespace nodes are merged together.

        :param file: Source code annotated with `RawTokenAnnotation`.
        """
        indentation = []
        for token in file.iter_by_type(RawTokenAnnotation):
            token_value = file[token.span]
            if token.has_node:
                file.add(token.to_atomic_token_annotation())
                continue
            if not token_value.isspace():
                if token_value == "'":
                    file.add(ClassAnnotation(*token.span,
                                             (CLASS_INDEX[CLS_SINGLE_QUOTE],)))
                elif token_value == '"':
                    file.add(ClassAnnotation(*token.span,
                                             (CLASS_INDEX[CLS_DOUBLE_QUOTE],)))
                file.add(token.to_atomic_token_annotation())
                continue
            lines = token_value.splitlines(keepends=True)
            if lines[-1].splitlines()[0] != lines[-1]:
                # We add last line as empty one to mimic .split("\n") behaviour
                lines.append("")
            if len(lines) == 1:
                # only tabs and spaces are possible
                for i, char in enumerate(token_value):
                    if char == "\t":
                        cls = (CLASS_INDEX[CLS_TAB],)
                    else:
                        cls = (CLASS_INDEX[CLS_SPACE],)
                    offset = token.start
                    file.add(ClassAnnotation(offset + i, offset + i + 1, cls))
                    file.add(AtomicTokenAnnotation(offset + i, offset + i + 1))
                continue
            line_offset = 0
            traling_chars = lines[0].splitlines()[0]
            if traling_chars:
                # node contains trailing whitespaces from the previous line
                assert set(traling_chars) <= {" ", "\t"}
                file.add(ClassAnnotation(
                    token.start, token.start + len(traling_chars), tuple(
                        CLASS_INDEX[CLS_SPACE if yi == " " else CLS_TAB] for yi in traling_chars)))
                file.add(AtomicTokenAnnotation(token.start, token.start + len(traling_chars)))

                lines[0] = lines[0][len(traling_chars):]
                line_offset += len(traling_chars)

            for line in lines[:-1]:
                # `line` ends with \r\n, we prepend \r to the newline node
                start_offset = token.start + line_offset
                file.add(ClassAnnotation(start_offset, start_offset + len(line),
                                         (NEWLINE_INDEX,)))
                file.add(AtomicTokenAnnotation(start_offset, start_offset + len(line)))
                line_offset += len(line)
            line = lines[-1].splitlines()[0] if lines[-1] else ""
            my_indent = list(line)
            offset = token.stop
            offset -= len(line)
            try:
                for ws in indentation:
                    my_indent.remove(ws)
            except ValueError:
                if my_indent:
                    # mixed tabs and spaces, do not classify
                    file.add(AtomicTokenAnnotation(offset, token.stop))
                    continue
                # indentation decreases
                if indentation[:len(line)]:
                    file.add(AtomicTokenAnnotation(offset, token.stop))
                    file.add(AccumulatedIndentationAnnotation(offset, token.stop))
                dec_class = []
                for char in indentation[len(line):]:
                    if char == "\t":
                        cls = CLASS_INDEX[CLS_TAB_DEC]
                    else:
                        cls = CLASS_INDEX[CLS_SPACE_DEC]
                    dec_class.append(cls)

                file.add(AtomicTokenAnnotation(token.stop, token.stop))
                # It is not possible to have multiple zero-length intervals so we can only add it
                # with joined class
                file.add(ClassAnnotation(token.stop, token.stop, tuple(dec_class)))
                indentation = indentation[:len(line)]
            else:
                # indentation is stable or increases
                if indentation:
                    file.add(AtomicTokenAnnotation(offset, offset + len(indentation)))
                    file.add(AccumulatedIndentationAnnotation(offset, offset + len(indentation)))
                offset += len(indentation)
                for char in my_indent:
                    indentation.append(char)
                for i, char in enumerate(my_indent):
                    if char == "\t":
                        cls = (CLASS_INDEX[CLS_TAB_INC],)
                    else:
                        cls = (CLASS_INDEX[CLS_SPACE_INC],)
                    file.add(AtomicTokenAnnotation(offset + i, offset + i + 1))
                    file.add(ClassAnnotation(offset + i, offset + i + 1, cls))
                offset += len(my_indent)

    def _merge_classes_to_composite_labels(self, file: AnnotationManager) -> None:
        """
        Build "composite" `TokenAnnotation` and `LabelAnnotation` from predictable atomic tokens.

        :param file: Source code annotated with `AtomicTokenAnnotation`, `ClassAnnotation`, \
                     `AccumulatedIndentationAnnotation`.
        """
        def _class_seq_to_annotations(start, stop, current_class_seq):
            if NEWLINE_INDEX not in current_class_seq or \
                    current_class_seq[0] == NEWLINE_INDEX:
                # if there are no trailing whitespaces or tabs
                yield TokenAnnotation(start, stop)
                yield LabelAnnotation(start, stop, tuple(current_class_seq))
            else:
                index = current_class_seq.index(NEWLINE_INDEX)
                middle = start + index
                yield TokenAnnotation(start, middle)
                yield TokenAnnotation(middle, stop)
                yield LabelAnnotation(start, middle, tuple(current_class_seq[:index]))
                yield LabelAnnotation(middle, stop, tuple(current_class_seq[index:]))

        start, stop, current_class_seq = None, None, []

        for annotations in file.iter_by_type_nested(
                AtomicTokenAnnotation, ClassAnnotation, AccumulatedIndentationAnnotation):
            has_target = ClassAnnotation in annotations
            acc_indent = AccumulatedIndentationAnnotation in annotations
            if (not has_target and not acc_indent or (
                    has_target and annotations[ClassAnnotation].cls[0] in QUOTES_INDEX)):
                if current_class_seq:
                    file.add(*_class_seq_to_annotations(start, stop, current_class_seq))
                    start, stop, current_class_seq = None, None, []
                file.add(annotations[AtomicTokenAnnotation].to_token_annotation())
                if ClassAnnotation in annotations:
                    file.add(annotations[ClassAnnotation].to_target_annotation())
            else:
                if not current_class_seq:
                    start = annotations.start
                stop = annotations.stop
                if not acc_indent:
                    current_class_seq.extend(annotations[ClassAnnotation].cls)
        if current_class_seq:
            file.add(*_class_seq_to_annotations(start, stop, current_class_seq))

    def _add_noops(self, file: AnnotationManager) -> None:
        """
        Add `TokenAnnotation` with zero length in between `TokenAnnotation` without labeled nodes.

        Such zero length annotations means that some formatting sequence can be inserted to the
        annotation position.

        :param file: Source code annotated with `TokenAnnotation` and `LabelAnnotation`.
        """
        noop_target = (CLASS_INDEX[CLS_NOOP],)
        if not len(file):
            return

        prev_annotations = None
        for i, annotations in enumerate(file.iter_by_type_nested(
                TokenAnnotation, LabelAnnotation)):
            if i == 0:
                if LabelAnnotation not in annotations:
                    file.add(TokenAnnotation(0, 0))
                    file.add(LabelAnnotation(0, 0, noop_target))
            else:
                if LabelAnnotation not in prev_annotations and \
                        LabelAnnotation not in annotations:
                    file.add(TokenAnnotation(annotations.start, annotations.start))
                    file.add(LabelAnnotation(annotations.start, annotations.start, noop_target))
            prev_annotations = annotations

        if LabelAnnotation not in annotations:
            file.add(TokenAnnotation(annotations.stop, annotations.stop))
            file.add(LabelAnnotation(annotations.stop, annotations.stop, noop_target))

    @staticmethod
    def _find_parent(search_start_offset: int, file: AnnotationManager, closest_left_node_id: int,
                     ) -> Optional[bblfsh.Node]:
        """
        Compute the UAST parent of the `TokenAnnotation` as the LCA of the closest left and right \
        Babelfish nodes.

        :param search_start_offset: Offset of the current node.
        :param file: Source code annotated with `UASTAnnotation` and `TokenAnnotation`.
        :param closest_left_node_id: bblfsh node of the closest parent already gone through.
        :return: The `bblfsh.Node` of the found parent or None if no parent was found.
        """
        left_ancestors = set()
        current_left_ancestor_id = closest_left_node_id
        parents = file.get(UASTAnnotation).parents
        while current_left_ancestor_id in parents:
            left_ancestors.add(id(parents[current_left_ancestor_id]))
            current_left_ancestor_id = id(parents[current_left_ancestor_id])

        for future_vnode in file.iter_by_type(TokenAnnotation,
                                              start_offset=search_start_offset):
            if future_vnode.has_node:
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
    def _find_parent_old(vnode_index: int, vnodes: Sequence[VirtualNode],
                         parents: Mapping[int, bblfsh.Node], closest_left_node_id: int,
                         ) -> Optional[bblfsh.Node]:
        """
        Compute vnode parent as the LCA of the closest left and right babelfish nodes.

        :param vnode_index: the index of the current node
        :param vnodes: the sequence of `VirtualNode`-s being transformed into features
        :param parents: the id of bblfsh node to parent bblfsh node mapping
        :param closest_left_node_id: bblfsh node of the closest parent already gone through
        :return: The bblfsh.Node of the found parent or None if no parent was found.
        """
        # TODO(zurk): merge with new `_find_parent()`.
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

    def _keep_sibling(self, sibling: VirtualNode, vnode: VirtualNode, include_labeled: bool,
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

    def _parse_file(self, file: AnnotationManager) -> None:
        """
        Annotate source code with `RawTokenAnnotation`-s.

        Given the source text and the corresponding UAST this function covers all code with a
        `RawTokenAnnotation`-s.

        :param file: Source code annotated with `UASTAnnotation`.
        """
        # TODO(zurk): rename this function when the refactoring is finished.
        contents = file.sequence
        # build the line mapping
        lines = contents.splitlines(keepends=True)
        # Check if there is a newline in the end of file. Yes, you can just check
        # lines[-1][-1] == "\n" but if someone decide to use weird '\u2028' unicode character for
        # new line this condition gives wrong result.
        eof_new_line = lines[-1].splitlines()[0] != lines[-1]
        if eof_new_line:
            # We add last line as empty one because it actually exists, but .splitlines() does not
            # return it.
            lines.append("")
        line_offsets = numpy.zeros(len(lines) + 1, dtype=numpy.int32)
        pos = 0
        for i, line in enumerate(lines):
            line_offsets[i] = pos
            pos += len(line)
        line_offsets[-1] = pos + 1

        # walk the tree: collect nodes with assigned tokens
        node_tokens = []
        queue = [file.get(UASTAnnotation).uast]
        while queue:
            node = queue.pop()
            if node.internal_type in self.node_fixtures:
                self.node_fixtures[node.internal_type](node)
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
        pos = 0
        parser = self.tokens.PARSER
        for node in node_tokens:
            if node.start_position.offset < pos:
                continue
            if node.start_position.offset > pos:
                sumlen = 0
                diff = contents[pos:node.start_position.offset]
                for match in parser.finditer(diff):
                    offsets = []
                    for suboff in (match.start(), match.end()):
                        offsets.append(pos + suboff)
                    token = match.group()
                    sumlen += len(token)
                    file.add(RawTokenAnnotation(*offsets))
                assert sumlen == node.start_position.offset - pos, \
                    "missed some imaginary tokens: \"%s\"" % diff
            if node is sentinel:
                break
            uast_node_annot = list(VirtualNode.from_node(node, contents, self.token_unwrappers))
            file.add(*uast_node_annot)
            pos = node.end_position.offset

    def _compute_labels_mappings(self, vnodes: Iterable[VirtualNode]) -> None:
        """
        Calculate the label to class sequence and class sequence to label mappings.

        Takes into account self.cutoff_label_support and discard those with too little value.

        :param vnodes: The virtual nodes extracted from all the files.
        """
        assert len(self.class_sequences_to_labels) == 0, self.class_sequences_to_labels
        assert len(self.labels_to_class_sequences) == 0, self.labels_to_class_sequences
        support = defaultdict(int)
        for vnode in vnodes:
            if vnode.y is not None:
                support[vnode.y] += 1

        # Sort by support to create labels from most frequent to the least frequent
        self.labels_to_class_sequences = [
            key for key, val in sorted(support.items(), key=itemgetter(1), reverse=True)
            if val >= self.cutoff_label_support]
        self.class_sequences_to_labels = {
            class_seq: i for i, class_seq in enumerate(self.labels_to_class_sequences)}
        self._log.debug("Removed %d/%d labels by support %d",
                        len(support) - len(self.labels_to_class_sequences), len(support),
                        self.cutoff_label_support)

    def _fill_vnode_parents(self, file: AnnotationManager):
        closest_left_node_id = None
        uast_annotation = file.get(UASTAnnotation)
        for annotation in file.iter_by_type(TokenAnnotation):
            if annotation.has_node:
                closest_left_node_id = id(annotation.node)
                file.add(TokenParentAnnotation(*annotation.span,
                                               uast_annotation.parents[closest_left_node_id]))
            else:
                parent = (self._find_parent(annotation.stop, file, closest_left_node_id)
                          or uast_annotation.uast)
                file.add(TokenParentAnnotation(*annotation.span, parent))


def _to_position(raw_lines_data, _lines_start_offset, offset):
    line_num = numpy.argmax(_lines_start_offset > offset) - 1
    col = offset - _lines_start_offset[line_num]
    line = raw_lines_data[line_num]
    if len(line) == col:
        if line.splitlines()[0] != line:
            # ends with newline
            line_num += 1
            col = 0
    return Position(offset, line_num + 1, col + 1)


def file_to_old_parse_file_format(file: AnnotationManager) -> Tuple[List["VirtualNode"],
                                                                    Dict[int, bblfsh.Node]]:
    """
    Convert `AnnotationManager` instance to the deprecated output format of \
    `FeatureExtractor._parse_file()`.

    The function exists for backward compatibility and should be removed after the refactoring is \
    finished.

    :param file: file annotated with `UASTAnnotation`, `PathAnnotation` and `RawTokenAnnotation`. \
                 It is expected to be the output of  `FeatureExtractor._parse_file()`.
    :return: The old `FeatureExtractor._parse_file()` output format, that is \
             Tuple with `VirtualNode`-s and `bbfsh.Node` id to parent mapping.
    """
    vnodes = []
    path = file.get(PathAnnotation).path
    raw_lines_data = file.sequence.splitlines(keepends=True)
    line_lens = [0] + [len(d) for d in raw_lines_data]
    line_lens[-1] += 1
    _line_start_offsets = numpy.array(line_lens).cumsum()
    for annotation in file.iter_by_type(RawTokenAnnotation):
        vnode = VirtualNode(
            file[annotation.span],
            _to_position(raw_lines_data, _line_start_offsets, annotation.start),
            _to_position(raw_lines_data, _line_start_offsets, annotation.stop),
            is_accumulated_indentation=False,
            path=path,
            node=annotation.node,
            y=None,
        )
        vnodes.append(vnode)
    return vnodes, file.get(UASTAnnotation).parents


def _file_to_vnodes_and_parents(file: AnnotationManager) -> Tuple[List["VirtualNode"],
                                                                  Dict[int, bblfsh.Node]]:
    """
    Convert one `AnnotationManager` instance to the deprecated format of \
    `FeatureExtractor._annotate_files()` (`_parse_vnodes()` before refactoring).

    The old format is a sequence of vnodes and vnodes parents mapping. Used by
    `files_to_old_parse_file_format` to generate the old `_parse_vnodes`-like output format for a
    sequence of `AnnotationManager`-s. This function is different from
    `file_to_old_parse_file_format()` because it is created for `_parse_vnodes()` backward
    compatibility and `file_to_old_parse_file_format()` for `_parse_file()` backward compatibility.

    The function exists for backward compatibility and should be removed after the refactoring is \
    finished.

    :param file: file annotated with `Path`-, `Token`-, `Label`-, `TokenParent`- `Annotation`.
    :return: Tuple with `VirtualNode`-s and node id to parents mapping.
    """
    vnodes = []
    path = file.get(PathAnnotation).path
    raw_lines_data = file.sequence.splitlines(keepends=True)
    line_lens = [0] + [len(d) for d in raw_lines_data]
    line_lens[-1] += 1
    _line_start_offsets = numpy.array(line_lens).cumsum()
    vnode_parents = {}
    for annotations in file.iter_by_type_nested(TokenAnnotation,
                                                LabelAnnotation, TokenParentAnnotation):
        vnode = VirtualNode(
            file[annotations.span],
            _to_position(raw_lines_data, _line_start_offsets, annotations.start),
            _to_position(raw_lines_data, _line_start_offsets, annotations.stop),
            is_accumulated_indentation=False,
            path=path,
            node=annotations[TokenAnnotation].node,
            y=annotations[LabelAnnotation].label if LabelAnnotation in annotations else None,
        )
        vnodes.append(vnode)
        vnode_parents[id(vnode)] = annotations[TokenParentAnnotation].parent \
            if TokenParentAnnotation in annotations else None
    return vnodes, vnode_parents


def files_to_old_parse_vnodes_format(
        files: Sequence[AnnotationManager],
    ) -> Tuple[List[Tuple[List[VirtualNode], Dict[int, bblfsh.Node], Set[int]]],
               Dict[int, bblfsh.Node],
               Dict[int, bblfsh.Node]]:
    """
    Convert a sequence of `AnnotationManager` instances to the deprecated output format of \
    `FeatureExtractor._annotate_files()` (`_parse_vnodes()` before refactoring).

    In addition to `_file_to_vnodes_and_parents()` it provides the `node_parents` mapping.

    The function exists for backward compatibility and should be removed after the refactoring is \
    finished.

    :param files: Sequence of fully annotated files. It is expected to be the output of \
                  `FeatureExtractor._parse_vnodes()`.
    :return: The old `FeatureExtractor._parse_vnodes()` output format, that is \
             Tuple with `VirtualNode`-s, node parents mapping and vnode parents mapping.
    """
    vnode_parents = {}
    node_parents = {}
    vnodes = []
    for file in files:
        file_vnodes, file_vnode_parents = _file_to_vnodes_and_parents(file)
        file_node_parents = file.get(UASTAnnotation).parents
        try:
            file_lines = set(file.get(LinesToCheckAnnotation).lines)
        except KeyError:
            file_lines = None
        vnodes.append((file_vnodes, file_node_parents, file_lines))
        vnode_parents.update(file_vnode_parents)
        node_parents.update(file_node_parents)

    return vnodes, node_parents, vnode_parents
