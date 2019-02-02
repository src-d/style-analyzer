"""Features definition."""
from collections import Counter
from enum import Enum, unique
import importlib
from typing import (Any, Generic, Iterable, List, Mapping, MutableMapping, Optional, Sequence,
                    Tuple, TypeVar)

import bblfsh
from lookout.core.ports import Type
import numpy
from scipy.sparse import csr_matrix

from lookout.style.format.classes import CLASSES
from lookout.style.format.virtual_node import AnyNode, VirtualNode

FEATURES_NUMPY_TYPE = numpy.uint8
FEATURES_MIN = numpy.iinfo(FEATURES_NUMPY_TYPE).min
FEATURES_MAX = numpy.iinfo(FEATURES_NUMPY_TYPE).max


@unique
class FeatureGroup(Enum):
    """
    Feature groups.

    Each feature belongs to one and only one of these classes.
    """

    node = 1
    left = 2
    right = 3
    parents = 4

    def format(self, value) -> str:
        """
        Represent the feature group for user interfaces. The trailing dot is appended as needed.

        :param value: The feature group parameter. E.g., the index of the node for "left" \
                      and "right".
        :return: pretty-printed string, the trailing dot is appended as needed.
        """
        if self == FeatureGroup.node:
            return "•••"
        if self == FeatureGroup.left:
            return str(-value - 1) + "."
        if self == FeatureGroup.right:
            return "+%s." % (value + 1)
        if self == FeatureGroup.parents:
            return "^%d." % (value + 1)  # ↑ is displayed like shit in Ubuntu
        return "%s.%s." % (self.name, value)

    def __lt__(self, other: "FeatureGroup") -> bool:
        """Compare two groups: am I less than the other."""
        return self.value < other.value


FEATURE_GROUP_TYPES = {
    FeatureGroup.node: VirtualNode,
    FeatureGroup.left: VirtualNode,
    FeatureGroup.right: VirtualNode,
    FeatureGroup.parents: bblfsh.Node,
}


@unique
class FeatureId(Enum):
    """Feature identifiers."""

    diff_col = 0
    diff_line = 1
    diff_offset = 2
    index_internal_type = 3
    index_label = 4
    index_reserved = 5
    internal_type = 6
    label = 7
    reserved = 8
    roles = 9
    length = 10
    start_col = 11
    start_line = 12

    def __lt__(self, other: "FeatureId") -> bool:
        """
        Compare two identifiers: am I less than the other.

        This does not make much apart from ordering enums.
        """
        return self.value < other.value


TV = TypeVar("TV")
Layout = Mapping[FeatureGroup, Sequence[TV]]
MutableLayout = MutableMapping[FeatureGroup, List[TV]]
FeatureLayout = Layout[Mapping[FeatureId, TV]]
MutableFeatureLayout = MutableLayout[MutableMapping[FeatureId, TV]]
TVFeature = TypeVar("TVFeature", bound="Feature")
TVAnyNode = TypeVar("TVAnyNode", bblfsh.Node, VirtualNode)


class Feature:
    """Base type for features."""

    id = None  # type: FeatureId

    def __init__(self, *, language: str, labels_to_class_sequences: Sequence[Tuple[int, ...]],
                 selected_indices: Optional[Sequence[int]] = None, **kwargs: Any) -> None:
        """
        Construct the `Feature`.

        Consumes the arguments it requires and passes the rest to super.

        :param selected_indices: None if all the indices are used in the feature, otherwise a \
                                 of the used indices.
        :param language: Define which language-specific tweaks to use.
        :param labels_to_class_sequences: Index of labels to class sequences.
        :param kwargs: Keywords arguments to pass on to super.
        :return: Self.
        """
        super().__init__(**kwargs)
        self._selected_indices = selected_indices
        self.labels_to_class_sequences = labels_to_class_sequences
        self.class_sequences_to_labels = {tuple(s): i
                                          for i, s in enumerate(labels_to_class_sequences)}
        self.language = language
        self.roles = importlib.import_module(
            "lookout.style.format.langs.%s.roles" % self.language)
        self.tokens = importlib.import_module(
            "lookout.style.format.langs.%s.tokens" % self.language)
        self._selected_indices_index = None  # type: Optional[Mapping[int, int]]
        self._selected_names = None  # type: Optional[List[str]]
        self._selected_names_index = None  # type: Optional[Mapping[str, int]]

    def __call__(self, neighbours: Layout[Sequence[Optional[AnyNode]]]) -> csr_matrix:
        """
        Compute the relevant values for this feature.

        This method only creates the resulting array and delegates the filling to subclasses.

        :param neighbours: Neighbouring nodes to the current sample.
        :return: Numpy array containing the feature values. 2-dimensional.
        """
        X_shape = len(neighbours[FeatureGroup.node][0]), len(self.selected_names)
        values, row_indices, column_indices = self._compute(neighbours)
        if not X_shape[1]:
            return csr_matrix(X_shape, dtype=FEATURES_NUMPY_TYPE)
        return csr_matrix((values, (row_indices, column_indices)), dtype=FEATURES_NUMPY_TYPE,
                          shape=X_shape)

    @staticmethod
    def _clip_int(integer: int) -> int:
        return max(FEATURES_MIN, min(FEATURES_MAX, integer))

    def _compute(self, neighbours: Layout[Sequence[Optional[AnyNode]]],
                 ) -> Tuple[List[int], List[int],  List[int]]:
        raise NotImplementedError()

    @property
    def selected_indices(self) -> Sequence[int]:
        """Return the sequence of selected indices."""
        return (list(range(len(self.names)))
                if self._selected_indices is None
                else self._selected_indices)

    @property
    def selected_indices_index(self) -> Mapping[int, int]:
        """Return the set of selected indices."""
        if self._selected_indices_index is None:
            self._selected_indices_index = {index: i
                                            for i, index in enumerate(self.selected_indices)}
        return self._selected_indices_index

    @property
    def selected_names(self) -> Sequence[str]:
        """Return the names of the features this Feature computes (after feature selection)."""
        if self._selected_names is None:
            self._selected_names = [name for i, name in enumerate(self.names)
                                    if i in self.selected_indices_index]
        return self._selected_names

    @property
    def names(self) -> Sequence[str]:
        """Return the names of the features this Feature computes (before feature selection)."""
        if not hasattr(self, "_names"):
            raise NotImplementedError()
        return self._names

    @property
    def selected_names_index(self) -> Mapping[str, int]:
        """Return the index of a name to its position (after feature selection)."""
        if self._selected_names_index is None:
            self._selected_names_index = {name: i for i, name in enumerate(self.selected_names)}
        return self._selected_names_index


class MultipleValuesFeature(Feature):
    """Base type for features that produce multiple values."""

    pass


class CategoricalFeature(MultipleValuesFeature):
    """Base type for features that have a one hot encoded value."""

    pass


class BagFeature(MultipleValuesFeature):
    """Base type for features that have multiple values."""

    pass


class OrdinalFeature(Feature):
    """Base type for ordinal features."""

    def __init__(self, **kwargs: Any) -> None:
        """Construct an OrdinalFeature."""
        super().__init__(**kwargs)
        self._names = [self.id.name]


class NeighbourFeature(Feature, Generic[TVAnyNode]):
    """Base type for features that focus on a node among the neighbours of the current node."""

    def __init__(self, neighbour_group: FeatureGroup, neighbour_index: int, **kwargs: Any) -> None:
        """Construct an NeighbourFeature.

        :param neighbour_group: Group of the neighbour to compare with the current node.
        :param neighbour_index: Index of the neighbour to compare with the current node.
        :param kwargs: Keywords arguments to pass on to super.
        """
        super().__init__(**kwargs)
        self.neighbour_group = neighbour_group
        self.neighbour_index = neighbour_index
        self._check_whitelist()

    def _convert_nodes(self, nodes: Sequence[Optional[AnyNode]]) -> Sequence[Optional[TVAnyNode]]:
        if not hasattr(self, "target_type"):
            raise NotImplementedError()
        current_type = FEATURE_GROUP_TYPES[self.neighbour_group]
        if issubclass(current_type, self.target_type):
            return nodes
        if issubclass(self.target_type, VirtualNode) and issubclass(current_type, bblfsh.Node):
            raise RuntimeError("Cannot transform babelfish nodes into virtual nodes.")
        return [None if node is None else node.node for node in nodes]

    def _check_whitelist(self) -> None:
        whitelist = getattr(self, "feature_groups_whitelist", None)
        if whitelist is None:
            return
        if self.neighbour_group not in whitelist:
            raise RuntimeError(
                "Trying to apply the %s feature to the non-whitelisted feature group %s." %
                (self.id.name, self.neighbour_group.name))


class ComparisonFeature(NeighbourFeature[TVAnyNode]):
    """Base type for features that compare the current node to one of its neighbours."""

    def _compute(self, neighbours: Layout[Sequence[Optional[AnyNode]]],
                 ) -> Tuple[List[int], List[int], List[int]]:
        """Focus on the relevant nodes and compute the feature values."""
        return self._focused_compute(  # type: ignore
            neighbours[FeatureGroup.node][0],
            self._convert_nodes(neighbours[self.neighbour_group][self.neighbour_index]))

    def _focused_compute(self, nodes: Sequence[VirtualNode],
                         neighbours: Sequence[Optional[TVAnyNode]],
                         ) -> Tuple[List[int], List[int], List[int]]:
        """Compute the feature values."""
        all_values, all_row_indices, all_column_indices = [], [], []
        for row_index, (node, neighbour) in enumerate(zip(nodes, neighbours)):
            if neighbour is not None:
                try:
                    values, column_indices = zip(*self._compute_row(node, neighbour))
                    all_values.extend(values)
                    all_row_indices.extend([row_index] * len(values))
                    all_column_indices.extend(column_indices)
                except ValueError:
                    # Nothing returned by the feature
                    pass
        return all_values, all_row_indices, all_column_indices

    def _compute_row(self, node: VirtualNode, neighbour: TVAnyNode) -> Iterable[Tuple[int, int]]:
        """Compute the feature value(s) for a given node and its neighbour."""
        raise NotImplementedError()


class VirtualNodeComparisonFeature(ComparisonFeature[VirtualNode]):
    """Base type for Feature-s that compare the current virtual node to another virtual node."""

    target_type = VirtualNode


class NodeFeature(NeighbourFeature[TVAnyNode]):
    """Base type for features that compute a property of a single node."""

    def _compute(self, neighbours: Layout[Sequence[Optional[AnyNode]]],
                 ) -> Tuple[List[int], List[int], List[int]]:
        """Focus on the relevant nodes and compute the feature values."""
        return self._focused_compute(
            self._convert_nodes(neighbours[self.neighbour_group][self.neighbour_index]))

    def _focused_compute(self, nodes: Sequence[Optional[TVAnyNode]],
                         ) -> Tuple[List[int], List[int], List[int]]:
        """Compute the feature values."""
        all_values, all_row_indices, all_column_indices = [], [], []
        for row_index, node in enumerate(nodes):
            if node is not None:
                try:
                    values, column_indices = zip(*self._compute_row(node))
                    all_values.extend(values)
                    all_row_indices.extend([row_index] * len(values))
                    all_column_indices.extend(column_indices)
                except ValueError:
                    # Nothing returned by the feature
                    pass
        return all_values, all_row_indices, all_column_indices

    def _compute_row(self, node: TVAnyNode) -> Iterable[Tuple[int, int]]:
        """Compute the feature value(s) for a given node."""
        raise NotImplementedError()


class VirtualNodeFeature(NodeFeature[VirtualNode]):
    """Base type for Feature-s computed on a virtual node per sample."""

    target_type = VirtualNode


class BblfshNodeFeature(NodeFeature[bblfsh.Node]):
    """Base type for Feature-s computed on a bblfsh node per sample."""

    target_type = bblfsh.Node


FEATURE_CLASSES = {}  # type: MutableMapping[FeatureId, Type[Feature]]


def register_feature(cls: Type[TVFeature]) -> Type[TVFeature]:
    """Register features in the __features__ module attribute."""
    if not issubclass(cls, Feature):
        raise TypeError("%s is not an instance of %s" % (cls.__name__, Feature.__name__))
    FEATURE_CLASSES[cls.id] = cls
    return cls


@register_feature
class _FeatureDiffCol(OrdinalFeature, VirtualNodeComparisonFeature):

    id = FeatureId.diff_col
    feature_groups_whitelist = (FeatureGroup.left,)

    def _compute_row(self, node: VirtualNode, neighbour: VirtualNode) -> Iterable[Tuple[int, int]]:
        yield self._clip_int(abs(neighbour.start.col - node.start.col)), 0


@register_feature
class _FeatureDiffLine(OrdinalFeature, VirtualNodeComparisonFeature):

    id = FeatureId.diff_line
    feature_groups_whitelist = (FeatureGroup.left,)

    def _compute_row(self, node: VirtualNode, neighbour: VirtualNode) -> Iterable[Tuple[int, int]]:
        yield (self._clip_int(min(abs(neighbour.start.line - node.end.line),
                                  abs(neighbour.end.line - node.start.line))),
               0)


@register_feature
class _FeatureDiffOffset(OrdinalFeature, VirtualNodeComparisonFeature):

    id = FeatureId.diff_offset
    feature_groups_whitelist = (FeatureGroup.left,)

    def _compute_row(self, node: VirtualNode, neighbour: VirtualNode) -> Iterable[Tuple[int, int]]:
        yield self._clip_int(abs(neighbour.start.offset - node.start.offset)), 0


@register_feature
class _FeatureIndexInternalType(OrdinalFeature, BblfshNodeFeature):

    id = FeatureId.index_internal_type

    def _compute_row(self, node: bblfsh.Node) -> Iterable[Tuple[int, int]]:
        if node.internal_type in self.roles.INTERNAL_TYPES_INDEX:
            yield self._clip_int(self.roles.INTERNAL_TYPES_INDEX[node.internal_type] + 1), 0


@register_feature
class _FeatureIndexLabel(OrdinalFeature, VirtualNodeFeature):

    id = FeatureId.index_label
    feature_groups_whitelist = (FeatureGroup.left,)

    def _compute_row(self, node: VirtualNode) -> Iterable[Tuple[int, int]]:
        if node.y is not None:
            yield self._clip_int(self.class_sequences_to_labels[node.y] + 1), 0


@register_feature
class _FeatureIndexReserved(OrdinalFeature, VirtualNodeFeature):

    id = FeatureId.index_reserved

    def _compute_row(self, node: VirtualNode) -> Iterable[Tuple[int, int]]:
        if not (node.y is None or node.node) and node.value in self.tokens.RESERVED_INDEX:
            yield self._clip_int(self.tokens.RESERVED_INDEX[node.value] + 1), 0


@register_feature
class _FeatureInternalType(CategoricalFeature, BblfshNodeFeature):

    id = FeatureId.internal_type

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._names = self.roles.INTERNAL_TYPES

    def _compute_row(self, node: bblfsh.Node) -> Iterable[Tuple[int, int]]:
        internal_type = node.internal_type
        if internal_type in self.selected_names_index:
            yield 1, self.selected_names_index[internal_type]


@register_feature
class _FeatureLabel(BagFeature, VirtualNodeFeature):

    id = FeatureId.label
    feature_groups_whitelist = (FeatureGroup.left,)

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._names = CLASSES

    def _compute_row(self, node: VirtualNode) -> Iterable[Tuple[int, int]]:
        if node.y is not None:
            res = Counter()
            for cls_index in node.y:
                if cls_index in self.selected_indices_index:
                    res[self.selected_indices_index[cls_index]] += 1
            for column_index, value in res.items():
                yield value, column_index


@register_feature
class _FeatureReserved(CategoricalFeature, VirtualNodeFeature):

    id = FeatureId.reserved

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._names = self.tokens.RESERVED

    def _compute_row(self, node: VirtualNode) -> Iterable[Tuple[int, int]]:
        if node.value in self.selected_names_index:
            yield 1, self.selected_names_index[node.value]


@register_feature
class _FeatureRoles(BagFeature, BblfshNodeFeature):

    id = FeatureId.roles

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._names = self.roles.ROLES

    def _compute_row(self, node: bblfsh.Node) -> Iterable[Tuple[int, int]]:
        for role_id in node.roles:
            role = bblfsh.role_name(role_id)
            if role in self.selected_names_index:
                yield 1, self.selected_names_index[role]


@register_feature
class _FeatureLength(OrdinalFeature, VirtualNodeFeature):

    id = FeatureId.length
    feature_groups_whitelist = (FeatureGroup.left, FeatureGroup.right)

    def _compute_row(self, node: VirtualNode) -> Iterable[Tuple[int, int]]:
        yield self._clip_int(node.end.offset - node.start.offset), 0


@register_feature
class _FeatureStartCol(OrdinalFeature, VirtualNodeFeature):

    id = FeatureId.start_col
    feature_groups_whitelist = (FeatureGroup.left, FeatureGroup.node)

    def _compute_row(self, node: VirtualNode) -> Iterable[Tuple[int, int]]:
        yield self._clip_int(node.start.col), 0


@register_feature
class _FeatureStartLine(OrdinalFeature, VirtualNodeFeature):

    id = FeatureId.start_line
    feature_groups_whitelist = (FeatureGroup.left, FeatureGroup.node)

    def _compute_row(self, node: VirtualNode) -> Iterable[Tuple[int, int]]:
        yield self._clip_int(node.start.line), 0
