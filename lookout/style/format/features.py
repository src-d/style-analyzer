"""Features definition."""
from enum import Enum, unique
import importlib
from typing import Callable, Iterable, Mapping, Union

import bblfsh

from lookout.core.ports import Type
from lookout.style.format.feature_utils import CLASSES, VirtualNode


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


@unique
class FeatureGroup(Enum):
    """Feature groups."""

    all = 0
    node = 1
    parents = 2
    left = 3
    right = 4


FeatureFunction = Callable[[Union[VirtualNode, bblfsh.Node], VirtualNode], Iterable[int]]


class Feature:
    """Base type for features."""

    id = None  # type: FeatureId

    def __call__(self, sibling: Union[VirtualNode, bblfsh.Node], node: VirtualNode
                 ) -> Iterable[int]:
        """Compute the relevant values for this feature."""
        raise NotImplementedError


class OHEFeature(Feature):
    """Base type for one hot encoded features."""

    @property
    def names(self) -> Iterable[str]:
        """Return the names of the features this Feature computes."""
        if not hasattr(self, "_names"):
            raise NotImplementedError()
        yield from ("%s_%s" % (self.id.name, name.lower()) for name in self._names)


class SingleOHEFeature(OHEFeature):
    """Base type for features that have a single one hot encoded value."""
    pass


class MultiOHEFeature(OHEFeature):
    """Base type for features that have multiple one hot encoded values ."""


class SimpleFeature(Feature):
    """Base type for features that return one value."""

    @property
    def names(self) -> Iterable[str]:
        """Return the name of the feature this Feature computes."""
        yield self.id.name


class RolesMixin:
    def __init__(self, language: str) -> None:
        self._roles = importlib.import_module("lookout.style.format.langs.%s.roles" % language)


class TokensMixin:
    def __init__(self, language: str) -> None:
        self._tokens = importlib.import_module("lookout.style.format.langs.%s.tokens" % language)


_feature_classes = {}


def get_features(language: str) -> Mapping[str, Feature]:
    def instantiate(cls):
        if issubclass(cls, TokensMixin) or issubclass(cls, RolesMixin):
            return cls(language)
        else:
            return cls()

    return {feature_id: instantiate(feature_class)
            for feature_id, feature_class in _feature_classes.items()}


def register_feature(cls: Type[Feature]) -> Type[Feature]:
    """Register features in the __features__ module attribute."""
    if not issubclass(cls, Feature):
        raise TypeError("%s is not an instance of %s" % (cls.__name__, Feature.__name__))
    _feature_classes[cls.id.name] = cls
    return cls


@register_feature
class _FeatureDiffCol(SimpleFeature):

    id = FeatureId.diff_col

    def __call__(self, sibling: Union[VirtualNode, bblfsh.Node], node: VirtualNode
                 ) -> Iterable[int]:
        yield min(abs(sibling.start.col - node.end.col),
                  abs(sibling.end.col - node.start.col))


@register_feature
class _FeatureDiffLine(SimpleFeature):

    id = FeatureId.diff_line

    def __call__(self, sibling: Union[VirtualNode, bblfsh.Node], node: VirtualNode
                 ) -> Iterable[int]:
        yield min(abs(sibling.start.line - node.end.line),
                  abs(sibling.end.line - node.start.line))


@register_feature
class _FeatureDiffOffset(SimpleFeature):

    id = FeatureId.diff_offset

    def __call__(self, sibling: Union[VirtualNode, bblfsh.Node], node: VirtualNode
                 ) -> Iterable[int]:
        yield min(abs(sibling.start.offset - node.end.offset),
                  abs(sibling.end.offset - node.start.offset))


@register_feature
class _FeatureIndexInternalType(SimpleFeature, RolesMixin):

    id = FeatureId.index_internal_type

    def __init__(self, language: str) -> None:
        RolesMixin.__init__(self, language)
        self._internal_types_index = self._roles.INTERNAL_TYPES_INDEX

    def __call__(self, sibling: Union[VirtualNode, bblfsh.Node], node: VirtualNode
                 ) -> Iterable[int]:
        internal_type_index = 0
        bblfsh_node = sibling.node if hasattr(sibling, "node") else sibling
        if bblfsh_node and bblfsh_node.internal_type in self._internal_types_index:
            internal_type_index = (self._internal_types_index[bblfsh_node.internal_type] + 1)
        yield internal_type_index


@register_feature
class _FeatureIndexLabel(SimpleFeature):

    id = FeatureId.index_label

    def __call__(self, sibling: Union[VirtualNode, bblfsh.Node], node: VirtualNode
                 ) -> Iterable[int]:
        yield sibling.y + 1 if sibling.y is not None else 0


@register_feature
class _FeatureIndexReserved(SimpleFeature, TokensMixin):

    id = FeatureId.index_reserved

    def __init__(self, language: str) -> None:
        TokensMixin.__init__(self, language)
        self._reserved_index = self._tokens.RESERVED_INDEX

    def __call__(self, sibling: Union[VirtualNode, bblfsh.Node], node: VirtualNode
                 ) -> Iterable[int]:
        reserved_index = 0
        bblfsh_node = sibling.node if hasattr(sibling, "node") else sibling
        if not bblfsh_node and sibling.value in self._reserved_index:
            reserved_index = self._reserved_index[sibling.value] + 1
        yield reserved_index


@register_feature
class _FeatureInternalType(SingleOHEFeature, RolesMixin):

    id = FeatureId.internal_type

    def __init__(self, language: str) -> None:
        RolesMixin.__init__(self, language)
        self._names = self._roles.INTERNAL_TYPES
        self._internal_types_index = self._roles.INTERNAL_TYPES_INDEX

    def __call__(self, sibling: Union[VirtualNode, bblfsh.Node], node: VirtualNode
                 ) -> Iterable[int]:
        internal_type_indices = [0] * len(self._names)
        bblfsh_node = sibling.node if hasattr(sibling, "node") else sibling
        if bblfsh_node:
            internal_type = bblfsh_node.internal_type
            if internal_type in self._internal_types_index:
                internal_type_indices[self._internal_types_index[internal_type]] = 1
        yield from internal_type_indices


@register_feature
class _FeatureLabel(SingleOHEFeature):

    id = FeatureId.label

    def __init__(self) -> None:
        self._names = CLASSES

    def __call__(self, sibling: Union[VirtualNode, bblfsh.Node], node: VirtualNode
                 ) -> Iterable[int]:
        label_indices = [0] * len(self._names)
        if sibling.y is not None:
            label_indices[sibling.y] = 1
        yield from label_indices


@register_feature
class _FeatureReserved(SingleOHEFeature, TokensMixin):

    id = FeatureId.reserved

    def __init__(self, language: str) -> None:
        TokensMixin.__init__(self, language)
        self._names = self._tokens.RESERVED
        self._reserved_index = self._tokens.RESERVED_INDEX

    def __call__(self, sibling: Union[VirtualNode, bblfsh.Node], node: VirtualNode
                 ) -> Iterable[int]:
        reserved_indices = [0] * len(self._names)
        if sibling.node is None and sibling.value in self._reserved_index:
            reserved_indices[self._reserved_index[sibling.value]] = 1
        yield from reserved_indices


@register_feature
class _FeatureRoles(MultiOHEFeature, RolesMixin):

    id = FeatureId.roles

    def __init__(self, language: str) -> None:
        RolesMixin.__init__(self, language)
        self._names = self._roles.ROLES
        self._roles_index = self._roles.ROLES_INDEX

    def __call__(self, sibling: Union[VirtualNode, bblfsh.Node], node: VirtualNode
                 ) -> Iterable[int]:
        role_indices = [0] * len(self._names)
        bblfsh_node = sibling.node if hasattr(sibling, "node") else sibling
        if bblfsh_node:
            for role_id in bblfsh_node.roles:
                role = bblfsh.role_name(role_id)
                role_indices[self._roles_index[role]] += 1
        yield from role_indices


@register_feature
class _FeatureLength(SimpleFeature):

    id = FeatureId.length

    def __call__(self, sibling: Union[VirtualNode, bblfsh.Node], node: VirtualNode
                 ) -> Iterable[int]:
        yield sibling.end.offset - sibling.start.offset


@register_feature
class _FeatureStartCol(SimpleFeature):

    id = FeatureId.start_col

    def __call__(self, sibling: Union[VirtualNode, bblfsh.Node], node: VirtualNode
                 ) -> Iterable[int]:
        yield sibling.start.col


@register_feature
class _FeatureStartLine(SimpleFeature):

    id = FeatureId.start_line

    def __call__(self, sibling: Union[VirtualNode, bblfsh.Node], node: VirtualNode
                 ) -> Iterable[int]:
        yield sibling.start.line
