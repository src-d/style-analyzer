"""Annotations for style-analyzer."""
from typing import FrozenSet, Optional, Tuple, Union

import numpy

from lookout.style.format.utils import get_uast_parents


def check_offset(value: Union[int, numpy.int], name: str) -> None:
    """
    Validate the offset: check the type and reject negative values.

    :param value: Offset value to check.
    :param name: Variable name for the exception message.
    :raises ValueError: if value is incorrect.
    """
    if not (isinstance(value, int) or isinstance(value, numpy.int)):
        raise ValueError("Type of `%s` should be int. Got %s" % (name, type(value)))
    if value < 0:
        raise ValueError("`%s` should be non-negative. Got %d" % (name, value))


def check_span(start: int, stop: int) -> None:
    """
    Validate span value: check the type, reject negative values and ensure an increasing order.

    :param start: Start offset of the span.
    :param stop: Stop offset of the span.
    :raises ValueError: if span value is incorrect.
    """
    check_offset(start, "start")
    check_offset(stop, "stop")
    if stop < start:
        raise ValueError("`stop` should be not less than `start`. Got start=%d, stop=%d" % (
            start, stop))


class Annotation:
    """Base class for all annotations."""

    def __init__(self, start: int, stop: int):
        """
        Initialize a new instance of `Annotation`.

        :param start: Start of the annotated span.
        :param stop: End of the annotated span. Stop point itself is excluded.
        """
        check_span(start, stop)
        self._start = start
        self._stop = stop

    start = property(lambda self: self._start)
    stop = property(lambda self: self._stop)
    span = property(lambda self: (self._start, self._stop))
    name = property(lambda self: type(self).__name__)

    def __repr__(self) -> str:
        """Format `Annotation` object as a string."""
        return self.__str__()

    def __str__(self) -> str:
        """Format `Annotation` description as a string."""
        return "%s[%d, %d)" % (self.name, self.start, self.stop)


class LineAnnotation(Annotation):
    """
    Line number annotation.

    Line numbers are 1-based.
    """

    def __init__(self, start: int, stop: int, line: int):
        """Initialize a new instance of `LineAnnotation`."""
        super().__init__(start, stop)
        if not isinstance(line, int):
            raise ValueError("Type of `line` should be int. Got %s" % type(line))
        if line < 1:
            raise ValueError("`line` value should be positive. Got %d" % line)
        self._line = line

    line = property(lambda self: self._line)


class UASTNodeAnnotation(Annotation):
    """UAST Node annotation."""

    def __init__(self, start: int, stop: int, node: "bblfsh.Node"):
        """Initialize a new instance of `UASTNodeAnnotation`."""
        super().__init__(start, stop)
        self._node = node

    node = property(lambda self: self._node)

    @staticmethod
    def from_node(node: "bblfsh.Node") -> "UASTNodeAnnotation":
        """Create the annotation from `bblfsh.Node`."""
        return UASTNodeAnnotation(node.start_position.offset, node.end_position.offset, node)


class UASTAnnotation(UASTNodeAnnotation):
    """Whole file UAST annotation."""

    def __init__(self, start: int, stop: int, uast: "bblfsh.Node"):
        """
        Initialize a new instance of `UASTAnnotation`.

        Parents mapping for the tree is built during init.

        :param start: Annotation start.
        :param stop: Annotation end.
        :param uast: UAST of the file.
        """
        super().__init__(start, stop, uast)
        self._parents = get_uast_parents(uast)

    uast = property(lambda self: self._node)
    parents = property(lambda self: self._parents)


class TokenAnnotation(Annotation):
    """Annotation for сode virtual token."""

    def __init__(self, start: int, stop: int,
                 uast_annotation: Optional[UASTNodeAnnotation] = None):
        """
        Initialize a new instance of `TokenAnnotation`.

        `TokenAnnotation` may have 0 length: [x, x). To avoid the complex search of the related
        UAST annotation it is required to pass `UASTNodeAnnotation` to `__init__()`.

        :param start: Annotation start.
        :param stop: Annotation end.
        :param uast_annotation: Related `UASTNodeAnnotation` if applicable.
        """
        super().__init__(start, stop)
        self._uast_annotation = uast_annotation

    uast_annotation = property(lambda self: self._uast_annotation)

    @property
    def node(self) -> "bblfsh.Node":
        """
        Get the UAST Node belonging to the underlying `UASTNodeAnnotation`.

        :return: Related bblfsh UAST Node. None if there is no related `UASTNodeAnnotation`.
        """
        return self._uast_annotation.node if self._uast_annotation else None

    @property
    def has_node(self) -> bool:
        """Check if token annotation has related UAST node annotation."""
        return self._uast_annotation is not None


class AtomicTokenAnnotation(TokenAnnotation):
    """Annotation for indivisible tokens generated by `FeatureExtractor._classify_vnodes()`."""

    def to_token_annotation(self) -> TokenAnnotation:
        """Convert to `TokenAnnotation`."""
        return TokenAnnotation(self.start, self.stop, self._uast_annotation)


class RawTokenAnnotation(TokenAnnotation):
    """Annotation for raw tokens generated by `FeatureExtractor._parse_file()`."""

    def to_atomic_token_annotation(self) -> AtomicTokenAnnotation:
        """Convert to `AtomicTokenAnnotation`."""
        return AtomicTokenAnnotation(self.start, self.stop, self._uast_annotation)


class LanguageAnnotation(Annotation):
    """Language of the file annotation."""

    def __init__(self, start: int, stop: int, language: str):
        """Initialize a new instance of `LanguageAnnotation`."""
        super().__init__(start, stop)
        self._language = language

    language = property(lambda self: self._language)


class PathAnnotation(Annotation):
    """File path annotation."""

    def __init__(self, start: int, stop: int, path: str):
        """Initialize a new instance of `PathAnnotation`."""
        super().__init__(start, stop)
        self._path = path

    path = property(lambda self: self._path)


class LabelAnnotation(Annotation):
    """Label annotation that should be predicted by the ML model."""

    def __init__(self, start: int, stop: int, label: Tuple[int, ...]):
        """Initialize a new instance of `LabelAnnotation`."""
        super().__init__(start, stop)
        self._label = label

    label = property(lambda self: self._label)


class ClassAnnotation(Annotation):
    """Class Annotation for atomic format symbols like quote, tab, space, etc."""

    def __init__(self, start: int, stop: int, cls: Tuple[int, ...]):
        """Initialize a new instance of `LabelAnnotation`."""
        super().__init__(start, stop)
        self._cls = cls

    cls = property(lambda self: self._cls)

    def to_target_annotation(self) -> LabelAnnotation:
        """Convert annotation to `TargetAnnotation`."""
        return LabelAnnotation(self.start, self.stop, self._cls)


class AccumulatedIndentationAnnotation(Annotation):
    """Annotation for "accumulated indentation" token."""


class LinesToCheckAnnotation(Annotation):
    """
    Annotation to store a set of lines where the style shall be checked.

    Lines are 1-based.
    """

    def __init__(self, start: int, stop: int, lines: FrozenSet[int]):
        """Initialize a new instance of `LinesToCheckAnnotation`."""
        super().__init__(start, stop)
        for line in lines:
            if not isinstance(line, int):
                raise ValueError("Type of `line` should be int. Got %s" % type(line))
            if line < 1:
                raise ValueError("`line` value should be positive. Got %d" % line)
        self._lines = lines

    lines = property(lambda self: self._lines)


class TokenParentAnnotation(Annotation):
    """Annotation to store the UAST parent node of a token."""

    def __init__(self, start: int, stop: int, parent: "bblfsh.Node"):
        """Initialize a new instance of `TokenParentAnnotation`."""
        super().__init__(start, stop)
        self._parent = parent

    parent = property(lambda self: self._parent)
