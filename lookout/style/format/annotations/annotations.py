"""Annotations for style-analyzer."""
from typing import FrozenSet, Optional, Tuple

from lookout.style.format.utils import get_uast_parents


class Annotation:
    """Base class for annotation."""

    def __init__(self, start: int, stop: int):
        """
        Initialization.

        :param start: Annotation start.
        :param stop: Annotation end.
        """
        self._range = (start, stop)
        self._start = start
        self._stop = stop

    start = property(lambda self: self._start)

    stop = property(lambda self: self._stop)

    range = property(lambda self: self._range)

    name = property(lambda self: type(self).__name__)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "%s[%d, %d)" % (self.name, self.start, self.stop)


class LineAnnotation(Annotation):
    """Line number annotation."""

    def __init__(self, start: int, stop: int, number: int):
        """Init."""
        super().__init__(start, stop)
        self._number = number

    number = property(lambda self: self._number)


class UASTNodeAnnotation(Annotation):
    """UAST Node annotation."""

    def __init__(self, start: int, stop: int, node: "bblfsh.Node"):
        """Init."""
        super().__init__(start, stop)
        self._node = node

    node = property(lambda self: self._node)

    @staticmethod
    def from_node(node: "bblfsh.Node") -> "UASTNodeAnnotation":
        """Create the annotation from bblfsh node."""
        return UASTNodeAnnotation(node.start_position.offset, node.end_position.offset, node)


class UASTAnnotation(UASTNodeAnnotation):
    """Full UAST of the file annotation."""

    def __init__(self, start: int, stop: int, uast: "bblfsh.Node"):
        """
        Create new UASTAnnotation instance.

        Parents mapping for the tree is built during init.

        :param start: Annotation start.
        :param stop: Annotation end.
        :param uast: UAST.
        """
        super().__init__(start, stop, uast)
        self._parents = get_uast_parents(uast)

    uast = property(lambda self: self._node)

    parents = property(lambda self: self._parents)


class TokenAnnotation(Annotation):
    """Annotation for virtual token in the сode."""

    def __init__(self, start: int, stop: int,
                 uast_annotation: Optional[UASTNodeAnnotation] = None):
        """
        Initialization.

        :param start: Annotation start.
        :param stop: Annotation end.
        :param uast_annotation: Related UASTNodeAnnotation Annotation if applicable.
        """
        super().__init__(start, stop)
        self._uast_annotation = uast_annotation

    uast_annotation = property(lambda self: self._uast_annotation)

    @property
    def node(self) -> "bblfsh.Node":
        """
        Get UAST Node from related UASTNodeAnnotation.

        :return: related bblfsh UAST Node. None if there is no related annotation.
        """
        return self._uast_annotation.node if self._uast_annotation else None

    @property
    def has_node(self) -> bool:
        """Check if token annotation has related UAST node annotation."""
        return self._uast_annotation is not None


class AtomicTokenAnnotation(TokenAnnotation):
    """Annotation for undividable tokens generated by FeatureExtractor._classify_vnodes()."""

    def to_token_annotation(self) -> TokenAnnotation:
        """Convert to token annotation."""
        return TokenAnnotation(self.start, self.stop, self._uast_annotation)


class RawTokenAnnotation(TokenAnnotation):
    """Annotation for raw tokens generated by FeatureExtractor._parse_file()."""

    def to_atomic_token_annotation(self) -> AtomicTokenAnnotation:
        """Convert to atomic token annotation."""
        return AtomicTokenAnnotation(self.start, self.stop, self._uast_annotation)


class LanguageAnnotation(Annotation):
    """Language of the file annotation."""

    def __init__(self, start: int, stop: int, language: str):
        """Init."""
        super().__init__(start, stop)
        self._language = language

    language = property(lambda self: self._language)


class PathAnnotation(Annotation):
    """File language annotation."""

    def __init__(self, start: int, stop: int, path: str):
        """Init."""
        super().__init__(start, stop)
        self._path = path

    path = property(lambda self: self._path)


class TargetAnnotation(Annotation):
    """Target for model prediction annotation."""

    def __init__(self, start: int, stop: int, target: Tuple[int, ...]):
        """Init."""
        super().__init__(start, stop)
        self._target = target

    target = property(lambda self: self._target)


class AtomicTargetAnnotation(TargetAnnotation):
    """Annotation for atomic format symbols like quote, tab, space, etc."""

    def to_target_annotation(self) -> TargetAnnotation:
        """Convert annotation to TargetAnnotation."""
        return TargetAnnotation(self.start, self.stop, self._target)


class AccumulatedIndentationAnnotation(Annotation):
    """Annotates accumulated indentation Token."""


class LinesToCheckAnnotation(Annotation):
    """Annotation to store set of lines where the style should be checked."""

    def __init__(self, start: int, stop: int, lines: FrozenSet[int]):
        """Init."""
        super().__init__(start, stop)
        self._lines = lines

    lines = property(lambda self: self._lines)


class TokenParentAnnotation(Annotation):
    """Annotation to store token UAST parents."""

    def __init__(self, start: int, stop: int, parent: "bblfsh.Node"):
        """Init."""
        super().__init__(start, stop)
        self._parent = parent

    parent = property(lambda self: self._parent)
