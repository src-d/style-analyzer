from typing import Dict, Iterator, Optional, Tuple, Union, Type, Callable  # noqa F401

from lookout.core.analyzer import UnicodeFile
from sortedcontainers import SortedDict

from lookout.style.format.annotations.annotations import Annotation, check_offset, check_span, \
    LanguageAnnotation, PathAnnotation, UASTAnnotation


class NoAnnotation(Exception):
    """
    Raised by `AnnotationManager` methods if there is no Annotation found.

    See documentation about `AnnotationManager.find_overlapping_annotation()` or
    `AnnotationManager.find_covering_annotation()` for more information.
    """


class AnnotationsSpan(dict):
    """
    Annotations collection for a specific span (or range).

    Dictionary-like object.
    """

    def __init__(self, start, stop, *args, **kwargs):
        """
        Initialize a new instance of `AnnotationsSlice`.

        :param start: Start of the span.
        :param stop: End of the span. Stop point itself is excluded.
        :param args: The rest position arguments are passed to `dict.__init__()`.
        :param kwargs: The rest key arguments are passed to `dict.__init__()`.
        """
        check_span(start, stop)
        super().__init__(*args, **kwargs)
        self._span = (start, stop)
        self._start = start
        self._stop = stop

    start = property(lambda self: self._start)
    stop = property(lambda self: self._stop)
    span = property(lambda self: self._span)


class AnnotationManager:
    """
    Manager of `Annotation`-s for a text, e.g. source code.

    All the methods to work with annotated data should be placed in this class.
    Candidates can be found here:
    https://uima.apache.org/d/uimafit-current/api/org/apache/uima/fit/util/JCasUtil.html
    """

    def __init__(self, sequence: str):
        """
        Initialize a new `AnnotationManager` instance.

        :param sequence: Sequential data to annotate. It is expected to be string but can be any \
                         type with __getitem__() defined for int and slice input arguments.
        """
        self._sequence = sequence

        # Dictionary to store annotations for the whole file (aka `global` annotations)
        self._global_annotations = {}  # type: Dict[Type[Annotation], Annotation]

        # The following dictionaries are the main members. The most common use case we have in
        # style-analyzer is iterating through Token annotations in the sorted order. This is why
        # ordered dict is used.
        # `self._type_to_annotations` is the main storage of the Annotations. It is a mapping
        # from the annotation type to all annotations of this type which are stored in the \
        # dictionary that is sorted by spans.
        # `self._span_to_annotations` dict is an optimization to quickly lookup all
        # `Annotation`-s that belong to the same [start, stop) span.
        self._span_to_annotations = SortedDict()  # type: SortedDict[(int, int), Dict[Type[Annotation], Annotation]]  # noqa E501
        self._type_to_annotations = {}  # type: Dict[Type[Annotation], SortedDict[(int, int), Annotation]]  # noqa E501

    sequence = property(lambda self: self._sequence)

    def __len__(self):
        """Return the size of the underlying sequence."""
        return len(self._sequence)

    def __getitem__(self, item: Union[int, slice, Tuple[int, int]]) -> str:
        """
        Get the underlying sequence item or sequence slice for the specified range.

        :param item: index, slice or (start, stop) tuple.
        :return: The corresponding part of the sequence.
        """
        if isinstance(item, tuple):
            item = slice(*item)
        if isinstance(item, slice) and item.step is not None:
            raise KeyError("slice.step is not supported.")
        return self._sequence[item]

    def count(self, annotation_type: Type[Annotation]):
        """Count the number of annotations of a specific type."""
        return len(self._type_to_annotations.get(annotation_type, []))

    def add(self, *annotations: Annotation) -> None:
        """
        Add several annotations.
        """
        for annotation in annotations:
            self._add(annotation)

    def _add(self, annotation: Annotation) -> None:
        """
        Add an annotation. Annotations of the same type may not overlap.
        """
        annotation_type = type(annotation)
        if annotation.start == 0 and annotation.stop == len(self):
            if annotation_type in self._global_annotations:
                raise ValueError("Global annotation %s already exists" % annotation)
            self._global_annotations[annotation_type] = annotation
        else:
            if annotation.span not in self._span_to_annotations:
                self._span_to_annotations[annotation.span] = {}
            if annotation_type not in self._type_to_annotations:
                self._type_to_annotations[annotation_type] = SortedDict()
            if self._type_to_annotations[annotation_type]:
                right_span_index = self._type_to_annotations[annotation_type].bisect_left(
                    annotation.span)
                if right_span_index < len(self._type_to_annotations[annotation_type]):
                    right_span, right_annotation = \
                        self._type_to_annotations[annotation_type].peekitem(right_span_index)
                    if self._check_spans_overlap(*right_span, *annotation.span):
                        raise ValueError(
                            "Trying to add Annotation %s which overlaps with existing %s" % (
                                right_annotation, annotation))
                left_span_index = max(0, right_span_index - 1)
                left_span, left_annotation = self._type_to_annotations[annotation_type].peekitem(
                    left_span_index)
                if self._check_spans_overlap(*left_span, *annotation.span):
                    raise ValueError("Trying to add Annotation %s which overlaps with existing"
                                     "%s" % (left_annotation, annotation))
            self._span_to_annotations[annotation.span][annotation_type] = annotation
            self._type_to_annotations[annotation_type][annotation.span] = annotation

    def get(self, annotation_type: Type[Annotation], span: Optional[Tuple[int, int]] = None,
            ) -> Annotation:
        """
        Return an annotation for the given span and type.

        Looking for an exact (type and span) match only.

        :param annotation_type: Annotation type to get.
        :param span: Annotation span (range) to get. If span is not specified it returns an \
                     annotation that cover all content (aka global annotation).
        :return: Requested `Annotation`.
        """
        if span is None:
            return self._global_annotations[annotation_type]
        check_span(*span)
        return self._type_to_annotations[annotation_type][span]

    def iter_by_type_nested(self, annotation_type: Type[Annotation],
                            *covered_by: Type[Annotation],
                            start_offset: Optional[int] = None) -> Iterator[AnnotationsSpan]:
        """
        Iterate over annotations of the specified type which are covered by some other annotation \
        types.

        Iteration goes over `annotation_type` objects. Annotations which are specified in
        `covered_by` are added to the resulting `AnnotationsSpan` object. Spans of the additional
        annotations should fully cover the spans of `annotation_type`. For example, suppose that
        you have *line* and *token* annotations. Each line contains several tokens. If you try to
        iterate through *line* and set *token* as `covered_by` annotation you get only *line*
        annotation inside `AnnotationsSpan`. It happens because you can annotate *token* with
        *line* but not *line* with *token*: *token* is covered by only one line and not vice versa.

        So,
        manager.iter_by_type_nested(line, token)  # gives you only line annotation as output,
                                                  # *token* annotations not found
        manager.iter_by_type_nested(token, line)  # gives you *token* and *line* annotations
                                                  # because it is possible to find only one
                                                  # covering *line* annotation.

        `covered_by` can't be empty. If you need to iterate over a single annotation type, you
        should call `AnnotationManager.iter_annotation()` instead.

        :param annotation_type: Type of annotation to iterate through.
        :param covered_by: Additional annotations that should be added to the main one if they \
                           cover its span.
        :param start_offset: Start to iterate from a specific offset. \
                             Can be used as a key argument only.
        :return: Iterator over annotations of the requested type.
        """
        if not covered_by:
            raise ValueError("At least one covered_by annotation must be specified. "
                             "If you need to iterate over a single annotation type use "
                             "`iter_annotation()`.")
        types = set(covered_by) | {annotation_type}
        for annotation in self.iter_by_type(annotation_type, start_offset=start_offset):
            # Annotations with the same span
            same_span_annotations = self._span_to_annotations[annotation.span]
            same_span_annotations_type = set(same_span_annotations.keys())
            common_types = types & same_span_annotations_type
            missing_types = types - same_span_annotations_type
            annotations = dict()
            for missing_type in missing_types:
                try:
                    annotations[missing_type] = self.find_covering_annotation(missing_type,
                                                                              *annotation.span)
                except NoAnnotation:
                    pass
            annotations.update({type: same_span_annotations[type] for type in common_types})
            yield AnnotationsSpan(*annotation.span, annotations)

    def iter_by_type(self, annotation_type: Type[Annotation], *,
                     start_offset: Optional[int] = None) -> Iterator[Annotation]:
        """
        Iterate over annotations of the specified type.

        If you need to iterate through several annotations use \
        `AnnotationManager.iter_annotations()` instead.

        :param annotation_type: Type of annotation to iterate through.
        :param start_offset: Start to iterate from the spesific offset. \
                             Can be used as a key argument only.
        :return: Iterator through annotations of requested type.
        """
        search_from = 0
        if start_offset is not None:
            check_offset(start_offset, "start_offset")
            search_from = self._type_to_annotations[annotation_type].bisect_left(
                (start_offset, start_offset))
        for value in self._type_to_annotations[annotation_type].values()[search_from:]:
            yield value

    def _find_annotations(self, annotation_type: Type[Annotation], start: int, stop: int,
                          inspect: Callable, action: str) -> Annotation:
        try:
            annotation_layer = self._type_to_annotations[annotation_type]
        except KeyError:
            raise NoAnnotation("There is no annotation type %s" % annotation_type)
        check_span(start, stop)
        search_start = max(0, annotation_layer.bisect_left((start, start)) - 1)
        search_stop = annotation_layer.bisect_right((stop, stop))
        for span in annotation_layer.islice(search_start, search_stop):
            if inspect(*span):
                return annotation_layer[span]
        raise NoAnnotation("There is no annotation %s that %s [%d, %d)" % (
            annotation_type, action, start, stop))

    def find_overlapping_annotation(self, annotation_type: Type[Annotation],
                                    start: int, stop: int) -> Annotation:
        """
        Find an annotation of the given type that intersects the interval [start, stop).

        :param annotation_type: Annotation type to look for.
        :param start: Start of the search interval.
        :param stop: End of the search interval. Stop point itself is excluded.
        :raise NoAnnotation: There is no such annotation that overlaps with the given interval.
        :return: `Annotation` of the requested type.
        """
        return self._find_annotations(
            annotation_type, start, stop, action="overlaps",
            inspect=lambda span_start, span_stop: self._check_spans_overlap(
                start, stop, span_start, span_stop))

    def find_covering_annotation(self, annotation_type: Type[Annotation],
                                 start: int, stop: int) -> Annotation:
        """
        Find an annotation of the given type that fully covers the interval [start, stop).

        :param annotation_type: Annotation type to look for.
        :param start: Start of the search interval.
        :param stop: End of the search interval. Stop point itself is excluded.
        :raise NoAnnotation: There is no such annotation that overlaps with the given interval.
        :return: `Annotation` of the requested type.
        """
        def check_cover(span_start, span_stop):
            if start == stop:
                return self._check_spans_overlap(start, stop, span_start, span_stop)
            return span_start <= start and stop <= span_stop

        return self._find_annotations(annotation_type, start, stop, action="contains",
                                      inspect=check_cover)

    @classmethod
    def _check_spans_overlap(cls, start1: int, stop1: int, start2: int, stop2: int) -> bool:
        """
        Check if two spans have at least 1 common point.

        Span 1 is [start1, stop1). `stop1` itself is excluded.
        Span 2 is [start2, stop2). `stop2` itself is excluded.

        Everywhere in next examples x < y < z.
        Corner cases explained:
        1. [x, y) and [y, z) have no overlap because y is excluded from the 1st interval.
        2. 0-intervals:
            2.1. [y, y) and [y, y) are overlapping because it is the same interval.
            2.2. [y, y) and [y, z) have no overlap.
            2.3. [x, y) and [y, y) have no overlap.
            2.4. [x, z) and [y, y) are overlapping because [x, z) fully covers y point.

        Despite the fact that overlapping rules are defined for 0-intervals, it is unsafe \
        to rely on them. If you want to get an additional annotation of the 0-interval \
        annotation, link one annotation to another. See `TokenAnnotation` for example.

        :param start1: Start offset of the first span.
        :param stop1: Stop offset of the first span.
        :param start2: Start offset of the second span.
        :param stop2: Stop offset of the second span.
        :return: True if two spans overlap, otherwise False.
        """
        if start1 == stop1:
            if start2 == stop2:
                return start1 == start2
            return start2 < start1 < stop2
        if start2 == stop2:
            return start1 < start2 < stop1
        return (start1 <= start2 < stop1 or
                start1 < stop2 < stop1 or
                start2 <= start1 < stop2)

    @classmethod
    def from_file(cls, file: UnicodeFile) -> "AnnotationManager":
        """
        Create `AnnotationManager` instance from `UnicodeFile`.

        :param file: `file.content` will be used as data to be annotated with \
                     `file.path`, `file.language` and `file.uast`.
        :return: new AnnotationManager instance.
        """
        raw_data = file.content
        annotated_data = AnnotationManager(raw_data)
        annotated_data.add(PathAnnotation(0, len(raw_data), file.path))
        annotated_data.add(UASTAnnotation(0, len(raw_data), file.uast))
        annotated_data.add(LanguageAnnotation(0, len(raw_data), file.language))
        return annotated_data
