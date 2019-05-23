import unittest

from lookout.style.format.annotations.annotated_data import AnnotationManager, AnnotationsSpan, \
    NoAnnotation
from lookout.style.format.annotations.annotations import Annotation, check_offset, check_span


class AnotherAnnotation(Annotation):
    pass


class AnnotationsTests(unittest.TestCase):
    def test_add(self):
        code = "0123456789"
        annotated_data = AnnotationManager(code)
        annotations = [Annotation(0, 3), Annotation(3, 3), Annotation(3, 4)]
        another_annotation = AnotherAnnotation(0, len(code) - 1)
        annotated_data.add(*annotations)
        annotated_data.add(another_annotation)

        overlapping_annotations = annotations + [
            Annotation(0, 1), Annotation(1, 4), Annotation(3, 3), Annotation(2, 5)]
        for annotation in overlapping_annotations:
            with self.assertRaises(ValueError):
                annotated_data.add(annotation)
        ok_annotations = [Annotation(0, 0), Annotation(4, 4), Annotation(9, 11), Annotation(4, 9)]
        for annotation in ok_annotations:
            annotated_data.add(annotation)

    def test_len(self):
        self.assertEqual(0, len(AnnotationManager("")))
        self.assertEqual(100, len(AnnotationManager(100*"1")))

    def test_getitem(self):
        am = AnnotationManager("")
        self.assertEqual(am[:1], "")
        with self.assertRaises(IndexError):
            am[0]

        seq = "0123456789"
        am = AnnotationManager(seq)
        self.assertEqual(am[0], seq[0])
        self.assertEqual(am[0:5], seq[0:5])
        self.assertEqual(am[(5, 7)], seq[5:7])
        self.assertEqual(am[7:5], seq[7:5])
        with self.assertRaises(IndexError):
            am[len(seq)+1]

    def test_count(self):
        seq = "0123456789"
        am = AnnotationManager(seq)

        self.assertEqual(am.count(Annotation), 0)
        am.add(Annotation(0, 1))
        self.assertEqual(am.count(Annotation), 1)
        am.add(Annotation(4, 5), Annotation(3, 4), Annotation(4, 4))
        self.assertEqual(am.count(Annotation), 4)
        self.assertEqual(am.count(AnotherAnnotation), 0)
        am.add(AnotherAnnotation(4, 8))
        self.assertEqual(am.count(Annotation), 4)
        self.assertEqual(am.count(AnotherAnnotation), 1)
        am.add(AnotherAnnotation(0, 3), AnotherAnnotation(3, 4), AnotherAnnotation(8, 8))
        self.assertEqual(am.count(Annotation), 4)
        self.assertEqual(am.count(AnotherAnnotation), 4)

    def test_get(self):
        code = "0123456789"
        annotated_data = AnnotationManager(code)
        with self.assertRaises(Exception):
            annotated_data.get(Annotation)
        annotated_data.add(Annotation(4, 7))
        annotated_data.get(Annotation, (4, 7))
        with self.assertRaises(Exception):
            annotated_data.get(Annotation)

    def test_iter_annotations(self):
        code = "0123456789"
        annotated_data = AnnotationManager(code)
        annotations = [Annotation(0, 3), Annotation(3, 3), Annotation(3, 4)]
        another_annotation = AnotherAnnotation(0, len(code)-1)
        annotated_data.add(*annotations[::-1])
        annotated_data.add(another_annotation)
        res = [AnnotationsSpan(0, 3, {Annotation: annotations[0],
                                      AnotherAnnotation: another_annotation}),
               AnnotationsSpan(3, 3, {Annotation: annotations[1],
                                      AnotherAnnotation: another_annotation}),
               AnnotationsSpan(3, 4, {Annotation: annotations[2],
                                      AnotherAnnotation: another_annotation})]
        self.assertEqual(list(annotated_data.iter_by_type_nested(
            Annotation, AnotherAnnotation)), res)

        annotations = list(annotated_data.iter_by_type_nested(AnotherAnnotation, Annotation))
        res = [AnnotationsSpan(0, len(code)-1, {AnotherAnnotation: another_annotation})]
        self.assertEqual(annotations, res)

    def test_iter_annotation(self):
        code = "0123456789"
        annotated_data = AnnotationManager(code)
        with self.assertRaises(KeyError):
            list(annotated_data.iter_by_type(Annotation))

        annotations = [Annotation(0, 3), Annotation(3, 3), Annotation(3, 4)]
        annotated_data.add(*annotations[::-1])
        self.assertEqual(list(annotated_data.iter_by_type(Annotation)),
                         annotations)
        more_annotations = [Annotation(0, 0), Annotation(4, 4), Annotation(4, 7)]
        annotated_data.add(*more_annotations)
        self.assertEqual(list(annotated_data.iter_by_type(Annotation)),
                         sorted(annotations + more_annotations, key=lambda x: x.span))

    def test_find_overlapping_annotation(self):
        code = "0123456789"
        annotated_data = AnnotationManager(code)
        annotations = [Annotation(0, 3), Annotation(3, 3), Annotation(3, 4)]
        annotated_data.add(*annotations[::-1])
        for annotation in annotations:
            self.assertEqual(
                annotated_data.find_overlapping_annotation(Annotation, *annotation.span),
                annotation)
        self.assertEqual(annotated_data.find_overlapping_annotation(Annotation, 1, 2),
                         annotations[0])
        self.assertEqual(annotated_data.find_overlapping_annotation(Annotation, 3, 5),
                         annotations[2])
        self.assertEqual(annotated_data.find_overlapping_annotation(Annotation, 2, 4),
                         annotations[0])
        for span in [(4, 4), (4, 5), (5, 5)]:
            with self.assertRaises(NoAnnotation):
                annotated_data.find_overlapping_annotation(Annotation, *span)

    def test_find_covering_annotation(self):
        code = "0123456789"
        annotated_data = AnnotationManager(code)
        annotations = [Annotation(0, 3), Annotation(3, 3), Annotation(3, 4)]
        annotated_data.add(*annotations[::-1])
        for annotation in annotations:
            self.assertEqual(
                annotated_data.find_covering_annotation(Annotation, *annotation.span),
                annotation)
        self.assertEqual(annotated_data.find_covering_annotation(Annotation, 1, 2),
                         annotations[0])
        self.assertEqual(annotated_data.find_covering_annotation(Annotation, 1, 1),
                         annotations[0])
        for span in [(4, 4), (4, 5), (5, 5), (3, 5), (2, 4)]:
            with self.assertRaises(NoAnnotation):
                annotated_data.find_covering_annotation(Annotation, *span)

    def test_check_interval_crossing(self):
        data = [
            ((9, 19), (19, 20), False),
            ((19, 20), (9, 19), False),
            ((1, 3), (2, 4), True),
            ((2, 4), (1, 3), True),
            ((-2, 4), (1, 3), True),
            ((-2, 3), (1, 3), True),
            ((1, 3), (1, 3), True),
            ((1, 3), (6, 7), False),
            ((10, 30), (6, 7), False),
            ((10, 10), (10, 10), True),
            ((10, 30), (10, 10), False),
            ((10, 10), (10, 30), False),
            ((10, 10), (5, 30), True),
            ((5, 30), (10, 10), True),
        ]
        for i, (interval1, interval2, res) in enumerate(data):
            self.assertEqual(AnnotationManager._check_spans_overlap(*interval1, *interval2),
                             res, "Case # %d" % i)

    def test_annotations_span(self):
        start = 10
        stop = 20
        span = AnnotationsSpan(start, stop)
        self.assertEqual(span.start, start)
        self.assertEqual(span.stop, stop)
        self.assertEqual(span.span, (start, stop))
        with self.assertRaises(AttributeError):
            span.start = 11
        with self.assertRaises(AttributeError):
            span.stop = 21
        with self.assertRaises(AttributeError):
            span.span = (1, 2)

    def test_check_span(self):
        check_span(1, 2)
        check_span(0, 20)
        for span in [(-1, 1), (-10, -20), (-1, -1), (20, 10), ("a", 3), (1, 5.5)]:
            with self.assertRaises(ValueError):
                check_span(*span)

    def test_check_offset(self):
        check_offset(1, "offset")
        check_offset(100, "offset")
        for offset in [-1, -1000, "a", 5.4]:
            with self.assertRaises(ValueError):
                check_offset(offset, "offset")


if __name__ == "__main__":
    unittest.main()
