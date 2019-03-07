import unittest

from lookout.style.format.annotations.annotated_data import AnnotationManager, AnnotationsSpan
from lookout.style.format.annotations.annotations import AtomicTokenAnnotation, PathAnnotation


class AnnotationsTests(unittest.TestCase):
    def test_annotations(self):
        code = "0123456789"
        annotated_data = AnnotationManager(code)
        token_annotations = [AtomicTokenAnnotation(0, 3), AtomicTokenAnnotation(3, 3),
                             AtomicTokenAnnotation(3, 4)]
        path_annotation = PathAnnotation(0, len(code)-1, "1")
        annotated_data.add(*token_annotations)
        annotated_data.add(path_annotation)
        annotations = list(annotated_data.iter_annotations(AtomicTokenAnnotation, PathAnnotation))
        res = [AnnotationsSpan(0, 3, {AtomicTokenAnnotation: token_annotations[0],
                                      PathAnnotation: path_annotation}),
               AnnotationsSpan(3, 3, {AtomicTokenAnnotation: token_annotations[1],
                                      PathAnnotation: path_annotation}),
               AnnotationsSpan(3, 4, {AtomicTokenAnnotation: token_annotations[2],
                                      PathAnnotation: path_annotation})]
        self.assertEqual(annotations, res)

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


if __name__ == "__main__":
    unittest.main()
