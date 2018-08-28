from collections import defaultdict
import lzma
from pathlib import Path
import unittest

import bblfsh
import numpy

from lookout.core.api.service_data_pb2 import File
from lookout.style.format.features import (
    CLASSES, CLS_NEWLINE, CLS_SINGLE_QUOTE, CLS_SPACE, CLS_SPACE_DEC, CLS_SPACE_INC,
    FeatureExtractor, VirtualNode)


class FeaturesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base = Path(__file__).parent
        # str() is needed for Python 3.5
        with lzma.open(str(base / "benchmark.js.xz"), mode="rt") as fin:
            cls.contents = fin.read()
        with lzma.open(str(base / "benchmark.uast.xz")) as fin:
            cls.uast = bblfsh.Node.FromString(fin.read())
        cls.extractor = FeatureExtractor("javascript", parents_depth=2, siblings_window=5)

    def test_parse_file(self):
        nodes, parents = self.extractor._parse_file(self.contents, self.uast)
        text = []
        offset = line = col = 0
        for n in nodes:
            if line == n.start.line - 1:
                line += 1
                col = 1
            self.assertEqual((offset, line, col), n.start, n.value)
            text.append(n.value)
            if n.node is not None:
                self.assertIsNotNone(parents.get(id(n.node)), n)
            offset, line, col = n.end
        self.assertEqual("".join(text), self.contents)

    def test_classify_vnodes(self):
        nodes, _ = self.extractor._parse_file(self.contents, self.uast)
        nodes = self.extractor._classify_vnodes(nodes)
        text = "".join(n.value for n in nodes)
        self.assertEqual(text, self.contents)
        cls_counts = defaultdict(int)
        offset = line = col = 0
        for n in nodes:
            if line == n.start.line - 1:
                line += 1
                col = 1
            self.assertEqual((offset, line, col), n.start, n.value)
            if n.y is not None:
                cls_counts[CLASSES[n.y]] += 1
            offset, line, col = n.end
        self.assertEqual(cls_counts[CLS_SPACE_INC], cls_counts[CLS_SPACE_DEC])
        self.assertGreater(cls_counts[CLS_SPACE_INC], 0)
        self.assertGreater(cls_counts[CLS_SPACE], 0)
        self.assertGreater(cls_counts[CLS_NEWLINE], 0)
        self.assertGreater(cls_counts[CLS_SINGLE_QUOTE], 0)
        self.assertTrue(cls_counts[CLS_SINGLE_QUOTE] % 2 == 0)

    def test_extract_features(self):
        file = File(content=bytes(self.contents, 'utf-8'),
                    uast=self.uast)
        files = [file, file]
        self.check_X_y(*self.extractor.extract_features(files))

    def check_X_y(self, X, y, vnodes):
        self.assertEqual(X.shape[0], y.shape[0])
        self.assertEqual(X.shape[0], len(vnodes))
        for vn in vnodes:
            self.assertIsInstance(vn, VirtualNode)
        self.assertEqual(
            X.shape[1],
            (len(self.extractor.self_features)
             + self.extractor.siblings_window * len(self.extractor.left_siblings_features)
             + self.extractor.siblings_window * len(self.extractor.right_siblings_features)
             + self.extractor.parents_depth * len(self.extractor.parents_features)))
        # check that no row is full of -1 (no offset errors on instances)
        not_set = X == -1
        unset_rows = numpy.nonzero(numpy.all(not_set, axis=1))[0]
        unset_columns = numpy.nonzero(numpy.all(not_set, axis=0))[0]
        self.assertEqual(len(unset_rows), 0, "%d rows are unset" % len(unset_rows))
        self.assertEqual(len(unset_columns), 0,
                         "columns %s are unset" % ", ".join(map(str, unset_columns)))

    def test_extract_features_all_lines(self):
        file = File(content=bytes(self.contents, 'utf-8'),
                    uast=self.uast)
        files = [file, file]

        self.check_X_y(*self.extractor.extract_features(
            files, [list(range(1, self.contents.count("\n") + 1))] * 2))

    def test_extract_features_some_lines(self):
        file = File(content=bytes(self.contents, 'utf-8'),
                    uast=self.uast)
        files = [file]

        X1, y1, vn1 = self.extractor.extract_features(
            files, [list(range(1, self.contents.count("\n") // 2 + 1))] * 2)
        self.check_X_y(X1, y1, vn1)
        X2, y2, vn2 = self.extractor.extract_features(files)
        self.assertTrue((X1 == X2[:len(X1)]).all())
        self.assertTrue((y1 == y2[:len(y1)]).all())
        self.assertTrue(vn1 == vn2[:len(vn1)])
        self.assertLess(len(y1), len(y2))


if __name__ == "__main__":
    unittest.main()
