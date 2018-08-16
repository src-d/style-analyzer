import lzma
import unittest
from collections import defaultdict
from pathlib import Path

import bblfsh

from lookout.core.api.service_data_pb2 import File
from lookout.style.format.features import (CLASSES, CLS_NEWLINE,
                                           CLS_SINGLE_QUOTE, CLS_SPACE,
                                           CLS_SPACE_DEC, CLS_SPACE_INC,
                                           FeatureExtractor)


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

        X, y = self.extractor.extract_features(files)
        self.assertEqual(X.shape[0], y.shape[0])
        self.assertTrue(X.shape[1] % (1 + self.extractor.parents_depth
                                      + self.extractor.siblings_window * 2) == 0)
        self.assertTrue(X.shape[1],
                        (1 + self.extractor.parents_depth + self.extractor.siblings_window * 2)
                        * self.extractor.feature_names)


if __name__ == "__main__":
    unittest.main()
