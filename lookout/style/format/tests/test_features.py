import lzma
from pathlib import Path
import unittest

import bblfsh

from lookout.style.format.features import parse_file


class FeaturesTests(unittest.TestCase):
    def test_parse_file(self):
        base = Path(__file__).parent
        # str() is needed for Python 3.5
        with lzma.open(str(base / "benchmark.js.xz"), mode="rt") as fin:
            contents = fin.read()
        with lzma.open(str(base / "benchmark.uast.xz")) as fin:
            uast = bblfsh.Node.FromString(fin.read())
        nodes, parents = parse_file(contents, uast, "javascript")
        text = []
        for n in nodes:
            text.append(n.value)
            if n.node is not None:
                self.assertIsNotNone(parents.get(id(n.node)), n)
        self.assertEqual("".join(text), contents)


if __name__ == "__main__":
    unittest.main()
