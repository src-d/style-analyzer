from pathlib import Path
import unittest

import bblfsh
from lookout.core.api.service_data_pb2 import File

from lookout.style.format.analyzer import FormatAnalyzer
from lookout.style.format.annotations.annotated_data import AnnotatedData
from lookout.style.format.feature_extractor import FeatureExtractor
from lookout.style.format.tests.test_analyzer import get_config


class FeaturesTests(unittest.TestCase):
    def setUp(self):
        config = FormatAnalyzer._load_config(get_config())["train"]
        self.extractor = FeatureExtractor(language="javascript",
                                          **config["javascript"]["feature_extractor"])

    def test_positions(self):
        test_js_code_filepath = Path(__file__).parent / "browser-policy-content.js"
        with open(str(test_js_code_filepath), mode="rb") as f:
            code = f.read()
        uast = bblfsh.BblfshClient("0.0.0.0:9432").parse(
            filename="", language="javascript", contents=code).uast
        file = File(content=code, uast=uast, language="javascript", path="test.js")
        nodes, parents = self.extractor._parse_file(AnnotatedData.from_file(file))
        for index, (node1, node2) in enumerate(zip(nodes, nodes[1:])):
            self.assertLessEqual(node1.start.line, node2.start.line,
                                 "Start line position decrease for %d, %d nodes" % (
                                     index, index + 1))
            self.assertLessEqual(node1.start.offset, node2.start.offset,
                                 "Start offset position decrease for %d, %d nodes" % (
                                     index, index + 1))


if __name__ == "__main__":
    unittest.main()
