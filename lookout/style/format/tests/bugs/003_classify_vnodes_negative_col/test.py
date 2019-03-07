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

    def test_vnode_positions(self):
        test_js_code_filepath = Path(__file__).parent / "jquery.layout.js"
        with open(str(test_js_code_filepath), mode="rb") as f:
            code = f.read()
        uast = bblfsh.BblfshClient("0.0.0.0:9432").parse(
            filename="", language="javascript", contents=code).uast
        file = File(content=code, uast=uast, language="javascript", path="test.js")
        nodes, parents = list(self.extractor._parse_file(AnnotatedData.from_file(file)))
        # Just should not fail
        list(self.extractor._classify_vnodes(nodes, "filepath"))


if __name__ == "__main__":
    unittest.main()
