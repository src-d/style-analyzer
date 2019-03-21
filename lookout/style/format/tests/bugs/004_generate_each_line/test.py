from itertools import repeat
from pathlib import Path
import sys
import unittest

import bblfsh

from lookout.style.format.analyzer import FormatAnalyzer
from lookout.style.format.code_generator import CodeGenerator
from lookout.style.format.feature_extractor import FeatureExtractor
from lookout.style.format.tests.test_analyzer import FakeFile, get_config


class FeaturesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config = FormatAnalyzer._load_config(get_config())["train"]
        cls.extractor = FeatureExtractor(language="javascript",
                                         **config["javascript"]["feature_extractor"])
        test_js_code_filepath = Path(__file__).parent / "jquery.layout.js"
        with open(str(test_js_code_filepath), mode="r") as f:
            cls.code = f.read()
        cls.uast = bblfsh.BblfshClient("0.0.0.0:9432").parse(
            filename="", language="javascript", contents=cls.code.encode()).uast
        feature_extractor_output = cls.extractor.extract_features([FakeFile(
            path="test.py", content=cls.code, uast=cls.uast, language="JavaScript")])
        X, cls.y, (cls.vnodes_y, cls.vnodes, vnode_parents, node_parents) = \
            feature_extractor_output

    def test_token_parser(self):
        file_content = "".join([node.value for node in self.vnodes])
        ok = True
        for n, (correct_line, line) in enumerate(zip(
                self.code.splitlines(keepends=True),
                file_content.splitlines(keepends=True))):
            if correct_line != line:
                print("Lines %d are different" % (n + 1), file=sys.stderr)
                print("    Correct:  ", repr(correct_line), file=sys.stderr)
                print("    Restored: ", repr(line), file=sys.stderr)
                ok = False
        self.assertTrue(ok, "Original and restored files are different")

    def test_vnode_positions(self):
        code_generator = CodeGenerator(feature_extractor=self.extractor)
        lines = self.code.splitlines()
        lines.append("\n")
        ok = True
        for line_number, line in FormatAnalyzer._group_line_nodes(
                self.y, self.y - 1, self.vnodes_y, self.vnodes, repeat(0)):
            line_ys, line_ys_pred, line_vnodes_y, new_line_vnodes, line_winners = line
            new_code_line = code_generator.generate_new_line(new_line_vnodes)
            if lines[line_number - 1] != new_code_line:
                print("Lines %d are different" % line_number, file=sys.stderr)
                print(repr(lines[line_number - 1]), file=sys.stderr)
                print(repr(new_code_line), file=sys.stderr)
                print()
                ok = False
        self.assertTrue(ok, "Original and restored lines are different")


if __name__ == "__main__":
    unittest.main()
