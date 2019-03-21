from copy import deepcopy
import lzma
from pathlib import Path
import unittest

import bblfsh
from lookout.core.analyzer import UnicodeFile

from lookout.style.format.analyzer import FormatAnalyzer
import lookout.style.format.classes as cls
from lookout.style.format.code_generator import CodeGenerator, InapplicableIndentation
from lookout.style.format.feature_extractor import FeatureExtractor
from lookout.style.format.rules import Rule, RuleStats
from lookout.style.format.tests.code_generator_data import cases, label_composites
from lookout.style.format.tests.test_analyzer import get_config
from lookout.style.format.virtual_node import Position, VirtualNode


class FakeSeq:
    def __init__(self, ys):
        self.ys = ys

    def __getitem__(self, item):
        return Rule(tuple(), RuleStats(self.ys[item], 1., 1000), False)


class FakeRules:
    def __init__(self, ys):
        self.rules = FakeSeq(ys)


class GeneratorTestsMeta(type):
    def __new__(mcs, name, bases, attrs):
        base = Path(__file__).parent
        # str() is needed for Python 3.5
        with lzma.open(str(base / "benchmark_small.js.xz"), mode="rt") as fin:
            contents = fin.read()
        with lzma.open(str(base / "benchmark_small.js.uast.xz")) as fin:
            uast = bblfsh.Node.FromString(fin.read())
        for case in cases:
            name = case.replace(" ", "_")
            attrs["test_local_%s" % name] = mcs.generate_local_test(case, uast, contents)

        return super(GeneratorTestsMeta, mcs).__new__(mcs, name, bases, attrs)

    @classmethod
    def generate_local_test(mcs, case_name, uast, contents):
        fe_config = FormatAnalyzer._load_config(get_config())["train"]["javascript"]
        feature_extractor = FeatureExtractor(language="javascript",
                                             label_composites=label_composites,
                                             **fe_config["feature_extractor"])
        file = UnicodeFile(content=contents, uast=uast, path="", language="")
        _, _, (vnodes_y, _, _, _) = feature_extractor.extract_features([file])
        offsets, y_pred, result = cases[case_name]

        def _test(self):
            y_cur = deepcopy(self.y)
            for offset, yi in zip(offsets, y_pred):
                i = None
                for i, vnode in enumerate(vnodes_y):  # noqa: B007
                    if offset == vnode.start.offset:
                        break
                y_cur[i] = yi
            code_generator = CodeGenerator(self.feature_extractor)
            pred_vnodes = code_generator.apply_predicted_y(
                self.vnodes, self.vnodes_y, list(range(len(self.vnodes_y))), FakeRules(y_cur))
            generated_file = code_generator.generate(pred_vnodes)
            self.assertEqual(generated_file, result)

        return _test


class CodeGeneratorTests(unittest.TestCase, metaclass=GeneratorTestsMeta):
    @classmethod
    def setUpClass(cls):
        cls.maxDiff = None
        base = Path(__file__).parent
        # str() is needed for Python 3.5
        with lzma.open(str(base / "benchmark_small.js.xz"), mode="rt") as fin:
            contents = fin.read()
        with lzma.open(str(base / "benchmark_small.js.uast.xz")) as fin:
            uast = bblfsh.Node.FromString(fin.read())
        config = FormatAnalyzer._load_config(get_config())
        fe_config = config["train"]["javascript"]
        cls.feature_extractor = FeatureExtractor(language="javascript",
                                                 label_composites=label_composites,
                                                 **fe_config["feature_extractor"])
        cls.file = UnicodeFile(content=contents, uast=uast, path="", language="")
        cls.X, cls.y, (cls.vnodes_y, cls.vnodes, cls.vnode_parents, cls.node_parents) = \
            cls.feature_extractor.extract_features([cls.file])

    def test_reproduction(self):
        code_generator = CodeGenerator(self.feature_extractor)
        generated_file = code_generator.generate(self.vnodes)
        self.assertEqual(generated_file, self.file.content)

    def test_generate_new_line(self):
        self.maxDiff = None
        expected_res = {
            "nothing changed": [],
            "remove new line in the end of 4th line": None,
            "indentation in the beginning": [
                " import { makeToast } from '../../common/app/Toasts/redux';"],
            "remove indentation in the 4th line till the end": [" return Object.keys(flash)",
                                                                " }"],
            "new line between 6th and 7th regular code lines": [
                "\n      return messages.map(message => ({"],
            "new line in the middle of the 7th code line with indentation increase": [
                "      return messages\n        .map(message => ({", "  })"],
            "new line in the middle of the 7th code line with indentation decrease": [
                "      return messages\n    .map(message => ({", "      })"],
            "new line in the middle of the 7th code line without indentation increase": [
                "      return messages\n      .map(message => ({"],
            "change quotes": ['import { makeToast } from "../../common/app/Toasts/redux";'],
            "remove indentation decrease 11th line": ["        }));"],
            "change indentation decrease to indentation increase 11th line": ["          }));"],
            "change indentation decrease to indentation increase 11th line but keep the rest": [
                "          }));", "})"],
        }

        base = Path(__file__).parent
        # str() is needed for Python 3.5
        with lzma.open(str(base / "benchmark_small.js.xz"), mode="rt") as fin:
            contents = fin.read()
        with lzma.open(str(base / "benchmark_small.js.uast.xz")) as fin:
            uast = bblfsh.Node.FromString(fin.read())
        config = FormatAnalyzer._load_config(get_config())
        fe_config = config["train"]["javascript"]

        for case in expected_res:
            offsets, y_pred, _ = cases[case]
            feature_extractor = FeatureExtractor(language="javascript",
                                                 label_composites=label_composites,
                                                 **fe_config["feature_extractor"])
            file = UnicodeFile(content=contents, uast=uast, path="", language="")
            X, y, (vnodes_y, vnodes, vnode_parents, node_parents) = \
                feature_extractor.extract_features([file])
            y_cur = deepcopy(y)
            for offset, yi in zip(offsets, y_pred):
                i = None
                for i, vnode in enumerate(vnodes_y):  # noqa: B007
                    if offset == vnode.start.offset:
                        break
                y_cur[i] = yi
            code_generator = CodeGenerator(feature_extractor)
            pred_vnodes = code_generator.apply_predicted_y(
                vnodes, vnodes_y, list(range(len(vnodes_y))), FakeRules(y_cur))
            res = []
            for gln in FormatAnalyzer._group_line_nodes(
                    y, y_cur, vnodes_y, pred_vnodes, [1] * len(y)):
                line, (line_y, line_y_pred, line_vnodes_y, line_vnodes, line_rule_winners) = gln
                new_code_line = code_generator.generate_new_line(line_vnodes)
                res.append(new_code_line)
            if expected_res[case] is not None:
                # None means that we delete some lines. We are not handle this properly now.
                self.assertEqual(res, expected_res[case], case)

    def test_revert_indentation_change(self):
        cases = [
            ("\n    ", (cls.CLS_NEWLINE, cls.CLS_SPACE_INC, cls.CLS_SPACE_INC), "\n  "),
            ("\n    ", (cls.CLS_NEWLINE, cls.CLS_SPACE_DEC, cls.CLS_SPACE_DEC), "\n      "),
            ("\n\t ", (cls.CLS_NEWLINE, cls.CLS_TAB_INC, cls.CLS_SPACE_INC), "\n"),
            ("\n    ", (cls.CLS_NEWLINE, cls.CLS_TAB_INC, cls.CLS_TAB_INC),
             InapplicableIndentation),
            ("   ", (cls.CLS_SPACE, cls.CLS_SPACE_INC, cls.CLS_SPACE_INC), ValueError),
        ]
        for value, y, result in cases:
            vnode = VirtualNode(value, Position(0, 1, 1), Position(len(value), 1, len(value)+1),
                                y=tuple(cls.CLASS_INDEX[i] for i in y))
            if isinstance(result, str):
                self.assertEqual(CodeGenerator.revert_indentation_change(vnode), result)
            else:
                with self.assertRaises(result):
                    CodeGenerator.revert_indentation_change(vnode)

    def test_apply_new_indentation(self):
        cases = [
            ("\n    ", ("\n", "  "),
             (cls.CLS_NEWLINE, cls.CLS_SPACE_INC, cls.CLS_SPACE_INC),
             (cls.CLS_NEWLINE, ),
             ""),
            ("\n    ", ("\n", "      "),
             (cls.CLS_NEWLINE, cls.CLS_SPACE_DEC, cls.CLS_SPACE_DEC),
             (cls.CLS_NEWLINE, ),
             ""),
            ("\n\t ", ("\n", ""),
             (cls.CLS_NEWLINE, cls.CLS_TAB_INC, cls.CLS_SPACE_INC),
             (cls.CLS_NEWLINE, ),
             ""),
            ("\n    ", InapplicableIndentation,
             (cls.CLS_NEWLINE, cls.CLS_TAB_INC, cls.CLS_TAB_INC),
             (cls.CLS_NEWLINE, ),
             ""),
            ("\n   ", ValueError,
             (cls.CLS_NEWLINE, cls.CLS_SPACE, cls.CLS_SPACE_INC, cls.CLS_SPACE_INC),
             (cls.CLS_NEWLINE, ),
             ""),
            ("\n\t  ", InapplicableIndentation,
             (cls.CLS_NEWLINE, cls.CLS_SPACE_INC, cls.CLS_SPACE_INC),
             (cls.CLS_NEWLINE, cls.CLS_SPACE_DEC),
             ""),
            ("\n\t   ", ValueError,
             (cls.CLS_NEWLINE, cls.CLS_SPACE_DEC),
             (cls.CLS_NEWLINE, cls.CLS_SPACE_DEC, cls.CLS_SPACE, cls.CLS_SPACE_DEC),
             ""),
            ("\n\n    ", ("\n", "  "),
             (cls.CLS_NEWLINE, cls.CLS_NEWLINE, cls.CLS_SPACE_DEC),
             (cls.CLS_NEWLINE, cls.CLS_SPACE_DEC, cls.CLS_SPACE_DEC, cls.CLS_SPACE_DEC),
             ""),
            ("", ("\n", "  "),
             (cls.CLS_NOOP, ),
             (cls.CLS_NEWLINE, ),
             "  "),
            ("", ("\n\n", ""),
             (cls.CLS_NOOP,),
             (cls.CLS_NEWLINE, cls.CLS_NEWLINE),
             ""),
        ]
        for value, result, y_old, y, last_ident in cases:
            vnode = VirtualNode(value, Position(0, 1, 1), Position(len(y), 1, len(y) + 1),
                                y=tuple(cls.CLASS_INDEX[i] for i in y))
            vnode.y_old = tuple(cls.CLASS_INDEX[i] for i in y_old)
            if isinstance(result, tuple):
                self.assertEqual(CodeGenerator.apply_new_indentation(vnode, last_ident), result)
            else:
                with self.assertRaises(result):
                    CodeGenerator.apply_new_indentation(vnode, last_ident)

        msg = None

        def _warning(*args):
            nonlocal msg
            msg = args[0]
        try:
            backup_warning = CodeGenerator._log.warning
            CodeGenerator._log.warning = _warning
            vnode = VirtualNode(
                "\n ", Position(0, 1, 1), Position(3, 1, 4),
                y=tuple(cls.CLASS_INDEX[i] for i in (
                    cls.CLS_NEWLINE, cls.CLS_SPACE_DEC, cls.CLS_SPACE_DEC, cls.CLS_SPACE_DEC)))
            vnode.y_old = tuple(cls.CLASS_INDEX[i] for i in (cls.CLS_NEWLINE, cls.CLS_SPACE_DEC))
            CodeGenerator.apply_new_indentation(vnode, "")
            expected_msg = "There is no indentation characters left to decrease for vnode"
            self.assertEqual(msg[:len(expected_msg)], expected_msg)
        finally:
            CodeGenerator._log.warning = backup_warning


if __name__ == "__main__":
    unittest.main()
