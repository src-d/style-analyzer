from copy import deepcopy
import lzma
from pathlib import Path
import unittest

import bblfsh
from lookout.core.api.service_data_pb2 import File

from lookout.style.format.analyzer import FormatAnalyzer
from lookout.style.format.code_generator import CodeGenerator
from lookout.style.format.feature_extractor import FeatureExtractor
from lookout.style.format.rules import Rule, RuleStats
from lookout.style.format.tests.code_generator_data import cases, label_composites
from lookout.style.format.tests.test_analyzer import get_train_config


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
        for case in cases:
            name = case.replace(" ", "_")
            attrs["test_local_%s" % name] = mcs.generate_local_test(case)

        return super(GeneratorTestsMeta, mcs).__new__(mcs, name, bases, attrs)

    @classmethod
    def generate_local_test(mcs, case_name):
        y_indexes, y_pred, result = cases[case_name]

        def _test(self):
            y_cur = deepcopy(self.y)
            for i, yi in zip(y_indexes, y_pred):
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
        config = FormatAnalyzer._load_train_config(get_train_config())
        fe_config = config["javascript"]
        cls.feature_extractor = FeatureExtractor(language="javascript",
                                                 label_composites=label_composites,
                                                 **fe_config["feature_extractor"])
        cls.file = File(content=bytes(contents, "utf-8"), uast=uast)
        cls.X, cls.y, (cls.vnodes_y, cls.vnodes, cls.vnode_parents, cls.node_parents) = \
            cls.feature_extractor.extract_features([cls.file])

    def test_reproduction(self):
        code_generator = CodeGenerator(self.feature_extractor)
        generated_file = code_generator.generate(self.vnodes)
        self.assertEqual(generated_file, self.file.content.decode("utf-8"))

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
        config = FormatAnalyzer._load_train_config(get_train_config())
        fe_config = config["javascript"]

        for case in cases:
            y_indexes, y_pred, _ = cases[case]
            feature_extractor = FeatureExtractor(language="javascript",
                                                 label_composites=label_composites,
                                                 **fe_config["feature_extractor"])
            file = File(content=bytes(contents, "utf-8"), uast=uast)
            X, y, (vnodes_y, vnodes, vnode_parents, node_parents) = \
                feature_extractor.extract_features([file])
            y_cur = deepcopy(y)
            for i, yi in zip(y_indexes, y_pred):
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


if __name__ == "__main__":
    unittest.main()
