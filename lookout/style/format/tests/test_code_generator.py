from copy import deepcopy
import lzma
from pathlib import Path
import unittest

import bblfsh

from lookout.core.api.service_data_pb2 import File
from lookout.style.format.analyzer import FormatAnalyzer
from lookout.style.format.code_generator import CodeGenerator
from lookout.style.format.feature_extractor import FeatureExtractor
from lookout.style.format.tests.code_generator_data import cases, label_composites


class GeneratorTestsMeta(type):
    def __new__(mcs, name, bases, attrs):
        for case in cases:
            name = case.replace(" ", "_")
            attrs["test_%s" % name] = mcs.generate_test(case)
        for case in cases:
            name = case.replace(" ", "_")
            attrs["test_local_%s" % name] = mcs.generate_local_test(case)

        return super(GeneratorTestsMeta, mcs).__new__(mcs, name, bases, attrs)

    @classmethod
    def generate_test(cls, case_name):
        y_indexes, y_pred, result, *_ = cases[case_name]

        def _test(self):
            y_cur = deepcopy(self.y)
            for i, yi in zip(y_indexes, y_pred):
                y_cur[i] = yi
            code_generator = CodeGenerator(self.feature_extractor)
            generated_file = code_generator.generate(
                self.vnodes_y, y_cur, self.vnodes)
            self.assertEqual(generated_file, result)
        return _test

    @classmethod
    def generate_local_test(cls, case_name):
        y_indexes, y_pred, *result = cases[case_name]
        result_local = result[-1]

        def _test(self):
            y_cur = deepcopy(self.y)
            for i, yi in zip(y_indexes, y_pred):
                y_cur[i] = yi
            code_generator = CodeGenerator(self.feature_extractor, change_locally=True)
            generated_file = code_generator.generate(
                self.vnodes_y, y_cur, self.vnodes)
            self.assertEqual(generated_file, result_local)

        return _test


class AnalyzerTests(unittest.TestCase, metaclass=GeneratorTestsMeta):
    @classmethod
    def setUpClass(cls):
        cls.maxDiff = None
        base = Path(__file__).parent
        # str() is needed for Python 3.5
        with lzma.open(str(base / "benchmark_small.js.xz"), mode="rt") as fin:
            contents = fin.read()
        with lzma.open(str(base / "benchmark_small.js.uast.xz")) as fin:
            uast = bblfsh.Node.FromString(fin.read())
        config = FormatAnalyzer._load_train_config({})
        fe_config = config["javascript"]
        fe_config["feature_extractor"]["insert_noops"] = True
        cls.feature_extractor = FeatureExtractor(language="javascript",
                                                 label_composites=label_composites,
                                                 **fe_config["feature_extractor"])
        cls.file = File(content=bytes(contents, "utf-8"), uast=uast)
        cls.X, cls.y, cls.vnodes_y, cls.vnodes = \
            cls.feature_extractor.extract_features([cls.file])


if __name__ == "__main__":
    unittest.main()
