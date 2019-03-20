from collections import OrderedDict
from typing import FrozenSet, Mapping, Optional, Sequence, Tuple
import unittest

import bblfsh
from lookout.core.analyzer import UnicodeFile
from lookout.core.data_requests import parse_uast
from lookout.core.slogging import setup as slogging_setup
import numpy

from lookout.style.format.analyzer import FormatAnalyzer
from lookout.style.format.classes import CLASS_INDEX, CLS_DOUBLE_QUOTE, CLS_NOOP, CLS_SINGLE_QUOTE
from lookout.style.format.feature_extractor import FeatureExtractor
from lookout.style.format.tests.test_analyzer import FakeDataService
from lookout.style.format.uast_stability_checker import UASTStabilityChecker
from lookout.style.format.virtual_node import VirtualNode


class PostprocessingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        slogging_setup("DEBUG", False)
        cls.language = "javascript"
        cls.bblfsh_client = bblfsh.BblfshClient("0.0.0.0:9432")
        cls.data_service = FakeDataService(cls.bblfsh_client, files=None, changes=None)
        cls.stub = cls.data_service.get_bblfsh()
        cls.config = FormatAnalyzer._load_config({
            "train": {"language_defaults": {"feature_extractor": {"cutoff_label_support": 0}}},
        })["train"][cls.language]["feature_extractor"]

    @classmethod
    def tearDownClass(cls):
        cls.bblfsh_client._channel.close()

    def setUp(self):
        self.fe = FeatureExtractor(language=self.language, **self.config)

    def _to_label(self, classes: Sequence[str]) -> Tuple[int, ...]:
        return self.fe.class_sequences_to_labels[tuple(CLASS_INDEX[cls] for cls in classes)]

    @staticmethod
    def _grouped_predictions_mapping(vnodes: Sequence[VirtualNode],
                                     indices: Optional[Sequence[int]]):
        result = OrderedDict()
        if indices is None:
            return result
        y_index = [i for i, vnode in enumerate(vnodes) if vnode.y is not None]
        for i in indices:
            y_i = y_index[i]
            result[id(vnodes[y_i])] = (vnodes[y_i], vnodes[y_i + 1], vnodes[y_i + 2])
            result[id(vnodes[y_i + 2])] = None
        return result

    def edit_and_test(self, code: str, modifs: Mapping[int, Sequence[str]], *,
                      quote_indices: Optional[Tuple[int, ...]] = None,
                      bad_indices: Optional[FrozenSet[int]] = None) -> None:
        uast, errors = parse_uast(self.stub, code, filename="", language=self.language,
                                  unicode=True)
        if errors:
            self.fail("Could not parse the testing code.")
        file = UnicodeFile(content=code, uast=uast, path="test_file", language="javascript")
        X, y, (vnodes_y, vnodes, vnode_parents, node_parents) = self.fe.extract_features([file])
        y_pred = y.copy()
        rule_winners = numpy.zeros(y.shape)
        for index, classes in modifs.items():
            y_pred[index] = self._to_label(classes)
        checker = UASTStabilityChecker(self.fe)
        grouped_quote_predictions = self._grouped_predictions_mapping(vnodes, quote_indices)
        new_y, new_y_pred, new_vnodes_y, new_rule_winners, safe_preds = checker.check(
            y, y_pred, vnodes_y, vnodes, [file], self.stub, vnode_parents,
            node_parents, rule_winners, grouped_quote_predictions=grouped_quote_predictions)
        bad_preds = set(range(y.shape[0])) - set(safe_preds)
        bad = modifs.keys() if bad_indices is None else bad_indices
        self.assertEqual(bad_preds, bad)
        self.assertEqual(len(y) - len(bad), len(new_y))
        self.assertEqual(len(y_pred) - len(bad), len(new_y_pred))
        self.assertEqual(len(vnodes_y) - len(bad), len(new_vnodes_y))
        self.assertEqual(len(rule_winners) - len(bad), len(new_rule_winners))

    def test_posprocess(self):
        self.edit_and_test("var a = 0", {1: (CLS_NOOP,)})

    def test_bad_and_good_quotes(self):
        self.edit_and_test("""var a = '"0"'; var c = "0";""",
                           {4: (CLS_DOUBLE_QUOTE,), 5: (CLS_DOUBLE_QUOTE,),
                            10: (CLS_SINGLE_QUOTE,), 11: (CLS_SINGLE_QUOTE,)},
                           quote_indices=(4, 10), bad_indices=frozenset((4, 5)))

    def test_lonely_quote(self):
        self.edit_and_test("var a = 0; var b = 'c';", {2: (CLS_SINGLE_QUOTE)}, quote_indices=(9,))

    def test_multiple_files(self):
        data = [
            ("var a = 0",
             {1: (CLS_NOOP,)}),
            ("var b = 123",
             {4: (CLS_NOOP,)}),
        ]
        files = []
        for i, (code, _) in enumerate(data):
            uast, errors = parse_uast(self.stub, code, filename="", language=self.language,
                                      unicode=True)
            if errors:
                self.fail("Could not parse the testing code.")
            files.append(UnicodeFile(content=code, uast=uast, path="test_file_%d" % i,
                                     language="javascript"))
        X, y, (vnodes_y, vnodes, vnode_parents, node_parents) = self.fe.extract_features(files)
        y_pred = y.copy()
        rule_winners = numpy.zeros(y.shape)
        for (_, modif) in data:
            for i in modif:
                y_pred[i] = self._to_label(modif[i])
        checker = UASTStabilityChecker(self.fe)
        new_y, new_y_pred, new_vnodes_y, new_rule_winners, safe_preds = checker.check(
            y, y_pred, vnodes_y, vnodes, files, self.stub, vnode_parents,
            node_parents, rule_winners, grouped_quote_predictions={})
        self.assertEqual(list(safe_preds), [0, 2, 3, 4, 5, 6, 7, 8])


if __name__ == "__main__":
    unittest.main()
