from pathlib import Path
import tempfile
import unittest

from lookout.core.api.service_data_pb2 import File
from lookout.core.data_requests import parse_uast
import numpy

from lookout.style.format.analyzer import FormatAnalyzer
from lookout.style.format.classes import CLASS_INDEX, CLS_NOOP, CLS_SPACE
from lookout.style.format.feature_extractor import FeatureExtractor
from lookout.style.format.postprocess import filter_uast_breaking_preds
from lookout.style.format.tests.test_analyzer import FakeDataService


class PostprocessingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.language = "javascript"
        cls.data_service = FakeDataService(files=None, changes=None)
        cls.stub = cls.data_service.get_bblfsh()
        cls.bblfsh_address = "0.0.0.0:9432"
        config = {"global": {"feature_extractor": {"cutoff_label_support": 0}}}
        cls.fe = FeatureExtractor(
            language=cls.language,
            **FormatAnalyzer._load_train_config(config)[cls.language]["feature_extractor"])
        cls.parent_loc = Path(__file__).parent.resolve()
        cls.base_dir_ = tempfile.TemporaryDirectory(dir=str(cls.parent_loc))
        cls.base_dir = cls.base_dir_.name
        cls.noop = (CLASS_INDEX[CLS_NOOP],)
        cls.space = (CLASS_INDEX[CLS_SPACE],)

    @classmethod
    def tearDownClass(cls):
        cls.base_dir_.cleanup()

    def test_posprocess(self):
        code = "var a = 15"
        uast, errors = parse_uast(self.stub, code, filename="", language=self.language)
        if errors:
            self.fail("Could not parse the testing code.")
        file = File(content=code.encode(), uast=uast, path="test_file")
        X, y, (vnodes_y, vnodes, vnode_parents, node_parents) = self.fe.extract_features([file])
        y_pred = y.copy()
        rule_winners = numpy.zeros(y.shape)
        # Introduce a bad pred in position 1 (between var and a): noop will break the code
        y_pred[1] = self.fe.class_sequences_to_labels[self.noop]
        new_y, new_y_pred, new_vnodes_y, new_rule_winners, safe_preds = filter_uast_breaking_preds(
            y, y_pred, vnodes_y, vnodes, {"test_file": file}, self.fe, "0.0.0.0:9432",
            vnode_parents, node_parents, rule_winners)
        bad_preds = set(range(y.shape[0])) - set(safe_preds)
        self.assertEqual(bad_preds, {1})
        self.assertEqual(len(y) - 1, len(new_y))
        self.assertEqual(len(y_pred) - 1, len(new_y_pred))
        self.assertEqual(len(vnodes_y) - 1, len(new_vnodes_y))
        self.assertEqual(len(rule_winners) - 1, len(new_rule_winners))


if __name__ == "__main__":
    unittest.main()
