import glob
import logging
import os
from pathlib import Path
import tarfile
import tempfile
import unittest

from bblfsh import BblfshClient
from lookout.core.api.service_data_pb2 import File
import numpy

from lookout.style.format.analyzer import FormatAnalyzer
from lookout.style.format.classes import CLASS_INDEX, CLASSES, CLS_NEWLINE, CLS_NOOP, \
    CLS_SINGLE_QUOTE, CLS_SPACE, CLS_SPACE_DEC, CLS_SPACE_INC
from lookout.style.format.feature_extractor import FeatureExtractor
from lookout.style.format.model import FormatModel
from lookout.style.format.postprocess import filter_uast_breaking_preds
from lookout.style.format.utils import prepare_files
from lookout.style.format.virtual_node import Position, VirtualNode


class PostprocessingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bblfsh = "0.0.0.0:9432"
        cls.language = "javascript"

        cls.parent_loc = Path(__file__).parent.resolve()
        cls.base_dir_ = tempfile.TemporaryDirectory(dir=str(cls.parent_loc))
        cls.base_dir = cls.base_dir_.name

        with tarfile.open(str(cls.parent_loc / "jquery_noisy.tar.xz")) as tar:
            tar.extractall(path=cls.base_dir)
        cls.jquery_noisy_dir = os.path.join(cls.base_dir, "jquery_noisy")
        cls.input_file = os.path.join(cls.jquery_noisy_dir, "28-xhr.js")
        cls.model_path = str(Path(__file__).parent.resolve() / "model_jquery.asdf")

    @classmethod
    def tearDownClass(cls):
        cls.base_dir_.cleanup()

    def test_posprocess(self):
        log = logging.getLogger("postprocess")
        model = FormatModel().load(self.model_path)
        rules = model[self.language]
        client = BblfshClient(self.bblfsh)
        files = prepare_files([self.input_file], client, self.language)
        fe = FeatureExtractor(language=self.language, **rules.origin_config["feature_extractor"])
        res = fe.extract_features(files)
        X, y, (vnodes_y, vnodes, vnode_parents, node_parents) = res
        y_pred, winners = rules.predict(X=X, vnodes_y=vnodes_y, vnodes=vnodes, feature_extractor=fe)
        y, y_pred, vnodes_y, safe_preds = filter_uast_breaking_preds(
            y=y, y_pred=y_pred, vnodes_y=vnodes_y, vnodes=vnodes, files={f.path: f for f in files},
            feature_extractor=fe, client=client, vnode_parents=vnode_parents,
            node_parents=node_parents, log=log)
        bad_preds = set(range(winners.shape[0])) - set(safe_preds)
        self.assertEqual(bad_preds, {15})


if __name__ == "__main__":
    unittest.main()
