import logging
import os
from pathlib import Path
import tarfile
import tempfile
import unittest

from lookout.style.format.feature_extractor import FeatureExtractor
from lookout.style.format.model import FormatModel
from lookout.style.format.postprocess import filter_uast_breaking_preds
from lookout.style.format.tests.test_analyzer import FakeDataService
from lookout.style.format.utils import prepare_files


class PostprocessingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.language = "javascript"
        cls.data_service = FakeDataService(files=None, changes=None)

        cls.parent_loc = Path(__file__).parent.resolve()
        cls.base_dir_ = tempfile.TemporaryDirectory(dir=str(cls.parent_loc))
        cls.base_dir = cls.base_dir_.name

        with tarfile.open(str(cls.parent_loc / "jquery_noisy.tar.xz")) as tar:
            tar.extractall(path=cls.base_dir)
        cls.jquery_noisy_dir = os.path.join(cls.base_dir, "jquery_noisy")
        cls.input_file = os.path.join(cls.jquery_noisy_dir, "28-xhr.js")
        # model trained on https://github.com/jquery/jquery repo with the following
        # non default parameters: left_siblings_window: 4, right_siblings_window: 4,
        # parents_depth: 1, left_features: ["length", "diff_offset", "diff_col",
        # "diff_line", "label", "reserved", "roles"], right_features: ["length",
        # "reserved", "roles"], parent_features: ["roles"]
        cls.model_path = str(Path(__file__).parent.resolve() / "model_jquery.asdf")

    @classmethod
    def tearDownClass(cls):
        cls.base_dir_.cleanup()

    def test_posprocess(self):
        log = logging.getLogger("postprocess")
        model = FormatModel().load(self.model_path)
        rules = model[self.language]
        files = prepare_files([self.input_file], self.data_service.bblfsh_client, self.language)
        fe = FeatureExtractor(language=self.language, **rules.origin_config["feature_extractor"])
        res = fe.extract_features(files)
        X, y, (vnodes_y, vnodes, vnode_parents, node_parents) = res
        y_pred, winners = rules.predict(X=X, vnodes_y=vnodes_y, vnodes=vnodes,
                                        feature_extractor=fe)
        y, y_pred, vnodes_y, safe_preds = filter_uast_breaking_preds(
            y=y, y_pred=y_pred, vnodes_y=vnodes_y, vnodes=vnodes, files={f.path: f for f in files},
            feature_extractor=fe, stub=self.data_service.get_bblfsh(), vnode_parents=vnode_parents,
            node_parents=node_parents, log=log)
        bad_preds = set(range(winners.shape[0])) - set(safe_preds)
        # On this file, the model makes exactly one prediction that is breaking the uast: X[15]
        self.assertEqual(bad_preds, {15})


if __name__ == "__main__":
    unittest.main()
