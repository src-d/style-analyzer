from collections import defaultdict
import logging
from pprint import pformat
from typing import Dict, Iterable, Any

from skopt import BayesSearchCV
from skopt.space import Categorical, Integer

from lookout.core.analyzer import Analyzer, AnalyzerModel, ReferencePointer
from lookout.core.api.service_analyzer_pb2 import Comment
from lookout.core.api.service_data_pb2 import File
from lookout.core.api.service_data_pb2_grpc import DataStub
from lookout.core.data_requests import with_changed_uasts_and_contents, with_uasts_and_contents
from lookout.style.format.diff import find_new_lines
from lookout.style.format.features import FeatureExtractor
from lookout.style.format.model import FormatModel
from lookout.style.format.rules import TopDownGreedyBudget, TrainableRules


class FormatAnalyzer(Analyzer):
    log = logging.getLogger("FormatAnalyzer")
    model_type = FormatModel
    version = "1"
    description = "Source code formatting: whitespace, new lines, quotes, braces."

    @with_changed_uasts_and_contents
    def analyze(self, ptr_from: ReferencePointer, ptr_to: ReferencePointer,
                data_request_stub: DataStub, **data) -> [Comment]:
        comments = []
        changes = list(data["changes"])
        base_files = self._files_by_language(c.base for c in changes)
        head_files = self._files_by_language(c.head for c in changes)
        for lang, lang_head_files in head_files.items():
            try:
                rules = self.model[lang]
            except KeyError:
                self.log.warning("skipped %d written in %s - model does not exist",
                                 len(lang_head_files), lang)
                continue
            for path, file in lang_head_files.items():
                try:
                    prev_file = base_files[lang][path]
                except KeyError:
                    lines = None
                else:
                    lines = find_new_lines(prev_file, file)
                X, y, vnodes = FeatureExtractor(lang, **getattr(self, "extractor", {})) \
                    .extract_features([file], [lines])
                y_pred, winners = rules.predict(X, True)
                assert len(y) == len(y_pred)
                for yi, y_predi, vnode, winner in zip(y, y_pred, vnodes, winners):
                    if yi != y_predi:
                        comment = vnode.to_comment(y_predi)
                        comment.file = path
                        comment.confidence = int(round(rules.rules[winner].stats.conf * 100))
                        comments.append(comment)
        return comments

    @classmethod
    @with_uasts_and_contents
    def train(cls, ptr: ReferencePointer, config: Dict[str, Any], data_request_stub: DataStub,
              **data) -> AnalyzerModel:
        """
        Train a model given the files available.

        :param ptr: Git repository state pointer.
        :param config: configuration dict.
        :param data: contains "files" - the list of files in the pointed state.
        :param data_request_stub: connection to the Lookout data retrieval service, not used.
        :return: AnalyzerModel containing the learned rules, per language.
        """
        config = cls._load_train_config(config)
        cls.log.info("train %s %s %s", ptr.url, ptr.commit,
                     pformat(config, width=4096, compact=True))
        files_by_language = cls._files_by_language(data["files"])
        model = FormatModel().construct(cls, ptr)
        for language, files in files_by_language.items():
            language = language.lower()
            try:
                fe = FeatureExtractor(language=language,
                                      siblings_window=config["siblings_window"],
                                      parents_depth=config["parents_depth"])
            except ImportError:
                cls.log.warning("skipped %d %s files - not supported", len(files), language)
                continue
            else:
                cls.log.info("training on %d %s files", len(files), language)
            # we sort to make the features reproducible
            X, y, _ = fe.extract_features(f[1] for f in sorted(files.items()))
            lower_bound_instances = config["lower_bound_instances"]
            if X.shape[0] < lower_bound_instances:
                cls.log.warning("skipped %d %s files: too few samples (%d/%d)",
                                len(files), language, X.shape[0], lower_bound_instances)
                continue
            cls.log.debug("training the rules model")
            bscv = BayesSearchCV(
                TrainableRules(
                    prune_branches_algorithms=config["prune_branches_algorithms"],
                    prune_attributes=config["prune_attributes"],
                    top_down_greedy_budget=config["top_down_greedy_budget"],
                    uncertain_attributes=config["uncertain_attributes"],
                    prune_dataset_ratio=config["prune_dataset_ratio"],
                    n_estimators=config["n_estimators"],
                    random_state=config["random_state"]),
                {"base_model_name": Categorical(["sklearn.ensemble.RandomForestClassifier",
                                                 "sklearn.tree.DecisionTreeClassifier"]),
                 "max_depth": Categorical([None, 5, 10]),
                 "max_features": Categorical([None, "auto"]),
                 "min_samples_split": Integer(2, 20),
                 "min_samples_leaf": Integer(1, 20)},
                n_jobs=-1,
                n_iter=config["n_iter"],
                random_state=config["random_state"])
            bscv.fit(X, y)
            cls.log.debug("score of the best estimator found: %.3f", bscv.best_score_)
            cls.log.debug("params of the best estimator found: %s", str(bscv.best_params_))
            cls.log.debug("training the model with complete data")
            trainable_rules = TrainableRules(prune_branches_algorithms=["reduced-error"],
                                             prune_attributes=True, random_state=42,
                                             uncertain_attributes=True, **bscv.best_params_)
            trainable_rules.fit(X, y)
            model[language] = trainable_rules.rules
        cls.log.info("trained %s", model)
        return model

    @classmethod
    def _files_by_language(cls, files: Iterable[File]) -> Dict[str, Dict[str, File]]:
        """
        Sorts files by programming language and path.
        :param files: iterable of `File`-s.
        :return: dictionary with languages as keys and files mapped to paths as values.
        """
        result = defaultdict(dict)
        for file in files:
            if not len(file.uast.children):
                continue
            result[file.language.lower()][file.path] = file
        return result

    @classmethod
    def _load_train_config(cls, config: Dict[str, Any]):
        final_config = {
            "siblings_window": 5,
            "parents_depth": 2,
            "lower_bound_instances": 500,
            "prune_branches_algorithms": ["reduced-error"],
            "top_down_greedy_budget": [False, .5],
            "prune_attributes": False,
            "uncertain_attributes": True,
            "prune_dataset_ratio": .2,
            "n_estimators": 10,
            "random_state": 42,
            "n_iter": 5
        }
        final_config.update(config)
        # the incoming value can be a list from ASDF
        final_config["top_down_greedy_budget"] = TopDownGreedyBudget(
            *final_config["top_down_greedy_budget"])
        return final_config
