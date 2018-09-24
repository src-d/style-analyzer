from collections import defaultdict
import logging
from pprint import pformat
import threading
from typing import Any, Dict, Iterable, List, Mapping

from skopt import BayesSearchCV
from skopt.space import Categorical, Integer

from lookout.core import slogging
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
    name = "style.format.analyzer.FormatAnalyzer"
    version = "1"
    description = "Source code formatting: whitespace, new lines, quotes, braces."

    def __init__(self, model: AnalyzerModel, url: str, config: Mapping[str, Any]) -> None:
        super().__init__(model, url, config)
        self.config = self._load_config(self.config)

    @with_changed_uasts_and_contents
    def analyze(self, ptr_from: ReferencePointer, ptr_to: ReferencePointer,
                data_request_stub: DataStub, **data) -> List[Comment]:
        """
        Analyze a set of changes from one revision to another.

        :param ptr_from: Git repository state pointer to the base revision.
        :param ptr_to: Git repository state pointer to the head revision.
        :param data_request_stub: Connection to the Lookout data retrieval service, not used.
        :param data: Contains "changes" - the list of changes in the pointed state.
        :return: List of comments.
        """
        comments = []
        changes = list(data["changes"])
        base_files = self._files_by_language(self._filter_files((c.base for c in changes),
                                                                self.config["line_length_limit"]))
        head_files = self._files_by_language(self._filter_files((c.head for c in changes),
                                                                self.config["line_length_limit"]))
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
                    lines = [find_new_lines(prev_file, file)]
                X, y, vnodes = FeatureExtractor(language=lang,
                                                siblings_window=self.config["siblings_window"],
                                                parents_depth=self.config["parents_depth"],
                                                debug_parsing=self.config["debug_parsing"]) \
                    .extract_features([file], lines)
                self.log.debug("predicting values for %d samples", len(y))
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
    def train(cls, ptr: ReferencePointer, config: Mapping[str, Any], data_request_stub: DataStub,
              **data) -> AnalyzerModel:
        """
        Train a model given the files available.

        :param ptr: Git repository state pointer.
        :param config: configuration dict.
        :param data: contains "files" - the list of files in the pointed state.
        :param data_request_stub: connection to the Lookout data retrieval service, not used.
        :return: AnalyzerModel containing the learned rules, per language.
        """
        config = cls._load_config(config)
        cls.log.info("train %s %s %s", ptr.url, ptr.commit,
                     pformat(config, width=4096, compact=True))
        files_by_language = cls._files_by_language(
            cls._filter_files(data["files"], config["line_length_limit"]))
        model = FormatModel().construct(cls, ptr)
        trainable_rules_config = {
            "prune_branches_algorithms": config["prune_branches_algorithms"],
            "prune_attributes": config["prune_attributes"],
            "top_down_greedy_budget": config["top_down_greedy_budget"],
            "uncertain_attributes": config["uncertain_attributes"],
            "prune_dataset_ratio": config["prune_dataset_ratio"],
            "n_estimators": config["n_estimators"],
            "random_state": config["random_state"],
        }
        for language, files in files_by_language.items():
            language = language.lower()
            try:
                fe = FeatureExtractor(language=language,
                                      siblings_window=config["siblings_window"],
                                      parents_depth=config["parents_depth"],
                                      debug_parsing=config["debug_parsing"])
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
            if not slogging.logs_are_structured:
                # workaround the check in joblib - everything still works without it
                threading._MainThread = threading.Thread
            bscv = BayesSearchCV(
                TrainableRules(**trainable_rules_config),
                {"base_model_name": Categorical(["sklearn.ensemble.RandomForestClassifier",
                                                 "sklearn.tree.DecisionTreeClassifier"]),
                 "max_depth": Categorical([None, 5, 10]),
                 "max_features": Categorical([None, "auto"]),
                 "min_samples_split": Integer(2, 20),
                 "min_samples_leaf": Integer(1, 20)},
                n_jobs=-1,
                n_iter=config["n_iter"],
                random_state=config["random_state"])
            if not slogging.logs_are_structured:
                # fool the check in joblib - everything still works without it
                # this trick allows to run parallel bscv.fit()
                from unittest.mock import patch
                with patch("threading._MainThread", threading.Thread):
                    cls.log.debug("patched joblib")
                    bscv.fit(X, y)
            else:
                bscv.fit(X, y)
            cls.log.debug("score of the best estimator found: %.3f", bscv.best_score_)
            cls.log.debug("params of the best estimator found: %s", str(bscv.best_params_))
            cls.log.debug("training the model with complete data")
            trainable_rules_config.update(bscv.best_params_)
            trainable_rules = TrainableRules(**trainable_rules_config)
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
    def _load_config(cls, config: Mapping[str, Any]):
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
            "n_iter": 5,
            "debug_parsing": False,
            "line_length_limit": 500
        }
        final_config.update(config)
        # the incoming value can be a list from ASDF
        final_config["top_down_greedy_budget"] = TopDownGreedyBudget(
            *final_config["top_down_greedy_budget"])
        return final_config

    @classmethod
    def _filter_files(cls, files: Iterable[File], line_length_limit: int) -> Iterable[File]:
        """
        Filter files based on their maximum line length.

        :param files: Files to filter.
        :param line_length_limit: Maximum line length to accept a file.
        :return: Files filtered.
        """
        excluded = total = 0
        for file in files:
            if len(max(file.content.splitlines(), key=len, default=b"")) <= line_length_limit:
                total += 1
                yield file
            else:
                excluded += 1
        if excluded > 0:
            cls.log.debug("excluded %d/%d files by max line length %d",
                          excluded, total, line_length_limit)
