from collections import defaultdict, ChainMap
import logging
from pprint import pformat
import threading
from typing import Any, Dict, Iterable, List, Mapping

from skopt import BayesSearchCV
from skopt.space import Categorical, Integer

from lookout.core import slogging
from lookout.core.analyzer import Analyzer, ReferencePointer
from lookout.core.api.service_analyzer_pb2 import Comment
from lookout.core.api.service_data_pb2 import File
from lookout.core.api.service_data_pb2_grpc import DataStub
from lookout.core.data_requests import with_changed_uasts_and_contents, with_uasts_and_contents
from lookout.style.format.diff import find_new_lines
from lookout.style.format.features import FeatureExtractor
from lookout.style.format.model import FormatModel
from lookout.style.format.rules import TrainableRules


class FormatAnalyzer(Analyzer):
    log = logging.getLogger("FormatAnalyzer")
    model_type = FormatModel
    name = "style.format.analyzer.FormatAnalyzer"
    version = "1"
    description = "Source code formatting: whitespace, new lines, quotes, braces."

    def __init__(self, model: FormatModel, url: str, config: Mapping[str, Any]) -> None:
        super().__init__(model, url, config)
        self.config = self._load_analyze_config(self.config)

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
        base_files_by_lang = self._files_by_language(c.base for c in changes)
        head_files_by_lang = self._files_by_language(c.head for c in changes)
        for lang, head_files in head_files_by_lang.items():
            if lang not in self.model:
                self.log.warning("skipped %d written in %s. Rules for %s do not exist in model",
                                 len(head_files), lang, lang)
                continue
            rules = self.model[lang]
            for file in self._filter_files(head_files,
                                           rules.origin_config["line_length_limit"]):
                try:
                    prev_file = base_files_by_lang[lang][file.path]
                except KeyError:
                    lines = None
                else:
                    lines = [find_new_lines(prev_file, file)]
                fe = FeatureExtractor(language=lang, **rules.origin_config["feature_extractor"])
                res = fe.extract_features([file], lines)
                if res is None:
                    comment = Comment()
                    comment.file = file.path
                    comment.confidence = 100
                    comment.line = 1
                    comment.text = "Failed to parse this file"
                    continue
                X, y, vnodes = res
                X, _ = fe.select_features(X, y)
                self.log.debug("predicting values for %d samples", len(y))
                y_pred, winners = rules.predict(X, True)
                assert len(y) == len(y_pred)

                for yi, y_predi, vnode, winner in zip(y, y_pred, vnodes, winners):
                    if yi != y_predi:
                        comment = vnode.to_comment(y_predi)
                        comment.file = file.path
                        comment.confidence = int(round(rules.rules[winner].stats.conf * 100))
                        comments.append(comment)
        return comments

    @classmethod
    @with_uasts_and_contents
    def train(cls, ptr: ReferencePointer, config: Mapping[str, Any], data_request_stub: DataStub,
              **data) -> FormatModel:
        """
        Train a model given the files available.

        :param ptr: Git repository state pointer.
        :param config: configuration dict.
        :param data: contains "files" - the list of files in the pointed state.
        :param data_request_stub: connection to the Lookout data retrieval service, not used.
        :return: AnalyzerModel containing the learned rules, per language.
        """
        cls.log.info("train %s %s %s", ptr.url, ptr.commit,
                     pformat(config, width=4096, compact=True))
        files_by_language = cls._files_by_language(data["files"])
        model = FormatModel().construct(cls, ptr)
        config = cls._load_train_config(config)
        for language, files in files_by_language.items():
            lang_config = dict(ChainMap(config.get(language, {}), config["global"]))
            files = list(cls._filter_files(files, lang_config["line_length_limit"]))
            if len(files) == 0:
                cls.log.info("Zero files after filtering, %s language is skipped.", language)
                continue
            try:
                fe = FeatureExtractor(language=language, **lang_config["feature_extractor"])
            except ImportError:
                cls.log.warning("skipped %d %s files - not supported", len(files), language)
                continue
            else:
                cls.log.info("training on %d %s files", len(files), language)
            # we sort to make the features reproducible
            X, y, _ = fe.extract_features(sorted(files, key=lambda x: x.path))
            X, selected_features = fe.select_features(X, y)
            lang_config["feature_extractor"]["selected_features"] = selected_features
            lower_bound_instances = lang_config["lower_bound_instances"]
            if X.shape[0] < lower_bound_instances:
                cls.log.warning("skipped %d %s files: too few samples (%d/%d)",
                                len(files), language, X.shape[0], lower_bound_instances)
                continue
            cls.log.debug("training the rules model")
            if not slogging.logs_are_structured:
                # workaround the check in joblib - everything still works without it
                threading._MainThread = threading.Thread
            bscv = BayesSearchCV(
                TrainableRules(**lang_config["trainable_rules"], origin_config=lang_config),
                {"base_model_name": Categorical(["sklearn.ensemble.RandomForestClassifier",
                                                 "sklearn.tree.DecisionTreeClassifier"]),
                 "max_depth": Categorical([None, 5, 10]),
                 "max_features": Categorical([None, "auto"]),
                 "min_samples_split": Integer(2, 20),
                 "min_samples_leaf": Integer(1, 20)},
                n_jobs=lang_config["n_jobs"],
                n_iter=lang_config["n_iter"],
                random_state=lang_config["trainable_rules"]["random_state"])
            if not slogging.logs_are_structured:
                # fool the check in joblib - everything still works without it
                # this trick allows to run parallel bscv.fit()
                from unittest.mock import patch
                with patch("threading._MainThread", threading.Thread):
                    cls.log.debug("patched joblib")
                    bscv.fit(X, y)
            else:
                bscv.fit(X, y)
            cls.log.debug("score of the best estimator found: %.6f", bscv.best_score_)
            cls.log.debug("params of the best estimator found: %s", str(bscv.best_params_))
            cls.log.debug("training the model with complete data")
            lang_config["trainable_rules"].update(bscv.best_params_)
            trainable_rules = TrainableRules(**lang_config["trainable_rules"],
                                             origin_config=lang_config)
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
    def _load_analyze_config(cls, config: Mapping[str, Any]) -> Mapping[str, Any]:
        """
        Merges config for analyze call with default config values stored inside this function.

        :param config: User-defined config.
        :return: Full config.
        """
        final_config = {
            "debug": False,
        }
        FormatAnalyzer.recursive_update(final_config, config)
        return final_config

    @classmethod
    def _load_train_config(cls, config: Mapping[str, Any]) -> Mapping[str, Any]:
        final_config = {
            "global": {
                "feature_extractor": {
                    "siblings_window": 5,
                    "parents_depth": 2,
                    "debug_parsing": False,
                    "select_features_number": 500,
                    "remove_empty_features": True,
                },
                "trainable_rules": {
                    "prune_branches_algorithms": ["reduced-error"],
                    "top_down_greedy_budget": [False, .5],
                    "prune_attributes": False,
                    "uncertain_attributes": True,
                    "prune_dataset_ratio": .2,
                    "n_estimators": 10,
                    "random_state": 42,
                },
                "n_jobs": -1,
                "n_iter": 5,
                "line_length_limit": 500,
                "lower_bound_instances": 500,
            },
            "javascript": {
                # The same structure here
            },
        }
        FormatAnalyzer.recursive_update(final_config, config)
        return final_config

    @classmethod
    def _filter_files(cls, files: Dict[str, File], line_length_limit: int
                      ) -> Iterable[File]:
        """
        Filter files based on their maximum line length.

        :param files: Files to filter.
        :param line_length_limit: Maximum line length to accept a file.
        :return: Files filtered.
        """
        excluded = total = 0
        for filename, file in files.items():
            if len(max(file.content.splitlines(), key=len, default=b"")) <= line_length_limit:
                total += 1
                yield file
            else:
                excluded += 1
        if excluded > 0:
            cls.log.debug("excluded %d/%d files by max line length %d",
                          excluded, total, line_length_limit)

    @staticmethod
    def recursive_update(mapping, other):
        for key, value in other.items():
            if isinstance(value, dict):
                FormatAnalyzer.recursive_update(mapping.setdefault(key, {}), value)
            else:
                mapping[key] = value
