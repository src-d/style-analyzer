"""Analyzer that detects bad formatting by learning on the existing code in the repository."""
from itertools import chain
import logging
from pprint import pformat
import threading
from typing import Any, List, Mapping

from bblfsh.client import BblfshClient
from lookout.core import slogging
from lookout.core.analyzer import Analyzer, ReferencePointer
from lookout.core.api.service_analyzer_pb2 import Comment
from lookout.core.api.service_data_pb2_grpc import DataStub
from lookout.core.data_requests import with_changed_uasts_and_contents, with_uasts_and_contents
from lookout.core.lib import files_by_language, filter_files, find_deleted_lines, find_new_lines
import numpy
from skopt import BayesSearchCV
from skopt.space import Categorical, Integer

from lookout.style.format.descriptions import get_code_chunk, get_error_description, \
    rule_to_comment
from lookout.style.format.feature_extractor import FeatureExtractor
from lookout.style.format.model import FormatModel
from lookout.style.format.postprocess import filter_uast_breaking_preds
from lookout.style.format.rules import TrainableRules
from lookout.style.format.utils import merge_dicts


class FormatAnalyzer(Analyzer):
    """Detect bad formatting by training on existing code and analyzing pull requests."""

    _log = logging.getLogger("FormatAnalyzer")
    model_type = FormatModel
    name = "style.format.analyzer.FormatAnalyzer"
    version = "1"
    description = "Source code formatting: whitespace, new lines, quotes, braces."
    defaults_for_analyze = {
        "bblfsh_address": "0.0.0.0:9432",
        "report_code_lines": True,
        "report_triggered_rules": True,
        "report_parse_failures": True,
    }
    defaults_for_train = {
        "global": {
            "feature_extractor": {
                "left_siblings_window": 5,
                "right_siblings_window": 5,
                "parents_depth": 2,
                "node_features": ["start_line", "start_col"],
                "left_features": ["length", "diff_offset", "diff_col", "diff_line",
                                  "internal_type", "label", "reserved", "roles"],
                "right_features": ["length", "internal_type", "reserved", "roles"],
                "parent_features": ["internal_type", "roles"],
                "no_labels_on_right": True,
                "debug_parsing": False,
                "select_features_number": 500,
                "remove_constant_features": True,
                "insert_noops": False,
                "return_sibling_indices": False,
                "cutoff_label_support": 50,
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
            "cutoff_label_precision": 0.95,
        },
        # selected settings for each particular language which overwrite "global"
        # empty {} is still required if we do not have any adjustments
        "javascript": {},
    }

    def __init__(self, model: FormatModel, url: str, config: Mapping[str, Any]) -> None:
        """
        Construct a FormatAnalyzer.

        :param model: FormatModel to use during pull request analysis.
        :param url: Git repository on which the model was trained.
        :param config: Configuration to use to analyze pull requests.
        """
        super().__init__(model, url, config)
        self.config = self._load_analyze_config(self.config)
        self.client = BblfshClient(self.config["bblfsh_address"])

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
        def group_line_nodes(y, y_pred, vnodes_y, rule_winners):
            line_nodes = []
            generate_comment = False
            for yi, y_predi, vnode_y, winner in zip(y, y_pred, vnodes_y, rule_winners):
                if not line_nodes or vnode_y.start.line == line_nodes[0][2].start.line:
                    # collect all nodes on the same line
                    line_nodes.append((yi, y_predi, vnode_y, winner))
                    if yi != y_predi:
                        generate_comment = True
                    continue
                else:
                    if generate_comment:
                        yield line_nodes
                    generate_comment = yi != y_predi
                    line_nodes = [(yi, y_predi, vnode_y, winner)]

        log = self._log
        comments = []
        changes = list(data["changes"])
        base_files_by_lang = files_by_language(c.base for c in changes)
        head_files_by_lang = files_by_language(c.head for c in changes)
        for lang, head_files in head_files_by_lang.items():
            if lang not in self.model:
                log.warning("skipped %d written in %s. Rules for %s do not exist in model",
                            len(head_files), lang, lang)
                continue
            rules = self.model[lang]
            for file in filter_files(head_files, rules.origin_config["line_length_limit"], log):
                log.debug("Analyze %s file", file.path)
                try:
                    prev_file = base_files_by_lang[lang][file.path]
                except KeyError:
                    lines = None
                else:
                    lines = sorted(chain.from_iterable((
                        find_new_lines(prev_file, file),
                        find_deleted_lines(prev_file, file),
                    )))
                fe = FeatureExtractor(language=lang, **rules.origin_config["feature_extractor"])
                res = fe.extract_features([file], [lines])
                if res is None:
                    if self.config["report_parse_failures"]:
                        comment = Comment()
                        comment.file = file.path
                        comment.confidence = 100
                        comment.line = 1
                        comment.text = "Failed to parse this file"
                        comment.append(comment)
                    log.warning("Failed to parse %s", file.path)
                    continue
                X, y, (vnodes_y, vnodes, vnode_parents, node_parents) = res
                y_pred, rule_winners = rules.predict(X=X, vnodes_y=vnodes_y, vnodes=vnodes,
                                                     feature_extractor=fe)
                y, y_pred, vnodes_y, safe_preds = filter_uast_breaking_preds(
                    y=y, y_pred=y_pred, vnodes_y=vnodes_y, files={file.path: file},
                    feature_extractor=fe, client=self.client, vnode_parents=vnode_parents,
                    node_parents=node_parents, log=log)
                assert len(y) == len(y_pred)

                code_lines = file.content.decode("utf-8", "replace").splitlines()
                for line_nodes in group_line_nodes(y, y_pred, vnodes_y, rule_winners):
                    code_line_number = line_nodes[0][2].start.line  # 1-based
                    if self.config["report_triggered_rules"]:
                        code_text = ""
                        if self.config["report_code_lines"]:
                            code_text = get_code_chunk(lang, code_lines, code_line_number)
                        vnodes_comments = [
                            "%s\n%s" % (get_error_description(vnode, y_predi, fe),
                                        rule_to_comment(rules.rules[winner], fe, winner))
                            for yi, y_predi, vnode, winner in line_nodes if yi != y_predi]
                        text = "format: style mismatch:\n%s%s\n" % (
                            code_text, "\n\n".join(vnodes_comments))
                    else:
                        vnodes_comments = [
                            get_error_description(vnode, y_predi, fe)
                            for yi, y_predi, vnode, winner in line_nodes if yi != y_predi]
                        text = "format: style mismatch:\n%s\n" % ("\n\n".join(vnodes_comments))

                    confidence = 0
                    confidence_count = 0
                    for yi, y_predi, _, winner in line_nodes:
                        if yi != y_predi:
                            confidence += rules.rules[winner].stats.conf
                            confidence_count += 1

                    comment = Comment()
                    comment.line = code_line_number
                    comment.text = text
                    comment.file = file.path
                    comment.confidence = int(round(confidence * 100 / confidence_count))
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
        cls._log.info("train %s %s %s", ptr.url, ptr.commit,
                      pformat(config, width=4096, compact=True))
        model = FormatModel().construct(cls, ptr)
        config = cls._load_train_config(config)
        for language, files in files_by_language(data["files"]).items():
            try:
                lang_config = config[language]
            except KeyError:
                cls._log.warning("Language %s is not supported, skipped", language)
                continue
            files = filter_files(files, lang_config["line_length_limit"], cls._log)
            if len(files) == 0:
                cls._log.info("Zero files after filtering, language %s is skipped.", language)
                continue
            try:
                fe = FeatureExtractor(language=language, **lang_config["feature_extractor"])
            except ImportError:
                cls._log.warning("skipped %d %s files - not supported", len(files), language)
                continue
            else:
                cls._log.info("training on %d %s files", len(files), language)
            # we sort to make the features reproducible
            X, y, _ = fe.extract_features(sorted(files, key=lambda x: x.path))
            X, selected_features = fe.select_features(X, y)
            lang_config["feature_extractor"]["selected_features"] = selected_features
            lang_config["feature_extractor"]["label_composites"] = fe.labels_to_class_sequences
            lower_bound_instances = lang_config["lower_bound_instances"]
            if X.shape[0] < lower_bound_instances:
                cls._log.warning("skipped %d %s files: too few samples (%d/%d)",
                                 len(files), language, X.shape[0], lower_bound_instances)
                continue
            cls._log.debug("training the rules model")
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
                    cls._log.debug("patched joblib")
                    bscv.fit(X, y)
            else:
                bscv.fit(X, y)
            cls._log.debug("score of the best estimator found: %.6f", bscv.best_score_)
            cls._log.debug("params of the best estimator found: %s", str(bscv.best_params_))
            cls._log.debug("training the model with complete data")
            lang_config["trainable_rules"].update(bscv.best_params_)
            trainable_rules = TrainableRules(**lang_config["trainable_rules"],
                                             origin_config=lang_config)
            trainable_rules.fit(X, y)
            importances = trainable_rules.feature_importances_
            cls._log.debug(
                "Feature importances from %s:\n\t%s",
                lang_config["trainable_rules"]["base_model_name"],
                "\n\t".join("%-55s %.5E" % (fe.feature_names[i], importances[i])
                            for i in numpy.argsort(-importances)[:25] if importances[i] > 1e-5))
            # throw away imprecise classes
            scores = trainable_rules.full_score(X, y)
            cutoff_precision = lang_config["cutoff_label_precision"]
            erased_labels = [label for label, score in scores.items()
                             if score.precision < cutoff_precision]
            if len(erased_labels) > 0:
                cls._log.debug("Removed %d/%d labels by precision %f", len(erased_labels),
                               len(fe.labels_to_class_sequences), cutoff_precision)
            trainable_rules.erase_labels(erased_labels)
            model[language] = trainable_rules.rules
        cls._log.info("trained %s", model)
        return model

    @classmethod
    def _load_analyze_config(cls, config: Mapping[str, Any]) -> Mapping[str, Any]:
        """
        Merge config for `analyze()` with the default config.

        :param config: User-defined config.
        :return: Full config.
        """
        return merge_dicts(cls.defaults_for_analyze, config)

    @classmethod
    def _load_train_config(cls, config: Mapping[str, Any]) -> Mapping[str, Any]:
        """
        Merge config for `train()` with the default config.

        :param config: User-defined config.
        :return: Full config.
        """
        config = merge_dicts(cls.defaults_for_train, config)
        global_config = config.pop("global")
        try:
            return {lang: merge_dicts(global_config, lang_config)
                    for lang, lang_config in config.items()}
        except AttributeError as e:
            raise ValueError("Config %s can not be merged with default values config: %s" % (
                config, global_config
            ))
