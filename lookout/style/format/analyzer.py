"""Analyzer that detects bad formatting by learning on the existing code in the repository."""
from collections import defaultdict
import functools
from itertools import chain
import logging
import os
from pprint import pformat
import random
from typing import Any, Iterable, List, Mapping, NamedTuple, Sequence, Tuple

import bblfsh  # noqa: F401
from jinja2 import Template
from lookout.core.analyzer import Analyzer, ReferencePointer
from lookout.core.api.service_analyzer_pb2 import Comment
from lookout.core.api.service_data_pb2 import Change, File
from lookout.core.data_requests import DataService, \
    with_changed_uasts_and_contents, with_uasts_and_contents
from lookout.core.lib import files_by_language, filter_files, find_deleted_lines, find_new_lines
from lookout.core.metrics import submit_event
import numpy

from lookout.style.format.classes import CLASS_INDEX, CLS_NEWLINE
from lookout.style.format.code_generator import CodeGenerator
from lookout.style.format.descriptions import describe_rule, get_change_description, hash_rule
from lookout.style.format.feature_extractor import FeatureExtractor
from lookout.style.format.model import FormatModel
from lookout.style.format.optimizer import Optimizer
from lookout.style.format.postprocess import filter_uast_breaking_preds
from lookout.style.format.rules import Rules, TrainableRules
from lookout.style.format.utils import generate_comment, merge_dicts
from lookout.style.format.virtual_node import VirtualNode

LineFix = NamedTuple("LineFix", (
    ("line_number", int),                     # line number for the comment
    ("suggested_code", str),                  # code line predicted by our model
    ("fixed_vnodes", List[VirtualNode]),      # VirtualNode-s with fixed y
    ("confidence", int),                      # overall confidence in the prediction, 0-100
))

FileFix = NamedTuple("FileFix", (
    ("error", str),                           # error message
    ("line_fixes", Sequence[LineFix]),        # fixes for file lines. Can be empty.
    ("language", str),                        # programming language of the code
    ("feature_extractor", FeatureExtractor),  # FeatureExtractor involved
    ("file_vnodes", List[VirtualNode]),       # fixed VirtualNode-s which correspond to this file
    ("head_file", File),                      # file from head revision
    ("base_file", File),                      # file from base revision
    ("y_pred_pure", numpy.ndarray),           # raw rules.predict() output. It can contain
                                              # negative values for a prediction refusal.
    ("y", numpy.ndarray),                     # final vector of predictions with all fixes applied
))


class FormatAnalyzer(Analyzer):
    """Detect bad formatting by training on existing code and analyzing pull requests."""

    model_type = FormatModel
    name = "style.format.analyzer.FormatAnalyzer"
    version = 1
    description = "Source code formatting: whitespace, new lines, quotes, braces."
    defaults_for_analyze = {
        "confidence_threshold": 0.92,
        "support_threshold": 80,
        "report_code_lines": False,
        "report_triggered_rules": False,
        "report_parse_failures": False,
        "uast_break_check": True,
        "comment_template": os.path.join(os.path.dirname(__file__), "templates",
                                         "comment.jinja2"),
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
                "return_sibling_indices": False,
                "cutoff_label_support": 80,
            },
            "trainable_rules": {
                "prune_branches_algorithms": ["reduced-error"],
                "top_down_greedy_budget": [False, .5],
                "prune_attributes": True,
                "uncertain_attributes": True,
                "confidence_threshold": 0.8,
                "prune_dataset_ratio": .2,
                "n_estimators": 10,
            },
            "optimizer": {
                "n_iter": 10,
                "cv": 3,
                "n_jobs": -1,
                "base_model_name_categories": ["sklearn.ensemble.RandomForestClassifier",
                                               "sklearn.tree.DecisionTreeClassifier"],
                "max_depth_categories": [None, 5, 10],
                "max_features_categories": [None, "auto"],
                "min_samples_leaf_min": 90,
                "min_samples_leaf_max": 120,
                "min_samples_split_min": 180,
                "min_samples_split_max": 240,
            },
            "random_state": 42,
            "test_dataset_ratio": 0.0,
            "line_length_limit": 500,
            "lower_bound_instances": 500,
            "overall_size_limit": 2 << 20,  # 2 MB
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
        self._log = logging.getLogger(type(self).__name__)
        self.config = self._load_analyze_config(self.config)
        with open(self.config["comment_template"], encoding="utf-8") as f:
            self.comment_template = Template(f.read(), trim_blocks=True, lstrip_blocks=True)

    @with_changed_uasts_and_contents
    def analyze(self, ptr_from: ReferencePointer, ptr_to: ReferencePointer,
                data_service: DataService, **data) -> List[Comment]:
        """
        Analyze a set of changes from one revision to another.

        :param ptr_from: Git repository state pointer to the base revision.
        :param ptr_to: Git repository state pointer to the head revision.
        :param data_service: Connection to the Lookout data retrieval service.
        :param data: Contains "changes" - the list of changes in the pointed state.
        :return: List of comments.
        """
        comments = []
        for file_fix in self.generate_file_fixes(data_service, list(data["changes"])):
            filename = file_fix.head_file.path
            if file_fix.error:
                comments.append(generate_comment(text=file_fix.error, line=0, confidence=100,
                                                 filename=filename))

            else:
                comments.extend(generate_comment(
                    text=self.render_comment_text(file_fix, fix_index), filename=filename,
                    line=line_fix.line_number, confidence=line_fix.confidence)
                                for fix_index, line_fix in enumerate(file_fix.line_fixes))
        return comments

    @classmethod
    @with_uasts_and_contents
    def train(cls, ptr: ReferencePointer, config: Mapping[str, Any], data_service: DataService,
              **data) -> FormatModel:
        """
        Train a model given the files available.

        :param ptr: Git repository state pointer.
        :param config: configuration dict.
        :param data: contains "files" - the list of files in the pointed state.
        :param data_service: connection to the Lookout data retrieval service.
        :return: AnalyzerModel containing the learned rules, per language.
        """
        _log = logging.getLogger(cls.__name__)
        _log.info("train %s %s %s", ptr.url, ptr.commit, pformat(config, width=4096, compact=True))
        model = FormatModel().construct(cls, ptr)
        config = cls._load_train_config(config)
        for language, files in files_by_language(data["files"]).items():
            try:
                lang_config = config[language]
            except KeyError:
                _log.warning("language %s is not supported, skipped", language)
                continue
            random_state = lang_config["random_state"]
            files = filter_files(
                files, lang_config["line_length_limit"], lang_config["overall_size_limit"],
                random_state, _log)
            submit_event("%s.train.%s.files" % (cls.name, language), len(files))
            if len(files) == 0:
                _log.info("zero files after filtering, language %s is skipped.", language)
                continue
            try:
                fe = FeatureExtractor(language=language, **lang_config["feature_extractor"])
            except ImportError:
                _log.warning("skipped %d %s files - not supported", len(files), language)
                continue
            else:
                _log.info("training on %d %s files", len(files), language)
            train_files, test_files = FormatAnalyzer.get_train_test_split(
                files, lang_config["test_dataset_ratio"], random_state=random_state)
            # ensure that the features are reproducible
            train_files = sorted(train_files, key=lambda x: x.path)
            test_files = sorted(test_files, key=lambda x: x.path)
            X_train, y_train, _ = fe.extract_features(train_files)
            X_train, selected_features = fe.select_features(X_train, y_train)
            if test_files:
                X_test, y_test, _ = fe.extract_features(test_files)
                X_test, selected_features = fe.select_features(X_test, y_test)
            if lang_config["test_dataset_ratio"]:
                _log.debug("Real test ratio is %.3f",
                           X_test.shape[0] / (X_test.shape[0] + X_train.shape[0])
                           if test_files else 0)
            lang_config["feature_extractor"]["selected_features"] = selected_features
            lang_config["feature_extractor"]["label_composites"] = fe.labels_to_class_sequences
            lower_bound_instances = lang_config["lower_bound_instances"]
            if X_train.shape[0] < lower_bound_instances:
                _log.warning("skipped %d %s files: too few samples (%d/%d)",
                             len(files), language, X_train.shape[0], lower_bound_instances)
                continue
            _log.debug("training the rules model")
            optimizer = Optimizer(**lang_config["optimizer"], random_state=random_state)
            best_score, best_params = optimizer.optimize(X_train, y_train)
            _log.debug("score of the best estimator found: %.6f", best_score)
            _log.debug("params of the best estimator found: %s", str(best_params))
            _log.debug("training the model with complete data")
            lang_config["trainable_rules"].update(best_params)
            trainable_rules = TrainableRules(**lang_config["trainable_rules"],
                                             random_state=random_state,
                                             origin_config=lang_config)
            trainable_rules.fit(X_train, y_train)
            importances = trainable_rules.feature_importances_
            _log.debug(
                "feature importances from %s:\n\t%s",
                lang_config["trainable_rules"]["base_model_name"],
                "\n\t".join("%-55s %.5E" % (fe.feature_names[i], importances[i])
                            for i in numpy.argsort(-importances)[:25] if importances[i] > 1e-5))
            trainable_rules.rules.generate_classification_report(
                X_train, y_train, "train", fe.composite_class_representations)
            if test_files:
                trainable_rules.rules.generate_classification_report(
                    X_test, y_test, "test", fe.composite_class_representations)
            submit_event("%s.train.%s.rules" % (cls.name, language), len(trainable_rules.rules))
            if trainable_rules.rules.rules:
                model[language] = trainable_rules.rules
            else:
                _log.warning("model for %s has 0 rules. Skipping.", language)
        _log.info("trained %s", model)
        return model

    def generate_file_fixes(self, data_service: DataService, changes: Sequence[Change],
                            ) -> Iterable[FileFix]:
        """
        Generate all data required for any type of further processing.

        Next processing can be comment generation or performance report generation.

        :param data_service: Connection to the Lookout data retrieval service.
        :param changes: The list of changes in the pointed state.
        :return: Iterator with unrendered data per comment.
        """
        log = self._log
        base_files_by_lang = files_by_language(c.base for c in changes)
        head_files_by_lang = files_by_language(c.head for c in changes)
        processed_files_counter = defaultdict(int)
        processed_fixes_counter = defaultdict(int)
        for lang, head_files in head_files_by_lang.items():
            if lang not in self.model:
                log.warning("skipped %d written in %s. Rules for %s do not exist in model",
                            len(head_files), lang, lang)
                continue
            rules = self.model[lang]
            rules = rules.filter_by_confidence(self.config["confidence_threshold"]) \
                .filter_by_support(self.config["support_threshold"])
            for file in filter_files(head_files, rules.origin_config["line_length_limit"],
                                     rules.origin_config["overall_size_limit"], log=log):
                processed_files_counter[lang] += 1
                try:
                    prev_file = base_files_by_lang[lang][file.path]
                except KeyError:
                    prev_file = None
                    lines = None
                else:
                    lines = sorted(chain.from_iterable((
                        find_new_lines(prev_file, file),
                        find_deleted_lines(prev_file, file),
                    )))
                log.debug("%s %s", file.path, lines)
                fe = FeatureExtractor(language=lang, **rules.origin_config["feature_extractor"])
                feature_extractor_output = fe.extract_features([file], [lines])
                if feature_extractor_output is None:
                    submit_event("%s.analyze.%s.parse_failures" % (self.name, lang), 1)
                    if self.config["report_parse_failures"]:
                        log.warning("Failed to parse %s", file.path)
                        yield FileFix(error="Failed to parse", head_file=file, language=lang,
                                      feature_extractor=fe, base_file=prev_file, file_vnodes=[],
                                      line_fixes=[], y_pred_pure=None, y=None)
                else:
                    fixes, file_vnodes, y_pred_pure, y = self._generate_token_fixes(
                        file, fe, feature_extractor_output, data_service.get_bblfsh(), rules)
                    log.debug("%s %d fixes", file.path, len(fixes))
                    processed_fixes_counter[lang] += len(fixes)
                    yield FileFix(error="", head_file=file, language=lang, feature_extractor=fe,
                                  base_file=prev_file, file_vnodes=file_vnodes, line_fixes=fixes,
                                  y_pred_pure=y_pred_pure, y=y)
        for key, val in processed_files_counter.items():
            submit_event("%s.analyze.%s.files" % (self.name, key), val)
        for key, val in processed_fixes_counter.items():
            submit_event("%s.analyze.%s.fixes" % (self.name, key), val)

    def render_comment_text(self, file_fix: FileFix, fix_index: int) -> str:
        """
        Generate the text of the comment for the specified line fix.

        :param file_fix: Information about file fix required to render a comment text.
        :param fix_index: Index for `file_fix.line_fixes`. Comment is generated for this line fix.
        :return: string with the generated comment.
        """
        rules = self.model[file_fix.language]
        _describe_rule = functools.partial(
            describe_rule, feature_extractor=file_fix.feature_extractor)
        _hash_rule = functools.partial(
            hash_rule, feature_extractor=file_fix.feature_extractor)
        _describe_change = functools.partial(
            get_change_description, feature_extractor=file_fix.feature_extractor)
        code_lines = file_fix.head_file.content.decode("utf-8", "replace").splitlines()
        line_fix = file_fix.line_fixes[fix_index]
        return self.comment_template.render(
                config=self.config,                     # configuration of the analyzer
                language=file_fix.language,             # programming language of the code
                line_number=line_fix.line_number,       # line number for the comment
                code_lines=code_lines,                  # original file code lines
                new_code_line=line_fix.suggested_code,  # code line suggested by our model
                rules=rules.rules,                      # list of rules belonging to the model
                fixed_vnodes=line_fix.fixed_vnodes,     # VirtualNode-s with changed y
                confidence=line_fix.confidence,         # overall confidence in the prediction
                describe_rule=_describe_rule,           # function to format a rule as text
                hash_rule=_hash_rule,                   # function to generate a 8-char hash
                describe_change=_describe_change,       # function to format a change as text
                zip=zip,                                # Jinja2 does not have zip() by default
            )

    def _generate_token_fixes(
            self, file: File, fe: FeatureExtractor, feature_extractor_output,
            bblfsh_stub: "bblfsh.aliases.ProtocolServiceStub", rules: Rules,
    ) -> Tuple[List[LineFix], List[VirtualNode], numpy.ndarray, numpy.ndarray]:
        X, y, (vnodes_y, vnodes, vnode_parents, node_parents) = feature_extractor_output
        y_pred_pure, rule_winners, new_rules, grouped_quote_predictions = rules.predict(
            X=X, vnodes_y=vnodes_y, vnodes=vnodes, feature_extractor=fe)
        y_pred = rules.fill_missing_predictions(y_pred_pure, y)
        if self.config["uast_break_check"]:
            y, y_pred, vnodes_y, rule_winners, safe_preds = filter_uast_breaking_preds(
                y=y, y_pred=y_pred, vnodes_y=vnodes_y, vnodes=vnodes, files={file.path: file},
                feature_extractor=fe, stub=bblfsh_stub, vnode_parents=vnode_parents,
                node_parents=node_parents, rule_winners=rule_winners,
                grouped_quote_predictions=grouped_quote_predictions)
            y_pred_pure = y_pred_pure[safe_preds]
        assert len(y) == len(y_pred)
        assert len(y) == len(rule_winners)
        code_generator = CodeGenerator(fe, skip_errors=True)
        new_vnodes = code_generator.apply_predicted_y(vnodes, vnodes_y, rule_winners, new_rules)
        token_fixes = []
        for line_number, line in self._group_line_nodes(
                y, y_pred, vnodes_y, new_vnodes, rule_winners):
            line_ys, line_ys_pred, line_vnodes_y, new_line_vnodes, line_winners = line
            new_code_line = code_generator.generate(
                new_line_vnodes, "local").lstrip("\n").splitlines()[0]
            confidence = self._get_comment_confidence(line_ys, line_ys_pred, line_winners,
                                                      new_rules)
            fixed_vnodes = [vnode for vnode in new_line_vnodes if
                            hasattr(vnode, "y_old") and vnode.y_old != vnode.y]
            token_fixes.append(LineFix(
                line_number=line_number,        # line number for the comment
                suggested_code=new_code_line,   # code line suggested by our model
                fixed_vnodes=fixed_vnodes,      # VirtualNode-s with changed y
                confidence=confidence,          # overall confidence in the prediction, 0-100
            ))
        return token_fixes, new_vnodes, y_pred_pure, y

    @staticmethod
    def get_train_test_split(files: Sequence[File], test_dataset_ratio: float,
                             random_state: int) -> Tuple[Sequence[File], Sequence[File]]:
        """
        Create train test split for the files collection.

        File size is estimated by its length. If there is at least two files, it is guaranteed to
        have at least one in test dataset.

        :param files: The list of `File`-s (see service_data.proto) of the same language.
        :param test_dataset_ratio: The fraction of data that should be taken for test dataset.

        :param random_state: Random state.
        :return: Train files and test files.
        """
        if test_dataset_ratio == 0:
            return files, []
        random.seed(random_state)
        files = random.sample(files, k=len(files))
        target_train_length = sum(map(lambda f: len(f.content), files)) * (1 - test_dataset_ratio)

        accumulated_train_length = 0
        i = 0
        for i, file in enumerate(files):  # noqa: B007
            accumulated_train_length += len(file.content)
            if accumulated_train_length >= target_train_length:
                break
        min_files_number = min(1, len(files) - 1)
        if len(files) - i - 1 < min_files_number:
            i = len(files) - min_files_number - 1
        return files[:i + 1], files[i + 1:]

    @staticmethod
    def _get_comment_confidence(line_ys: Sequence[int], line_ys_pred: Sequence[int],
                                line_winners: Sequence[int], rules: Rules) -> int:
        confidence = 0
        confidence_count = 0
        for yi, y_predi, winner in zip(line_ys, line_ys_pred, line_winners):
            if winner < 0:
                continue
            if yi != y_predi:
                confidence += rules.rules[winner].stats.conf
                confidence_count += 1
        return int(round(confidence * 100 / confidence_count))

    @staticmethod
    def _group_line_nodes(
        y: Sequence[int], y_pred: Sequence[int], vnodes_y: Sequence[VirtualNode],
        new_vnodes: Sequence[VirtualNode], rule_winners: Sequence[int],
        ) -> Tuple[int, Tuple[Sequence[int], Sequence[int], Sequence[VirtualNode],
                              Sequence[VirtualNode], Sequence[int]]]:
        """
        Group virtual nodes and related lists from feature extractor by line number.

        It yields line number and sublists of corresponding items from all input sequences.
        Line sublists are skipped in case there is no difference in predicted and original labels.
        Line sublists are merged in case new line on the end was replaced by target without
        newline. It is a helper function for `FormatAnalyser._generate_file_comments()`

        :param y: Sequence of original labels.
        :param y_pred: Sequence of predicted labels by the model.
        :param vnodes_y: Sequence of the labeled `VirtualNode`-s corresponding to labeled samples.
        :param new_vnodes: Sequence of all the `VirtualNode`-s corresponding to the input with \
                           applied predictions. `CodeGenerator.apply_predicted_y()` is used for \
                           that.
        :param rule_winners: List of rule winners.
        :return: 1-based line number and sublists of corresponding items from all input sequences.
        """
        line_no = None
        line_items = None
        result = []
        for (yi, y_predi, vnode_y, winner) in zip(y, y_pred, vnodes_y, rule_winners):
            if winner < 0:
                continue
            if vnode_y.start.line != line_no:
                if line_items:
                    result.append((line_no, zip(*line_items)))
                line_items = []
                line_no = vnode_y.start.line
            if yi != y_predi:
                line_items.append((yi, y_predi, vnode_y, winner))
        if line_items:
            result.append((line_no, zip(*line_items)))

        for line_no, (line_y, line_y_pred, line_vnodes_y, line_rule_winners) in result:
            line_vnodes = []
            for vnode in new_vnodes:
                if vnode.start.line > line_no:
                    if (line_vnodes and line_vnodes[-1].y is not None and
                            CLASS_INDEX[CLS_NEWLINE] in line_vnodes[-1].y):
                        break
                    else:
                        line_vnodes.append(vnode)
                        continue
                elif vnode.end.line >= line_no:
                    line_vnodes.append(vnode)
            yield (int(line_vnodes_y[0].start.line),
                   (line_y, line_y_pred, line_vnodes_y, line_vnodes, line_rule_winners))

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
                config, global_config,
            ))
