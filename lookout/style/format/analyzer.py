"""Analyzer that detects bad formatting by learning on the existing code in the repository."""
import functools
from itertools import chain
import logging
import os
from pprint import pformat
from typing import Any, Iterable, List, Mapping, NamedTuple, Sequence, Tuple

import bblfsh  # noqa: F401
from jinja2 import Template
from lookout.core.analyzer import Analyzer, ReferencePointer
from lookout.core.api.service_analyzer_pb2 import Comment
from lookout.core.api.service_data_pb2 import File
from lookout.core.data_requests import DataService, \
    with_changed_uasts_and_contents, with_uasts_and_contents
from lookout.core.lib import files_by_language, filter_files, find_deleted_lines, find_new_lines
import numpy

from lookout.style.format.classes import CLASS_INDEX, CLS_NEWLINE
from lookout.style.format.code_generator import CodeGenerator
from lookout.style.format.descriptions import describe_rule, get_change_description
from lookout.style.format.feature_extractor import FeatureExtractor
from lookout.style.format.model import FormatModel
from lookout.style.format.optimizer import Optimizer
from lookout.style.format.postprocess import filter_uast_breaking_preds
from lookout.style.format.rules import Rules, TrainableRules
from lookout.style.format.utils import generate_comment, merge_dicts
from lookout.style.format.virtual_node import VirtualNode


FixData = NamedTuple("FixData", (
    ("error", str),                           # error message
    ("language", str),                        # programming language of the code
    ("line_number", int),                     # line number for the comment
    ("head_file", File),                      # file from head revision
    ("base_file", File),                      # file from base revision
    ("suggested_code", str),                  # code line predicted by our model
    ("all_vnodes", List[VirtualNode]),        # VirtualNode-s which correspond to this file
    ("fixed_vnodes", List[VirtualNode]),      # VirtualNode-s with fixed y
    ("winner_rules", List[int]),              # Rule indices
    ("confidence", int),                      # overall confidence in the prediction, 0-100
    ("feature_extractor", FeatureExtractor),  # FeatureExtractor involved
))


class FormatAnalyzer(Analyzer):
    """Detect bad formatting by training on existing code and analyzing pull requests."""

    _log = logging.getLogger("FormatAnalyzer")
    model_type = FormatModel
    name = "style.format.analyzer.FormatAnalyzer"
    version = 1
    description = "Source code formatting: whitespace, new lines, quotes, braces."
    defaults_for_analyze = {
        "confidence_threshold": 0.95,
        "support_threshold": 80,
        "report_code_lines": False,
        "report_triggered_rules": False,
        "report_parse_failures": False,
        "uast_break_check": True,
        "comment_template": os.path.join(os.path.dirname(__file__), "templates",
                                         "comment_default.jinja2"),
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
                "prune_dataset_ratio": .2,
                "n_estimators": 10,
                "random_state": 42,
            },
            "n_jobs": -1,
            "n_iter": 5,
            "cv": 3,
            "line_length_limit": 500,
            "lower_bound_instances": 500,
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
        for fix_data in self.generate_fixes(data_service, list(data["changes"])):
            text = fix_data.error or self.render_comment_text(fix_data)
            comments.append(generate_comment(
                text=text,
                filename=fix_data.head_file.path,
                line=fix_data.line_number,
                confidence=fix_data.confidence))
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
            optimizer = Optimizer(n_jobs=lang_config["n_jobs"],
                                  n_iter=lang_config["n_iter"],
                                  cv=lang_config["cv"],
                                  random_state=lang_config["trainable_rules"]["random_state"])
            best_score, best_params = optimizer.optimize(X, y)
            cls._log.debug("score of the best estimator found: %.6f", best_score)
            cls._log.debug("params of the best estimator found: %s", str(best_params))
            cls._log.debug("training the model with complete data")
            lang_config["trainable_rules"].update(best_params)
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
            if trainable_rules.rules.rules:
                model[language] = trainable_rules.rules
            else:
                cls._log.warning("Model for %s had 0 rules. Skipping." % language)
        cls._log.info("trained %s", model)
        return model

    def generate_fixes(self, data_service: DataService, changes: Sequence) -> Iterable[FixData]:
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
        for lang, head_files in head_files_by_lang.items():
            if lang not in self.model:
                log.warning("skipped %d written in %s. Rules for %s do not exist in model",
                            len(head_files), lang, lang)
                continue
            rules = self.model[lang]
            rules = rules.filter_by_confidence(self.config["confidence_threshold"]) \
                .filter_by_support(self.config["support_threshold"])
            for file in filter_files(head_files, rules.origin_config["line_length_limit"], log):
                log.debug("Analyze %s", file.path)
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
                fe = FeatureExtractor(language=lang, **rules.origin_config["feature_extractor"])
                feature_extractor_output = fe.extract_features([file], [lines])
                if feature_extractor_output is None:
                    if self.config["report_parse_failures"]:
                        log.warning("Failed to parse %s", file.path)
                        yield FixData(
                            base_file=prev_file, head_file=file, confidence=100, line_number=0,
                            error="Failed to parse", language=lang, feature_extractor=fe,
                            all_vnodes=[], fixed_vnodes=[], winner_rules=[], suggested_code="")
                else:
                    yield from self._generate_comments_data(
                        lang, file, prev_file, fe, feature_extractor_output,
                        data_service.get_bblfsh(), rules)

    def render_comment_text(self, fix_data: FixData) -> str:
        """
        Generate the text of the comment at the specified line.

        :param fix_data: Full information required to render a comment text.
        :return: string with the generated comment.
        """
        rules = self.model[fix_data.language]
        _describe_rule = functools.partial(
            describe_rule, feature_extractor=fix_data.feature_extractor)
        _describe_change = functools.partial(
            get_change_description, feature_extractor=fix_data.feature_extractor)
        code_lines = fix_data.head_file.content.decode("utf-8", "replace").splitlines()
        return self.comment_template.render(
                config=self.config,                     # configuration of the analyzer
                language=fix_data.language,             # programming language of the code
                line_number=fix_data.line_number,       # line number for the comment
                code_lines=code_lines,                  # original file code lines
                new_code_line=fix_data.suggested_code,  # code line suggested by our model
                rules=rules.rules,                      # list of rules belonging to the model
                fixed_vnodes=fix_data.fixed_vnodes,     # VirtualNode-s which changed y
                winner_rules=fix_data.winner_rules,     # changed vnodes and winner rule indices
                confidence=fix_data.confidence,         # overall confidence in the prediction
                describe_rule=_describe_rule,           # function to format a rule as text
                describe_change=_describe_change,       # function to format a change as text
                zip=zip,                                # Jinja2 does not have zip() by default
            )

    def _generate_comments_data(
            self, language: str, file: File, base_file: File, fe: FeatureExtractor,
            feature_extractor_output, bblfsh_stub: "bblfsh.aliases.ProtocolServiceStub",
            rules: Rules) -> Iterable[FixData]:
        X, y, (vnodes_y, vnodes, vnode_parents, node_parents) = feature_extractor_output
        y_pred, rule_winners, new_rules = rules.predict(X=X, vnodes_y=vnodes_y, vnodes=vnodes,
                                                        feature_extractor=fe)
        if self.config["uast_break_check"]:
            y, y_pred, vnodes_y, rule_winners, safe_preds = filter_uast_breaking_preds(
                y=y, y_pred=y_pred, vnodes_y=vnodes_y, vnodes=vnodes, files={file.path: file},
                feature_extractor=fe, stub=bblfsh_stub, vnode_parents=vnode_parents,
                node_parents=node_parents, rule_winners=rule_winners, log=self._log)
        assert len(y) == len(y_pred)
        assert len(y) == len(rule_winners)
        code_generator = CodeGenerator(fe, skip_errors=True)
        new_vnodes = code_generator.apply_predicted_y(vnodes, vnodes_y, y_pred)
        for line_number, line in self._group_line_nodes(
                y, y_pred, vnodes_y, new_vnodes, rule_winners):
            line_ys, line_ys_pred, line_vnodes_y, new_line_vnodes, line_winners = line
            new_code_line = code_generator.generate(
                new_line_vnodes, "local").lstrip("\n").splitlines()[0]
            confidence = self._get_comment_confidence(line_ys, line_ys_pred, line_winners,
                                                      new_rules)
            vnodes_changed = [vnode for vnode in new_line_vnodes if
                              hasattr(vnode, "y_old") and vnode.y_old != vnode.y]
            yield FixData(
                error="",                      # success
                language=language,             # programming language of the code
                line_number=line_number,       # line number for the comment
                head_file=file,                # file from head revision
                base_file=base_file,           # file from base revision
                suggested_code=new_code_line,  # code line suggested by our model
                all_vnodes=new_vnodes,         # VirtualNode-s which construct the file
                fixed_vnodes=vnodes_changed,   # VirtualNode-s with changed y
                winner_rules=line_winners,     # applied rule indices
                confidence=confidence,         # overall confidence in the prediction, 0-100
                feature_extractor=fe,          # FeatureExtractor involved
            )

    @staticmethod
    def _get_comment_confidence(line_ys: Sequence[int], line_ys_pred: Sequence[int],
                                line_winners: Sequence[int], rules: Rules) -> int:
        confidence = 0
        confidence_count = 0
        for yi, y_predi, winner in zip(line_ys, line_ys_pred, line_winners):
            if yi != y_predi:
                confidence += rules.rules[winner].stats.conf
                confidence_count += 1
        return int(round(confidence * 100 / confidence_count))

    @staticmethod
    def _group_line_nodes(y: Sequence[int], y_pred: Sequence[int], vnodes_y: Sequence[VirtualNode],
                          new_vnodes: Sequence[VirtualNode], rule_winners: Sequence[int],
                          ) -> Tuple[int, Tuple]:
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
        line_ys, line_ys_pred, line_vnodes_y, line_winners = [], [], [], []
        generate_comment = False
        vnodes_index = 0
        for yi, y_predi, vnode_y, winner in zip(y, y_pred, vnodes_y, rule_winners):
            if not line_ys or vnode_y.start.line == line_vnodes_y[0].start.line:
                # collect all nodes on the same line
                line_ys.append(yi)
                line_ys_pred.append(y_predi)
                line_vnodes_y.append(vnode_y)
                line_winners.append(winner)
                if yi != y_predi:
                    generate_comment = True
                continue
            else:
                if generate_comment:
                    line_vnodes = []
                    for vnode in new_vnodes[vnodes_index:]:
                        if vnode.start.line > line_vnodes_y[0].start.line:
                            if (line_vnodes and line_vnodes[-1].y is not None and
                                    CLASS_INDEX[CLS_NEWLINE] in line_vnodes[-1].y):
                                break
                            else:
                                line_vnodes.append(vnode)
                                continue
                        if vnode.end.line < line_vnodes_y[0].start.line:
                            continue
                        line_vnodes.append(vnode)
                        vnodes_index += 1
                    yield (int(line_vnodes_y[0].start.line),
                           (line_ys, line_ys_pred, line_vnodes_y, line_vnodes, line_winners))
                generate_comment = yi != y_predi
                line_ys, line_ys_pred, line_vnodes_y, line_winners = (
                    [yi], [y_predi], [vnode_y], [winner])

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
