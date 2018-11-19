"""Facilities to report the quality of a given model on a given dataset."""
from collections import Counter, OrderedDict
import glob
import logging
import os
from typing import Any, Iterable, List, MutableMapping, Union
import warnings

from bblfsh import BblfshClient
import jinja2
from lookout.core.analyzer import Analyzer, AnalyzerModel, ReferencePointer
from lookout.core.api.service_analyzer_pb2 import Comment
from lookout.core.data_requests import DataService, \
    with_changed_uasts_and_contents, with_uasts_and_contents
from lookout.core.lib import files_by_language, filter_files, find_new_lines
import numpy
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.metrics.classification import _check_targets
from sklearn.utils.multiclass import unique_labels

from lookout.style.format.analyzer import FormatAnalyzer
from lookout.style.format.benchmarks.time_profile import profile
from lookout.style.format.descriptions import describe_rule
from lookout.style.format.feature_extractor import FeatureExtractor
from lookout.style.format.model import FormatModel
from lookout.style.format.postprocess import filter_uast_breaking_preds
from lookout.style.format.utils import generate_comment, merge_dicts, prepare_files
from lookout.style.format.virtual_node import VirtualNode


def classification_report(y_true, y_pred, labels=None, target_names=None, sample_weight=None,
                          digits=2, output_dict=False):  # noqa: W9015,W9011
    """
    Fake classification report docstring to avoid bug https://gitlab.com/pycqa/flake8/issues/375.

    :param y_true: 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) target values.
    :param y_pred: 1d array-like, or label indicator array / sparse matrix
        Estimated targets as returned by a classifier.
    :param labels: array, shape = [n_labels]
        Optional list of label indices to include in the report.
    :param target_names: list of strings
        Optional display names matching the labels (same order).
    :param sample_weight: array-like of shape = [n_samples], optional
        Sample weights.
    :param digits: int
        Number of digits for formatting output floating point values.
        When ``output_dict`` is ``True``, this will be ignored and the
        returned values will not be rounded.
    :param output_dict: bool (default = False)
        If True, return output as dict
    :return: string / dict
        Text summary of the precision, recall, F1 score for each class.
        Dictionary returned if output_dict is True. Dictionary has the
        following structure::
            {'label 1': {'precision':0.5,
                         'recall':1.0,
                         'f1-score':0.67,
                         'support':1},
             'label 2': { ... },
              ...
            }
        The reported averages include micro average (averaging the
        total true positives, false negatives and false positives), macro
        average (averaging the unweighted mean per label), weighted average
        (averaging the support-weighted mean per label) and sample average
        (only for multilabel classification). See also
        :func:`precision_recall_fscore_support` for more details on averages.
        Note that in binary classification, recall of the positive class
        is also known as "sensitivity"; recall of the negative class is
        "specificity".
    """
    """Build a text report showing the main classification metrics
    Read more in the :ref:`User Guide <classification_report>`.

    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) target values.
    y_pred : 1d array-like, or label indicator array / sparse matrix
        Estimated targets as returned by a classifier.
    labels : array, shape = [n_labels]
        Optional list of label indices to include in the report.
    target_names : list of strings
        Optional display names matching the labels (same order).
    sample_weight : array-like of shape = [n_samples], optional
        Sample weights.
    digits : int
        Number of digits for formatting output floating point values.
        When ``output_dict`` is ``True``, this will be ignored and the
        returned values will not be rounded.
    output_dict : bool (default = False)
        If True, return output as dict
    Returns
    -------
    report : string / dict
        Text summary of the precision, recall, F1 score for each class.
        Dictionary returned if output_dict is True. Dictionary has the
        following structure::
            {'label 1': {'precision':0.5,
                         'recall':1.0,
                         'f1-score':0.67,
                         'support':1},
             'label 2': { ... },
              ...
            }
        The reported averages include micro average (averaging the
        total true positives, false negatives and false positives), macro
        average (averaging the unweighted mean per label), weighted average
        (averaging the support-weighted mean per label) and sample average
        (only for multilabel classification). See also
        :func:`precision_recall_fscore_support` for more details on averages.
        Note that in binary classification, recall of the positive class
        is also known as "sensitivity"; recall of the negative class is
        "specificity".
    Examples
    --------
    >>> from sklearn.metrics import classification_report
    >>> y_true = [0, 1, 2, 2, 2]
    >>> y_pred = [0, 0, 2, 2, 1]
    >>> target_names = ['class 0', 'class 1', 'class 2']
    >>> print(classification_report(y_true, y_pred, target_names=target_names))
                  precision    recall  f1-score   support
    <BLANKLINE>
         class 0       0.50      1.00      0.67         1
         class 1       0.00      0.00      0.00         1
         class 2       1.00      0.67      0.80         3
    <BLANKLINE>
       micro avg       0.60      0.60      0.60         5
       macro avg       0.50      0.56      0.49         5
    weighted avg       0.70      0.60      0.61         5
    <BLANKLINE>
    """ # noqa: D202,D205,D400,D413,E261,E501,W9015,W9011,Q000

    y_type, y_true, y_pred = _check_targets(y_true, y_pred)

    labels_given = True
    if labels is None:
        labels = unique_labels(y_true, y_pred)
        labels_given = False
    else:
        labels = numpy.asarray(labels)

    if target_names is not None and len(labels) != len(target_names):
        if labels_given:
            warnings.warn(
                "labels size, {0}, does not match size of target_names, {1}"
                .format(len(labels), len(target_names))
            )
        else:
            raise ValueError(
                "Number of classes, {0}, does not match size of "
                "target_names, {1}. Try specifying the labels "
                "parameter".format(len(labels), len(target_names))
            )
    if target_names is None:
        target_names = [u"%s" % l for l in labels]

    headers = ["precision", "recall", "f1-score", "support"]
    # compute per-class results without averaging
    p, r, f1, s = precision_recall_fscore_support(y_true, y_pred,
                                                  labels=labels,
                                                  average=None,
                                                  sample_weight=sample_weight)
    rows = zip(target_names, p, r, f1, s)

    if y_type.startswith("multilabel"):
        average_options = ("micro", "macro", "weighted", "samples")
    else:
        average_options = ("micro", "macro", "weighted")

    if output_dict:
        report_dict = {label[0]: label[1:] for label in rows}
        for label, scores in report_dict.items():
            report_dict[label] = dict(zip(headers,
                                          [i.item() for i in scores]))
    else:
        longest_last_line_heading = "weighted avg"
        name_width = max(len(cn) for cn in target_names)
        width = max(name_width, len(longest_last_line_heading), digits)
        head_fmt = u"{:>{width}s} ' + u' {:>9}" * len(headers)
        report = head_fmt.format(u"", *headers, width=width)
        report += u"\n\n"
        row_fmt = u"{:>{width}s} " + u" {:>9.{digits}f}" * 3 + u" {:>9}\n"
        for row in rows:
            report += row_fmt.format(*row, width=width, digits=digits)
        report += u"\n"

    # compute all applicable averages
    for average in average_options:
        line_heading = average + " avg"
        # compute averages with specified averaging method
        avg_p, avg_r, avg_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=labels,
            average=average, sample_weight=sample_weight)
        avg = [avg_p, avg_r, avg_f1, numpy.sum(s)]

        if output_dict:
            report_dict[line_heading] = dict(
                zip(headers, [i.item() for i in avg]))
        else:
            report += row_fmt.format(line_heading, *avg,
                                     width=width, digits=digits)

    if output_dict:
        return report_dict
    else:
        return report


def generate_report(y, y_pred, vnodes_y, n_files, target_names):
    """Generate report: classification report, confusion matrix, files with most errors."""
    # classification report
    c_report = classification_report(y, y_pred, output_dict=True, target_names=target_names,
                                     labels=unique_labels(y, y_pred))

    avr_keys = ["macro avg", "micro avg", "weighted avg"]
    c_sorted = OrderedDict((key, c_report[key])
                           for key in sorted(c_report, key=lambda k: -c_report[k]["support"])
                           if key not in avr_keys)
    for key in avr_keys:
        c_sorted[key] = c_report[key]

    # confusion matrix
    mat = confusion_matrix(y, y_pred)

    # sort files by mispredictions
    file_mispred = []
    for gt, pred, vn in zip(y, y_pred, vnodes_y):
        if gt != pred:
            file_mispred.append(vn.path)
    file_stat = Counter(file_mispred)

    to_show = file_stat.most_common()
    if n_files > 0:
        to_show = to_show[:n_files]

    loader = jinja2.FileSystemLoader(("/", os.path.dirname(__file__), os.getcwd()),
                                     followlinks=True)
    env = jinja2.Environment(
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )
    env.globals.update(range=range)

    template = loader.load(env, "templates/quality_report.md.jinja2")
    res = template.render(cl_report=c_sorted, conf_mat=mat, target_names=target_names,
                          files=to_show)
    return res


def generate_model_report(model: FormatModel, language: str,
                          feature_extractor: FeatureExtractor) -> str:
    """
    Generate report about model - description for each rule, min/max support, min/max confidence.

    :param model: trained format model.
    :param language: language.
    :param feature_extractor: feature extractor.
    :return: report in str format.
    """
    rules = model[language]
    min_support, max_support = float("inf"), -1
    min_conf, max_conf = 1, 0
    rules_desc = []
    packages = model.meta["environment"]["packages"]
    for i, rule in enumerate(rules.rules):
        min_support = min(min_support, rule.stats.support)
        max_support = max(max_support, rule.stats.support)
        min_conf = min(min_conf, rule.stats.conf)
        max_conf = max(max_conf, rule.stats.conf)
        rules_desc.append((i, describe_rule(rule, feature_extractor).replace("\n", "<br>")))

    loader = jinja2.FileSystemLoader(("/", os.path.dirname(__file__), os.getcwd()),
                                     followlinks=True)
    env = jinja2.Environment(
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )
    env.globals.update(range=range)

    template = loader.load(env, "templates/model_report.md.jinja2")
    res = template.render(rules_desc=rules_desc, min_support=min_support, max_support=max_support,
                          min_conf=min_conf, max_conf=max_conf, language=language, rules=rules,
                          packages=packages)
    return res


@profile
def quality_report(input_pattern: str, bblfsh: str, language: str, n_files: int, model_path: str,
                   ) -> None:
    """Print several different reports for a given model on a given dataset."""
    model = FormatModel().load(model_path)
    rules = model[language]
    print("Model parameters: %s" % rules.origin_config)
    print("Stats about rules: %s" % rules)
    # prepare input files
    client = BblfshClient(bblfsh)
    filenames = glob.glob(input_pattern, recursive=True)
    files = prepare_files(filenames, client, language)
    print("Number of files: %s" % (len(files)))
    # extract features and check result
    fe = FeatureExtractor(language=language, **rules.origin_config["feature_extractor"])
    res = fe.extract_features(files)
    if res is None:
        print("Failed to parse files, aborting report...")
        return
    X, y, (vnodes_y, vnodes, vnode_parents, node_parents) = res
    # predict with model and generate report
    y_pred, rule_winners, _ = rules.predict(X=X, vnodes_y=vnodes_y, vnodes=vnodes,
                                            feature_extractor=fe)
    y, y_pred, vnodes_y, rule_winners, safe_preds = filter_uast_breaking_preds(
        y=y, y_pred=y_pred, vnodes_y=vnodes_y, vnodes=vnodes, files={f.path: f for f in files},
        feature_extractor=fe, stub=client._stub, vnode_parents=vnode_parents,
        node_parents=node_parents, rule_winners=rule_winners,
        log=logging.getLogger("quality_report"))
    target_names = fe.composite_class_representations
    target_names = [target_names[label] for label in unique_labels(y, y_pred)]
    print(generate_report(y=y, y_pred=y_pred, target_names=target_names, vnodes_y=vnodes_y,
                          n_files=n_files))


class ReportModel(AnalyzerModel):
    """Model for ReportAnalyzer - store configuration."""

    NAME = "style.format.quality_report_analyzer.ReportModel"

    def __init__(self, config=None, **kwargs):
        """Initialize ReportModel with configuration."""
        super().__init__(**kwargs)
        self.config = config

    def _generate_tree(self) -> dict:
        """
        Return the tree to store in ASDF file.

        :return: dict.
        """
        return self.config

    def _load_tree(self, tree: dict) -> None:
        """
        Attach the needed data from the tree.

        :param tree: asdf file tree.
        :return: None
        """
        self.config = tree


class ReportAnalyzer(Analyzer):
    """
    Base class for different kind of reports.

    * analyze - generate report per changed files or aggregated changes.
    * train - generate report for all files.

    Child classes are required to implement 3 methods:
    * generate_report
    * generate_model_report (optional - by default it will return empty string)
    * defaults  (optional - by default it will be empty)
    """

    model_type = ReportModel
    base_config = {"aggregate": False, "model_path": None, "report_parse_failures": False}
    defaults = {}  # has to be filled in child class. It should store analyzer-specific configs.

    def __init__(self, model: FormatModel, url: str, config: MutableMapping[str, Any]) -> None:
        """Initialization."""
        super().__init__(model, url, config)
        self.config = self._load_analysis_config(self.config)

    @classmethod
    def generate_report(cls, y: Union[numpy.ndarray, Iterable[Union[int, float]]],
                        y_pred: Union[numpy.ndarray, Iterable[Union[int, float]]],
                        vnodes_y: Iterable[VirtualNode], target_names: Iterable[str],
                        rule_winners: Iterable[int],
                        config: MutableMapping[str, Any],
                        model: FormatAnalyzer,
                        feature_extractor: FeatureExtractor) -> str:
        """
        Generate report. Has to be implemented in child class.

        :param y: labels.
        :param y_pred: predicted labels.
        :param vnodes_y: virtual nodes for labels.
        :param target_names: class names.
        :param rule_winners: rule winner.
        :param config: configuration.
        :param model: pretrained model that was loaded to make report.
        :param feature_extractor: feature extractor.
        :return: report.
        """
        raise NotImplemented

    @classmethod
    def generate_model_report(cls, model: FormatModel, config: MutableMapping[str, Any],
                              feature_extractor: FeatureExtractor) -> str:
        """
        Generate report about model. Has to be implemented in child class.

        :param model: model that should be described in report.
        :param config: configuration.
        :param feature_extractor: feature extractor.
        :return: report.
        """
        return ""

    @with_changed_uasts_and_contents
    def analyze(self, ptr_from: ReferencePointer, ptr_to: ReferencePointer,
                data_service: DataService, **data) -> List[Comment]:
        """
        Analyze changes between revisions and make quality report per file or for all changes.

        :param ptr_from: Git repository state pointer to the base revision.
        :param ptr_to: Git repository state pointer to the head revision.
        :param data_service: Connection to the Lookout data retrieval service, not used.
        :param data: Contains "changes" - the list of changes in the pointed state.
        :return: List of comments.
        """
        self.model = FormatModel().load(self.config["model_path"])
        # prepare data
        changes = list(data["changes"])
        base_files_by_lang = files_by_language(c.base for c in changes)
        head_files_by_lang = files_by_language(c.head for c in changes)
        # prepare containers for comments and intermediate results
        if self.config["aggregate"]:
            agg_y, agg_vnodes_y, agg_y_pred = [], [], []
        comments = []
        for lang, head_files in head_files_by_lang.items():
            if lang not in self.model:
                self.log.warning("skipped %d written in %s. Rules for %s do not exist in model",
                                 len(head_files), lang, lang)
                continue
            rules = self.model[lang]
            for file in filter_files(head_files, rules.origin_config["line_length_limit"]):
                try:
                    prev_file = base_files_by_lang[lang][file.path]
                except KeyError:
                    lines = None
                else:
                    lines = [find_new_lines(prev_file, file)]
                fe = FeatureExtractor(language=lang, **rules.origin_config["feature_extractor"])
                res = fe.extract_features([file], lines)
                if res is None:
                    if self.config["report_parse_failures"]:
                        comments.append(generate_comment(filename=file.path, confidence=100,
                                                         line=1, text="Failed to parse this file"))
                    self.log.warning("Failed to parse %s", file.path)
                    continue
                X, y, (vnodes_y, vnodes, vnode_parents, node_parents) = res
                y_pred, rule_winners, _ = rules.predict(X=X, vnodes_y=vnodes_y, vnodes=vnodes,
                                                        feature_extractor=fe)
                y, y_pred, vnodes_y, rule_winners, safe_preds = filter_uast_breaking_preds(
                    y=y, y_pred=y_pred, vnodes_y=vnodes_y, vnodes=vnodes, files={file.path: file},
                    feature_extractor=fe, stub=data_service.get_bblfsh(),
                    vnode_parents=vnode_parents, node_parents=node_parents,
                    rule_winners=rule_winners, log=self.log)
                self.log.debug("y.shape %s" % y.shape)
                self.log.debug("len(vnodes_y) %s" % len(vnodes_y))
                self.log.debug("y_pred.shape %s" % y_pred.shape)
                if self.config["aggregate"] is True:
                    agg_y.append(y)
                    agg_vnodes_y.extend(vnodes_y)
                    agg_y_pred.append(y_pred)
                else:
                    target_names = fe.composite_class_representations
                    assert len(y) == len(y_pred)

                    report = self.generate_report(
                        y=y, y_pred=y_pred, vnodes_y=vnodes_y, target_names=target_names,
                        config=self.config, model=self.model, feature_extractor=fe,
                        rule_winners=rule_winners,
                    )
                    comments.append(generate_comment(filename=file.path, confidence=100,
                                                     line=0, text=report))
            model_report = self.generate_model_report(self.model, self.config, fe)
            if model_report:
                comments.append(generate_comment(filename=file.path, confidence=100,
                                                 line=0, text=model_report))
            if self.config["aggregate"]:
                agg_y = numpy.hstack(agg_y)
                target_names = fe.composite_class_representations
                report = self.generate_report(
                    y=agg_y, y_pred=numpy.hstack(agg_y_pred), vnodes_y=agg_vnodes_y,
                    target_names=target_names, config=self.config, model=self.model,
                    feature_extractor=fe, rule_winners=rule_winners,
                )
                comments.append(generate_comment(filename=file.path, confidence=100,
                                                 line=0, text=report))
        return comments

    @classmethod
    @with_uasts_and_contents
    def train(cls, ptr: ReferencePointer, config: MutableMapping[str, Any],
              data_service: DataService, **data) -> None:
        """
        Analyze a set of changes from one revision to another and make report for all files.

        :param ptr: Git repository state pointer.
        :param config: configuration dict.
        :param data: contains "files" - the list of files in the pointed state.
        :param data_service: connection to the Lookout data retrieval service, not used.
        :return: AnalyzerModel containing the learned rules, per language.
        """
        config = cls._load_analysis_config(config)
        assert config["model_path"] is not None, \
            "'model_path' is missed in configuration. Please check documentation how to make " \
            "query correct."
        model = FormatModel().load(config["model_path"])
        for lang, head_files in files_by_language(data["files"]).items():
            if lang not in model:
                cls.log.warning("skipped %d written in %s. Rules for %s do not exist in model",
                                len(head_files), lang, lang)
                continue
            rules = model[lang]
            filtered_files = list(filter_files(head_files,
                                               rules.origin_config["line_length_limit"]))
            fe = FeatureExtractor(language=lang, **rules.origin_config["feature_extractor"])
            res = fe.extract_features(filtered_files)
            if res is None:
                cls.log.warning("Failed to parse files")
                continue
            X, y, (vnodes_y, vnodes, vnode_parents, node_parents) = res

            y_pred, rule_winners, _ = rules.predict(X=X, vnodes_y=vnodes_y, vnodes=vnodes,
                                                    feature_extractor=fe)
            y, y_pred, vnodes_y, rule_winners, safe_preds = filter_uast_breaking_preds(
                y=y, y_pred=y_pred, vnodes_y=vnodes_y, vnodes=vnodes,
                files={f.path: f for f in filtered_files}, feature_extractor=fe,
                stub=data_service.get_bblfsh(), vnode_parents=vnode_parents,
                node_parents=node_parents, rule_winners=rule_winners, log=cls.log)
            target_names = fe.composite_class_representations
            assert len(y) == len(y_pred)
            report = cls.generate_report(y=y, y_pred=y_pred, vnodes_y=vnodes_y,
                                         target_names=target_names, config=config, model=model,
                                         feature_extractor=fe, rule_winners=rule_winners)
            model_report = cls.generate_model_report(model, config, fe)
            if model_report:
                cls.log.info("-" * 20)
                cls.log.info(model_report)
            cls.log.info("-" * 20)
            cls.log.info(report)
            return ReportModel(config=config)

    @classmethod
    def _load_analysis_config(cls, config: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
        """
        Merge config for analyze call with default config values stored inside this function.

        :param config: User-defined config.
        :return: Full config.
        """
        assert cls.defaults is not None
        assert isinstance(cls.defaults, MutableMapping)
        return merge_dicts(cls.base_config, cls.defaults, config)


class QualityReportAnalyzer(ReportAnalyzer):
    """
    Quality report analyzer.

    * analyze - generate report per changed files or aggregated changes.
    * train - generate report for all files.

    How-to:

    1) Launch analyzer: `analyzer run lookout.style.format.quality_report_analyzer -c config.yml`

    2) Query analyzer

    2.1) Quality report query for all files:
    ```
    >>> lookout-sdk push ipv4://analyzer:port --git-dir /git/dir/ --from REV1 --to  REV2  \
    --config-json='{"style.format.analyzer.QualityReportAnalyzer":  \
    {"model_path": "/saved/model.asdf"}}'
    ```

    2.2) Quality report for changes per file:
    ```
    >>> lookout-sdk review ipv4://analyzer:port --git-dir /git/dir/ --from REV1 --to  REV2  \
    --config-json='{"style.format.analyzer.QualityReportAnalyzer":  \
    {"model_path": "/saved/model.asdf"}}'
    ```

    2.3) Quality report for aggregated changes:
    ```
    >>> lookout-sdk review ipv4://analyzer:port --git-dir /git/dir/ --from REV1 --to  REV2  \
    --config-json='{"style.format.analyzer.QualityReportAnalyzer":  \
    {"model_path": "/saved/model.asdf", "aggregate": true}}'
    ```
    """

    log = logging.getLogger("QualityReportAnalyzer")
    name = "style.format.analyzer.QualityReportAnalyzer"
    version = "1"
    description = "Source code formatting quality report analysis: " \
                  "whitespace, new lines, quotes, etc."
    defaults = {"n_files": 10}

    def __init__(self, model: FormatModel, url: str, config: MutableMapping[str, Any]) -> None:
        """Initialize QualityReportAnalyzer with pretrained model and configuration."""
        super().__init__(model, url, config)
        self.config = self._load_analysis_config(self.config)

    @classmethod
    def generate_model_report(cls, model: FormatModel, config: MutableMapping[str, Any],
                              feature_extractor: FeatureExtractor):
        """
        Generate report about model. Has to be implemented in child class.

        :param model: model that should be described in report.
        :param config: configuration.
        :param feature_extractor: feature extractor.
        :return: report.
        """
        res = []
        for lang in model:
            res.append(generate_model_report(model=model,
                                             feature_extractor=feature_extractor,
                                             language=lang))
        return "\n".join(res)

    @classmethod
    def generate_report(cls, y: Union[numpy.ndarray, Iterable[Union[int, float]]],
                        y_pred: Union[numpy.ndarray, Iterable[Union[int, float]]],
                        vnodes_y: Iterable[VirtualNode], target_names: Iterable[str],
                        config: MutableMapping[str, Any], **kwargs) -> str:
        """
        Generate quality report: classification report, confusion matrix, files with most errors.

        :param y: labels.
        :param y_pred: predicted labels.
        :param vnodes_y: virtual nodes for labels.
        :param target_names: class names.
        :param config: configuration.
        :param kwargs: helper to handel additional parameters.
        :return: report.
        """
        return generate_report(y, y_pred, vnodes_y, config["n_files"], target_names)


analyzer_class = QualityReportAnalyzer
