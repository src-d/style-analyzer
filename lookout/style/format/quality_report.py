"""Facilities to report the quality of a given model on a given dataset."""
from collections import Counter
import glob
import io
import logging
from typing import Any, Iterable, List, MutableMapping, Union

from bblfsh import BblfshClient
from lookout.core.analyzer import Analyzer, AnalyzerModel, ReferencePointer
from lookout.core.api.service_analyzer_pb2 import Comment
from lookout.core.api.service_data_pb2_grpc import DataStub
from lookout.core.data_requests import (with_changed_uasts_and_contents,
                                        with_uasts_and_contents)
from lookout.core.lib import files_by_language, filter_files, find_new_lines
import numpy
from sklearn.metrics import classification_report, confusion_matrix

from lookout.style.format.analyzer import FormatAnalyzer
from lookout.style.format.benchmarks.profile import profile
from lookout.style.format.descriptions import get_composite_class_representations
from lookout.style.format.feature_extractor import FeatureExtractor
from lookout.style.format.feature_utils import VirtualNode
from lookout.style.format.model import FormatModel
from lookout.style.format.postprocess import filter_uast_breaking_preds
from lookout.style.format.utils import generate_comment, merge_dicts, prepare_files


def generate_report(y, y_pred, vnodes_y, n_files, target_names):
    """Generate report: classification report, confusion matrix, files with most errors."""
    with io.StringIO() as output:
        print("Classification report:\n" + classification_report(y, y_pred,
                                                                 target_names=target_names),
              file=output)
        print("Confusion matrix:\n" + str(confusion_matrix(y, y_pred)), file=output)

        # sort files by mispredictions and print them
        file_mispred = []
        for gt, pred, vn in zip(y, y_pred, vnodes_y):
            if gt != pred:
                file_mispred.append(vn.path)
        file_stat = Counter(file_mispred)

        to_show = file_stat.most_common()
        if n_files > 0:
            to_show = to_show[:n_files]

        print("Files with most errors:\n" + "\n".join(map(str, to_show)), file=output)
        return output.getvalue()


@profile
def quality_report(input_pattern: str, bblfsh: str, language: str, n_files: int, model_path: str
                   ) -> None:
    """Print several different reports for a given model on a given dataset."""
    log = logging.getLogger("quality_report")
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
    y_pred, _ = rules.predict(X=X, vnodes_y=vnodes_y, vnodes=vnodes, feature_extractor=fe)
    y, y_pred, vnodes_y, safe_preds = filter_uast_breaking_preds(
        y=y, y_pred=y_pred, vnodes_y=vnodes_y, vnodes=vnodes, files={f.path: f for f in files},
        feature_extractor=fe, client=client, vnode_parents=vnode_parents,
        node_parents=node_parents, log=log)
    target_names = get_composite_class_representations(fe)
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
        self.client = BblfshClient(self.config["bblfsh_address"])

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
                data_request_stub: DataStub, **data) -> List[Comment]:
        """
        Analyze changes between revisions and make quality report per file or for all changes.

        :param ptr_from: Git repository state pointer to the base revision.
        :param ptr_to: Git repository state pointer to the head revision.
        :param data_request_stub: Connection to the Lookout data retrieval service, not used.
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
                y_pred, rule_winners = rules.predict(X=X, vnodes_y=vnodes_y, vnodes=vnodes,
                                                     feature_extractor=fe)
                y, y_pred, vnodes_y, safe_preds = filter_uast_breaking_preds(
                    y=y, y_pred=y_pred, vnodes_y=vnodes_y, vnodes=vnodes, files={file.path: file},
                    feature_extractor=fe, client=self.client, vnode_parents=vnode_parents,
                    node_parents=node_parents, log=self.log)
                rule_winners = rule_winners[safe_preds]
                self.log.debug("y.shape %s" % y.shape)
                self.log.debug("len(vnodes_y) %s" % len(vnodes_y))
                self.log.debug("y_pred.shape %s" % y_pred.shape)
                if self.config["aggregate"] is True:
                    agg_y.append(y)
                    agg_vnodes_y.extend(vnodes_y)
                    agg_y_pred.append(y_pred)
                else:
                    target_names = get_composite_class_representations(fe)
                    assert len(y) == len(y_pred)

                    report = self.generate_report(
                        y=y, y_pred=y_pred, vnodes_y=vnodes_y, target_names=target_names,
                        config=self.config, model=self.model, feature_extractor=fe,
                        rule_winners=rule_winners
                    )
                    comments.append(generate_comment(filename=file.path, confidence=100,
                                                     line=0, text=report))
            model_report = self.generate_model_report(self.model, self.config, fe)
            if model_report:
                comments.append(generate_comment(filename=file.path, confidence=100,
                                                 line=0, text=model_report))
            if self.config["aggregate"]:
                agg_y = numpy.hstack(agg_y)
                target_names = get_composite_class_representations(fe)
                report = self.generate_report(
                    y=agg_y, y_pred=numpy.hstack(agg_y_pred), vnodes_y=agg_vnodes_y,
                    target_names=target_names, config=self.config, model=self.model,
                    feature_extractor=fe, rule_winners=rule_winners
                )
                comments.append(generate_comment(filename=file.path, confidence=100,
                                                 line=0, text=report))
        return comments

    @classmethod
    @with_uasts_and_contents
    def train(cls, ptr: ReferencePointer, config: MutableMapping[str, Any],
              data_request_stub: DataStub, **data) -> None:
        """
        Analyze a set of changes from one revision to another and make report for all files.

        :param ptr: Git repository state pointer.
        :param config: configuration dict.
        :param data: contains "files" - the list of files in the pointed state.
        :param data_request_stub: connection to the Lookout data retrieval service, not used.
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

            y_pred, rule_winners = rules.predict(X=X, vnodes_y=vnodes_y, vnodes=vnodes,
                                                 feature_extractor=fe)
            y, y_pred, vnodes_y, safe_preds = filter_uast_breaking_preds(
                y=y, y_pred=y_pred, vnodes_y=vnodes_y, vnodes=vnodes,
                files={f.path: f for f in filtered_files}, feature_extractor=fe,
                client=cls.client, vnode_parents=vnode_parents, node_parents=node_parents,
                log=cls.log)
            rule_winners = rule_winners[safe_preds]
            target_names = get_composite_class_representations(fe)
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
    lookout-sdk push ipv4://analyzer:port --git-dir /git/dir/ --from REV1 --to  REV2  \
    --config-json='{"style.format.analyzer.QualityReportAnalyzer":  \
    {"model_path": "/saved/model.asdf"}}'
    ```

    2.2) Quality report for changes per file:
    ```
    lookout-sdk review ipv4://analyzer:port --git-dir /git/dir/ --from REV1 --to  REV2  \
    --config-json='{"style.format.analyzer.QualityReportAnalyzer":  \
    {"model_path": "/saved/model.asdf"}}'
    ```

    2.3) Quality report for aggregated changes:
    ```
    lookout-sdk review ipv4://analyzer:port --git-dir /git/dir/ --from REV1 --to  REV2  \
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
