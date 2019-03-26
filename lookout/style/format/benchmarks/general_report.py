"""Facilities to report the quality of a given model on a given dataset."""
from collections import Counter, OrderedDict
import glob
from itertools import chain
import json
import logging
import os
from typing import Any, Dict, Iterable, List, Mapping, NamedTuple, Optional, Sequence, Tuple, \
    Type, Union

import bblfsh
from lookout.core.analyzer import ReferencePointer
from lookout.core.api.service_analyzer_pb2 import Comment
from lookout.core.api.service_data_pb2 import Change, File
from lookout.core.api.service_data_pb2_grpc import DataStub
from lookout.core.data_requests import DataService, request_files
from lookout.core.lib import parse_files
import numpy
from sklearn.exceptions import NotFittedError

from lookout.style.common import load_jinja2_template, merge_dicts
from lookout.style.format import TEMPLATES_ROOT
from lookout.style.format.analyzer import FileFix, FormatAnalyzer
from lookout.style.format.benchmarks.time_profile import profile
from lookout.style.format.descriptions import describe_rule
from lookout.style.format.feature_extractor import FeatureExtractor
from lookout.style.format.model import FormatModel
from lookout.style.format.utils import generate_comment, get_classification_report
from lookout.style.format.virtual_node import VirtualNode


def generate_quality_report(language: str, report: Mapping[str, Any], ptr: ReferencePointer,
                            vnodes: Sequence[VirtualNode], max_files: int, name: str) -> str:
    """Generate report: classification report, confusion matrix, files with most errors."""
    avg_keys = {"macro avg", "micro avg", "weighted avg"}
    sorted_report = OrderedDict((key, report["report"][key])
                                for key in sorted(report["report"],
                                                  key=lambda k: -report["report"][k]["support"])
                                if key not in avg_keys)
    for key in avg_keys:
        sorted_report[key] = report["report"][key]
    # sort files by mispredictions
    file_mispred = []
    for vnode in vnodes:
        if vnode.y != getattr(vnode, "y_old", vnode.y):
            file_mispred.append(vnode.path)
    file_stat = Counter(file_mispred)
    to_show = file_stat.most_common()
    if max_files > 0:
        to_show = to_show[:max_files]

    template = load_jinja2_template(os.path.join(TEMPLATES_ROOT, "quality_report.md.jinja2"))
    # TODO(vmarkovtsev): move all the logic inside the template
    res = template.render(language=language, ptr=ptr, conf_mat=report["confusion_matrix"],
                          target_names=report["target_names"], files=to_show,
                          cl_report=sorted_report, ppcr=report["ppcr"],
                          cl_report_full=report["report_full"], name=name)
    return res


def generate_model_report(model: FormatModel, analyze_config: Dict[str, Any],
                          languages: Optional[Union[str, Iterable[str]]] = None) -> str:
    """
    Generate report about model - description for each rule, min/max support, min/max confidence.

    :param model: trained format model.
    :param analyze_config: config that is used at the analysis stage. It is needed to calculate \
                           the real number of enabled rules.
    :param languages: Languages for which report should be created. You can specify one \
                      language as string, several as list of strings or None for all languages in \
                      the model.
    :return: report in str format.
    """
    languages = languages if languages is not None else model.languages
    languages = languages if isinstance(languages, Iterable) else [languages]
    for language in languages:
        if language not in model:
            raise NotFittedError(language)
    template = load_jinja2_template(os.path.join(TEMPLATES_ROOT, "model_report.md.jinja2"))
    return template.render(model=model, languages=languages, analyze_config=analyze_config,
                           FeatureExtractor=FeatureExtractor, describe_rule=describe_rule)


class FakeStub:
    """Data service stub mock which returns the list of bound files and changes."""

    def __init__(self, files: Iterable[File], changes: Iterable[Change]):
        """Initialize a new instance of FakeStub."""
        self.files = files
        self.changes = changes

    def GetFiles(self, *args, **kwargs):
        """Return the list of File-s."""
        return self.files

    def GetChanges(self, _):
        """Return the list of Change-s."""
        return self.changes


class FakeDataService:
    """Data service mock which returns the list of bound files and changes through FakeStub."""

    def __init__(self, bblfsh_client: bblfsh.BblfshClient, files: Iterable[File],
                 changes: Iterable[Change]):
        """Initialize a new instance of FakeDataService."""
        self.bblfsh_client = bblfsh_client
        self.files = files
        self.changes = changes

    def get_bblfsh(self) -> bblfsh.aliases.ProtocolServiceStub:
        """Return the Babelfish gRPC stub."""
        return self.bblfsh_client._stub

    def get_data(self) -> DataStub:
        """Return the FakeStub to pretend that the server is running."""
        return FakeStub(self.files, self.changes)

    def check_bblfsh_driver_versions(self, versions: Iterable[str]):
        """Do not care about the versions here."""
        pass


def analyze_files(analyzer_type: Type[FormatAnalyzer], config: dict, model_path: str,
                  language: str, bblfsh_addr: str, input_pattern: str, log: logging.Logger,
                  ) -> List[Comment]:
    """Run the model, record the fixes for each file and return them."""
    class FakePointer:
        def to_pb(self):
            return None

    model = FormatModel().load(model_path)
    if language not in model:
        raise NotFittedError()
    rules = model[language]
    client = bblfsh.BblfshClient(bblfsh_addr)
    files = parse_files(filepaths=glob.glob(input_pattern, recursive=True),
                        line_length_limit=rules.origin_config["line_length_limit"],
                        overall_size_limit=rules.origin_config["overall_size_limit"],
                        client=client, language=language, log=log)
    log.info("Model parameters: %s" % rules.origin_config)
    log.info("Rules stats: %s" % rules)
    log.info("Number of files: %s" % (len(files)))
    return analyzer_type(model, input_pattern, config).analyze(
        FakePointer(), None, data_service=FakeDataService(client, files, []))


@profile
def print_reports(input_pattern: str, bblfsh_addr: str, language: str, model_path: str,
                  config: Union[str, dict] = "{}") -> None:
    """Print reports for a given model on a given dataset."""
    log = logging.getLogger("quality_report")
    config = config if isinstance(config, dict) else json.loads(config)
    for report in analyze_files(
            QualityReportAnalyzer, config, model_path, language, bblfsh_addr, input_pattern, log):
        print(report.text)


class FormatAnalyzerSpy(FormatAnalyzer):
    """
    The class which runs the FormatAnalyzer and returns the found fixes.
    """

    Changes = NamedTuple("Changes", (("base", File), ("head", File)))

    def run(self, ptr_from: ReferencePointer,
            data_service_head: DataService,
            data_service_base: Optional[DataService] = None) -> Iterable[FileFix]:
        """
        Run `generate_file_fixes` for all files in ptr_from revision.

        :param ptr_from: Git repository state pointer to the base revision.
        :param data_service_head: Connection to the Lookout data retrieval service to get \
                                the new files.
        :param data_service_base: Connection to the Lookout data retrieval service to get \
                                  the initial files. If it is None, we assume the empty contents.
        :return: Generator of fixes for each file.
        """
        files_head = list(request_files(
            data_service_head.get_data(), ptr_from, contents=True, uast=True, unicode=True))

        if data_service_base is not None:
            files_base = list(request_files(
                data_service_base.get_data(), ptr_from, contents=True, uast=True, unicode=True))
        else:
            files_base = [File(path=f.path) for f in files_head]
        return self.generate_file_fixes(
            data_service_head, [self.Changes(f1, f2) for f1, f2 in zip(files_base, files_head)])

    @classmethod
    def train(cls, ptr: ReferencePointer, config: Mapping[str, Any], data_service: DataService,
              **data) -> FormatModel:
        """
        Train a model given the files available or load the existing model.

        If you set config["model"] to path in the file system model will be loaded otherwise
        a model is trained in a regular way.

        :param ptr: Git repository state pointer.
        :param config: Configuration dict.
        :param data: Contains "files" - the list of files in the pointed state.
        :param data_service: Connection to the Lookout data retrieval service.
        :return: FormatModel containing the learned rules, per language.
        """
        return FormatModel().load(config["model"]) if "model" in config else \
            super().train(ptr, config, data_service)


class ReportAnalyzer(FormatAnalyzerSpy):
    """
    Base class for different kind of reports.

    * analyze - generate report for all files. If you want only aggregated report set aggregate
    flag to True in analyze config.
    * train - train or load the model.

    Child classes are required to implement 2 methods:
    * generate_report
    * generate_model_report (optional - by default it will return empty string)
    """

    default_config = merge_dicts(FormatAnalyzer.default_config,
                                 {"aggregate": False})

    @classmethod
    def get_report_names(cls) -> Tuple[str, ...]:
        """
        Get all available report names.

        :return: List of report names.
        """
        raise NotImplementedError()

    def generate_reports(self, fixes: Iterable[FileFix]) -> Dict[str, str]:
        """
        General function to generate reports.

        :param fixes: List of fixes per file or for all files if config["aggregate"] is True.
        :return: Dictionary with report names as keys and report string as values.
        """
        raise NotImplementedError()

    def analyze(self, ptr_from: ReferencePointer, ptr_to: ReferencePointer,
                data_service: DataService, **data) -> List[Comment]:
        """
        Analyze ptr_from revision and generate reports for all files in it.

        If you want to get an aggregated report set aggregate flag to True in analyze config.

        :param ptr_from: Git repository state pointer to the base revision.
        :param ptr_to: Git repository state pointer to the head revision. Not used.
        :param data_service: Connection to the Lookout data retrieval service.
        :param data: Contains "files" - the list of changes in the pointed state.
        :return: List of comments.
        """
        def convert_fixes_to_report_comments(fixes: List[FileFix], filepath: str):
            for report in self.generate_reports(fixes=fixes).values():
                yield generate_comment(filename=filepath, line=0, confidence=100, text=report)

        comments = []
        if not self.config["aggregate"]:
            for fix in self.run(ptr_from, data_service):
                filepath = fix.head_file.path
                if fix.error:
                    continue
                comments.extend(convert_fixes_to_report_comments([fix], filepath))
        else:
            comments.extend(
                convert_fixes_to_report_comments(
                    [fix for fix in self.run(ptr_from, data_service) if not fix.error], ""))
        return comments


class QualityReportAnalyzer(ReportAnalyzer):
    """
    Generate basic quality reports for the model.

    * analyze - generate report for all files. If you want only aggregated report set aggregate
    flag to True in analyze config.
    * train - train or load the model.

    It is possible to run this analyzer independently and query it with lookout-sdk.
    If you want to use pretrained model it is possible to specify it in config, for example:
    `--config-json='{"style.format.analyzer.FormatAnalyzer": {"model": "/saved/model.asdf"}}`
    Otherwise model will be trained with `FormatAnalyzer.train()`

    Usage examples:
    1) Launch analyzer: `analyzer run lookout.style.format.quality_report_analyzer -c config.yml`
    2) Query analyzer
    2.1) Get one quality report per file for pretrained model /saved/model.asdf:
    ```
    lookout-sdk review ipv4://localhost:2000 --git-dir /git/dir/ --from REV1 --to REV2 \
    --config-json='{"style.format.analyzer.FormatAnalyzer": {"model": "/saved/model.asdf"}}'
    ```
    2.2) Get aggregated quality report for all files without pretrained model
    ```
    lookout-sdk review ipv4://localhost:2000 --git-dir /git/dir/ --from REV1 --to REV2 \
    --config-json='{"style.format.analyzer.FormatAnalyzer": {"aggregate": true}}'
    ```
    """

    version = 1
    description = "Source code formatting quality report generator: " \
                  "whitespace, new lines, quotes, etc."
    default_config = merge_dicts(ReportAnalyzer.default_config,
                                 {"max_files": 10,
                                  "train": {"language_defaults": {"test_dataset_ratio": 0.2}},
                                  })

    @classmethod
    def get_report_names(cls) -> Tuple[str, str, str]:
        """
        Get all available report names.

        :return: Tuple with report names.
        """
        return "model", "train", "test"

    def generate_reports(self, fixes: Iterable[FileFix]) -> Dict[str, str]:
        """
        Generate model train and test reports.

        Model report generated only if config["aggregate"] is True.

        :param fixes: List of fixes per file or for all files if config["aggregate"] is True.
        :return: Ordered dictionary with report names as keys and report string as values.
        """
        reports = OrderedDict()  # to keep reports order.
        if self.config["aggregate"]:
            reports["model"] = self.generate_model_report()
        try:
            reports["train"] = self.generate_train_report(fixes)
        except ValueError as e:
            self._log.warning("Train report generation failed. %s", e.args[0])
        reports["test"] = self.generate_test_report()
        return reports

    def generate_model_report(self) -> str:
        """
        Generate report about the trained model.

        :return: report.
        """
        return generate_model_report(model=self.model, analyze_config=self.analyze_config)

    def generate_train_report(self, fixes: Iterable[FileFix]) -> str:
        """
        Generate train report: classification report, confusion matrix, files with most errors.

        :return: report.
        """
        fixes = list(fixes)
        if not fixes:
            raise ValueError("There are no fixes for %s" % self.model.dump())
        vnodes = chain.from_iterable(fix.file_vnodes for fix in fixes)
        ys = numpy.hstack(fix.y for fix in fixes)
        y_pred_pure = numpy.hstack(fix.y_pred_pure for fix in fixes)
        report = get_classification_report(
            y_pred_pure, ys, fixes[0].feature_extractor.composite_class_representations)
        # FIXME(vmarkovtsev): we are taking the first fix here which does not work for >1 language
        return generate_quality_report(
            fixes[0].language, report, self.model.ptr, vnodes, self.config["max_files"],
            name="Train")

    def generate_test_report(self) -> str:
        """
        Generate report on the test dataset.

        :return: Report.
        """
        for lang in self.model:
            classification_report = self.model[lang].classification_report["test"]
            if not classification_report:
                raise ValueError(
                    "Test classification report is unavailable for language %s. Skipping." % lang)
            return generate_quality_report(lang, classification_report, self.model.ptr, [], 0,
                                           name="Test")
