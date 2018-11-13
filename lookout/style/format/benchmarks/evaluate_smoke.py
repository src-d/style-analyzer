"""Module for Smoke dataset evaluation."""
import csv
from difflib import SequenceMatcher
import json
import logging
from pathlib import Path
import tempfile
from typing import Any, List, Mapping

from bblfsh import BblfshClient

from lookout.core.analyzer import ReferencePointer
from lookout.core.api.service_analyzer_pb2 import Comment
from lookout.core.api.service_data_pb2_grpc import DataStub
from lookout.core.data_requests import with_changed_uasts_and_contents
from lookout.core.lib import files_by_language, filter_files, find_new_lines
from lookout.core.tests import server
from lookout.style.format.analyzer import FormatAnalyzer
from lookout.style.format.code_generator import CodeGenerator
from lookout.style.format.feature_extractor import FeatureExtractor
from lookout.style.format.model import FormatModel
from lookout.style.format.tests.test_analyzer_integration import TestAnalyzer


def _loss(head_lines, correct_lines, predicted_lines):
    def edit_distance(a, b):
        err = 0
        matcher = SequenceMatcher(a=a, b=b)
        matcher.ratio()
        actions = frozenset(("delete", "insert", "replace"))
        for action, _, _, _, _ in matcher.get_opcodes():
            if action == "equal":
                continue
            if action in actions:
                err += 1
        return err

    return edit_distance(predicted_lines, correct_lines) / \
        edit_distance(head_lines, correct_lines)


class SmokeEvalFormatAnalyzer(FormatAnalyzer):
    """
    Analyzer for Smoke dataset evaluation.
    """

    REPORT_COLNAMES = ["repo", "filepath", "style", "loss"]

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
        self.report = None

    def _dump_report(self, outputpath):
        with open(outputpath, "a") as f:
            writer = csv.DictWriter(f, fieldnames=self.REPORT_COLNAMES)
            writer.writerows(self.report)

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
        self.report = []
        log = self.log
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
                try:
                    prev_file = base_files_by_lang[lang][file.path]
                except KeyError:
                    continue
                else:
                    lines = [find_new_lines(prev_file, file)]
                fe = FeatureExtractor(language=lang, **rules.origin_config["feature_extractor"])
                res = fe.extract_features([file], lines)
                if res is None:
                    log.warning("Failed to parse %s", file.path)
                    continue
                X, y, vnodes_y, vnodes, vnodes_parents, parents = res
                y_pred, _, _ = rules.predict(X=X, y=y, vnodes_y=vnodes_y, vnodes=vnodes,
                                             files={file.path: file}, feature_extractor=fe,
                                             client=self.client, vnodes_parents=vnodes_parents,
                                             parents=parents, return_all_preds=True)
                assert len(y) == len(y_pred)

                correct_lines = prev_file.content.decode("utf-8", "replace").splitlines()
                head_lines = file.content.decode("utf-8", "replace").splitlines()
                code_generator = CodeGenerator(fe, skip_errors=True)
                predicted_lines = code_generator.generate(
                    vnodes_y=vnodes_y, y_pred=y_pred, vnodes=vnodes)
                file_loss = _loss(head_lines, correct_lines, predicted_lines)
                self.log.debug("Loss %d on file %s in the repo %s" % (
                    file_loss, file.path, self.config["repo_name"]))
                self.report.append({"repo": self.config["repo_name"],
                                    "filepath": file.path,
                                    "style": self.config["style_name"],
                                    "loss": file_loss})
        self._dump_report(self.config["report_path"])
        return []


analyzer_class = SmokeEvalFormatAnalyzer


def report_summary(reportpath: str) -> None:
    """
    Print to logs report summary.

    :param reportpath: path to report generated by `evaluate_smoke_entry`
    """
    log = logging.getLogger("report_summary")
    no_results = 0
    no_comments = 0
    rest_loss = 0
    rest_support = 0
    cases_number = 0
    with open(str(reportpath)) as index:
        reader = csv.DictReader(index, fieldnames=SmokeEvalFormatAnalyzer.REPORT_COLNAMES)
        for row in reader:
            cases_number += 1
            loss = float(row["loss"])
            if loss < 0:
                no_results += 1
            elif loss == 1:
                no_comments += 1
            else:
                rest_loss += loss
                rest_support += 1
    log.info("Cases number: %d.", cases_number)
    log.info("Code generation not implemented for %d cases.", no_results)
    log.info("Failed to fix any style violation for %d cases.", no_comments)
    log.info("Loss value for the rest cases is %.4f with support %d.",
             rest_loss / rest_support, rest_support)


def evaluate_smoke_entry(inputpath: str, reportpath: str) -> None:
    """
    CLI entry point.
    """
    log = logging.getLogger("evaluate_smoke")
    port = server.find_port()
    db = tempfile.NamedTemporaryFile(dir=inputpath)
    with tempfile.TemporaryDirectory(dir=inputpath) as fs:
        context_manager = TestAnalyzer(
            port=port, db=db.name, fs=fs,
            analyzer="lookout.style.format.benchmarks.evaluate_smoke")
        with context_manager:
            inputpath = Path(inputpath)
            if not server.file.exists():
                server.fetch()
            index_file = inputpath / "index.csv"
            with open(str(index_file)) as index:
                reader = csv.DictReader(index)
                for row in reader:
                    repopath = inputpath / row["repo"]
                    config_json = {
                        analyzer_class.name: {
                            "repo_name": row["repo"],
                            "style_name": row["style"],
                            "report_path": reportpath
                        }
                    }
                    server.run("push", fr=row["from"], to=row["to"], port=port,
                               git_dir=str(repopath), )
                    server.run("review", fr=row["from"], to=row["to"], port=port,
                               git_dir=str(repopath),
                               config_json=json.dumps(config_json))
            log.info("Quality report saved to %s", reportpath)

    report_summary(reportpath)
