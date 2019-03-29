"""Facilities to report the quality of a given model on a given dataset."""
import csv
import json
import logging
from operator import itemgetter
import os
import time
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from lookout.core.analyzer import ReferencePointer, UnicodeChange
from lookout.core.api.service_analyzer_pb2 import Comment
from lookout.core.api.service_data_pb2 import File
from lookout.core.data_requests import DataService, request_files
import pandas
from tabulate import tabulate

from lookout.style.cloner import Cloner
from lookout.style.common import load_jinja2_template, merge_dicts
from lookout.style.format.benchmarks.quality_report import handle_input_arg
from lookout.style.format.utils import generate_comment
from lookout.style.reporter import Reporter
from lookout.style.typos import IdTyposAnalyzer
from lookout.style.typos.analyzer import TypoFix
from lookout.style.typos.utils import TEMPLATE_DIR


class IdTyposAnalyzerSpy(IdTyposAnalyzer):
    """
    The Analyzer which returns fixes found by IdTyposAnalyzer as JSON structures.

    Note that all lines in the head revision (`ptr_to`) is analyzed, not only changed lines.
    Thus the result does not depend on base revision (`ptr_from`).
    """

    def run(self, ptr: ReferencePointer, data_service: DataService) -> Iterable[TypoFix]:
        """
        Run `generate_typos_fixes` for all lines and all files in `ptr_from` revision.

        :param ptr: Git repository state pointer to the revision that should be analyzed.
        :param data_service: Connection to the Lookout data retrieval service to get the files.
        :return: Generator of fixes for each file.
        """
        for file in request_files(data_service.get_data(), ptr, contents=True, uast=True,
                                  unicode=False):
            if file.path == self.config["filepath_to_analyze"]:
                break
        else:
            raise ValueError("No such file %s in %s" % (self.config["filepath_to_analyze"], ptr))

        typos_fixes = list(self.generate_typos_fixes([
            UnicodeChange(head=file, base=File(path=file.path, language=file.language))]))
        if typos_fixes:
            return typos_fixes
        identifiers_number = len(self._get_identifiers(file.uast, []))
        if not identifiers_number:
            raise ValueError("No identifiers for file %s in %s" % (
                self.config["filepath_to_analyze"], ptr))
        return [TypoFix(content=file.content.decode("utf-8", "replace"),
                        path=file.path, line_number=0, identifier="", candidates=[],
                        identifiers_number=identifiers_number)]

    def analyze(self, ptr_from: ReferencePointer, ptr_to: ReferencePointer,
                data_service: DataService, **data) -> List[Comment]:
        """
        Extract the list of `TypoFix`-es as `Comment`-s.

        `TypoFix`-es are generated in `run()`.

        :param ptr_from: The Git revision to analyze.
        :param ptr_to: Not used. ptr_from is used for both model training and analysis.
        :param data_service: The channel to the data service in Lookout server to query for \
                             UASTs, file contents, etc.
        :param data: Extra data passed into the method. Used by the decorators to simplify \
                     the data retrieval.
        :return: List of `Comment`-s with `TypoFix` in JSON format.
        """
        return [generate_comment(
            filename=typo_fix.path,
            line=typo_fix.line_number,
            text=json.dumps(typo_fix._asdict()),
            confidence=100) for typo_fix in self.run(ptr_to, data_service)]


class TypoCommitsReporter(Reporter):
    """
    Report system for Typos Analyser.
    """

    inspected_analyzer_type = IdTyposAnalyzerSpy
    report_template_path = os.path.join(TEMPLATE_DIR, "commits_with_typo_dataset_report.jinja2")

    @classmethod
    def get_report_names(cls) -> Tuple[str, ...]:
        """
        Get all the available report names.

        :return: Tuple with report names.
        """
        return ("report", )

    def _generate_reports(self, dataset_row: Dict[str, Any], fixes: Sequence[TypoFix],
                          ) -> Dict[str, str]:
        """
        Generate reports for a dataset row.

        :param dataset_row: Dataset row which triggered the analyze method of the analyzer.
        :param fixes: List of `TypoFix`-es provided by the `TyposAnalyzerSpy.analyze()` method.
        :return: Dictionary with report names as keys and report string as values.
        """
        return {
            "report": self.generate_commit_dataset_report(dataset_row, fixes),
        }

    def generate_commit_dataset_report(self, dataset_row: Dict[str, Any],
                                       fixes: Sequence[TypoFix]) -> str:
        """
        Generate the report for a dataset row.

        :param dataset_row: Dataset row which triggered the analyze method of the analyzer.
        :param fixes: List of `TypoFix`-es provided by the `TyposAnalyzerSpy.analyze()` method.
        :return: Dictionary with report names as keys and report string as values.
        """
        metrics = self.get_metrics_stub()
        metrics.review_time = self._review_time
        if not fixes:
            raise ValueError("Should be at least one fix.")
        correct_fix = dataset_row["correct_id"]
        wrong_identifier = dataset_row["wrong_id"]
        for fix in fixes:
            if fix.identifier == "":
                # no fixes where suggested
                assert len(fixes) == 1
                break
            candidates = [candidate[0] for candidate in fix.candidates]
            best_candidate = max(fix.candidates, key=itemgetter(1))
            if fix.identifier == wrong_identifier:
                if correct_fix in candidates:
                    metrics.detection_true_positive += 1
                    metrics.top3_fix_accuracy += 1
                if correct_fix == best_candidate:
                    metrics.fix_accuracy += 1
            else:
                metrics.detection_false_positive += 1

        metrics.support = fixes[0].identifiers_number
        # Because there is one typo per commit
        metrics.detection_false_negatives = 1 - metrics.detection_true_positive
        return json.dumps(metrics.to_dict())

    def _trigger_review_event(self, dataset_row: Dict[str, Any]) -> Sequence[TypoFix]:
        config = merge_dicts(
            self._config if self._config is not None else {},
            {IdTyposAnalyzerSpy.name: {"filepath_to_analyze": dataset_row["file"]}})
        start_time = time.perf_counter()
        comments = self._analyzer_context_manager.review(
            dataset_row["commit_typo"], "HEAD", git_dir=dataset_row["repo_path"],
            bblfsh=self._bblfsh, log_level="info", config_json=config)
        self._review_time = time.perf_counter() - start_time
        return [TypoFix(**json.loads(comment.text)) for comment in comments]

    def _finalize(self, reports: Iterable[Dict[str, str]]) -> Iterator[Dict[str, str]]:
        """
        Summarize all individual reports.

        :param reports: Reports generated by `TypoCommitsReporter.generate_commit_dataset_report()`
        :return: Summarized final report
        """
        scores = self.get_metrics_stub()
        reports = list(reports)
        for report in reports:
            scores += pandas.Series(json.loads(report["report"]))
        scores.detection_precision = scores.detection_true_positive / (
            scores.detection_true_positive + scores.detection_false_positive)
        scores.detection_recall = scores.detection_true_positive / (
            scores.detection_true_positive + scores.detection_false_negatives
        )
        scores.fix_accuracy = scores.fix_accuracy / len(reports)
        scores.top3_fix_accuracy = scores.top3_fix_accuracy / len(reports)
        scores.review_time = scores.review_time / len(reports)

        template = load_jinja2_template(self.report_template_path)
        report = template.render(scores=scores, commit=self._get_commit(),
                                 package_version=self._get_package_version(),
                                 failures=self._failures, tabulate=tabulate)

        yield {"report": report}

    @staticmethod
    def get_metrics_stub() -> pandas.Series:
        """
        Generate pandas series with `TypoCommitsReporter`'s  metrics.

        `detection_` prefix relates metric to typo detection and `fix_` to a metrics for founded
        typos. Support is a number of analyzed identifiers.
        """
        metrics = (
            ("detection_true_positive", 0.0),
            ("detection_false_positive", 0.0),
            ("detection_false_negatives", 0.0),
            ("detection_precision", 0.0),
            ("detection_recall", 0.0),
            ("top3_fix_accuracy", 0.0),
            ("fix_accuracy", 0.0),
            ("support", 0.0),
            ("review_time", 0.0),
        )
        index, defaults = zip(*metrics)
        return pandas.Series(data=defaults, index=index)

    @staticmethod
    def _get_row_repr(dataset_row: Dict[str, Any]) -> str:
        """Convert dataset row to its representation for logging purposes."""
        return dataset_row["repo"]


def generate_typos_report_entry(dataset: str, output: str, bblfsh: str, config: dict,
                                database: Optional[str] = None, fs: Optional[str] = None,
                                repos_cache: Optional[str] = None) -> None:
    """
    Entry point for the command line interface to generate typos quality report.

    :param dataset: csv file with commits. Must contain wrong_id, correct_id, file, line,
                    commit_fix, repo, commit_typo.
    :param output: Directory where to save the report.
    :param bblfsh: bblfsh address to use for `lookout-sdk`.
    :param config: config for IdTypoAnalyzer.
    :param database: sqlite3 database path to store the models. A temporary file is used if not \
                     set.
    :param fs: Model repository file system root. Temporary directory is used if not set.
    :param repos_cache: Directory where to download repositories from the dataset. It is strongly \
                        recommended to set this parameter if there are more than 20 repositories \
                        in the dataset. Temporary directory is used if not set.
    """
    log = logging.getLogger("TyposReporter")
    os.makedirs(output, exist_ok=True)
    dataset = list(csv.DictReader(handle_input_arg(dataset)))
    repositories = sorted(set(row["repo"] for row in dataset))

    log.info("Generate report for dataset with %d entries", len(dataset))
    repositories_path = Cloner(repos_cache).clone(repositories)
    local_dataset = []
    for entry in dataset:
        if entry["repo"] in repositories_path:
            local_dataset.append(dict(entry))
            local_dataset[-1]["repo_path"] = repositories_path[entry["repo"]]
    with TypoCommitsReporter(config, bblfsh, database, fs) as reporter:
        reports = list(reporter.run(local_dataset))
        for report in reports:
            for report_name in reporter.get_report_names():
                report_path = os.path.join(output, "%s_report.md" % report_name)
                with open(report_path, "w") as f:
                    f.write(report[report_name])
                log.info("Report %s is saved to %s", report_name, report_path)
    log.info("Done.")
