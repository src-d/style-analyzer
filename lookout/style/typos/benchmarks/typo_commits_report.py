"""Facilities to report the quality of a given model on a given dataset."""
import csv
import json
import logging
import os
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


from lookout.core.analyzer import ReferencePointer
from lookout.core.api.service_analyzer_pb2 import Comment
from lookout.core.api.service_data_pb2 import Change, File
from lookout.core.data_requests import DataService, request_files
import pandas

from lookout.style.cloner import Cloner
from lookout.style.format.benchmarks.quality_report import handle_input_arg
from lookout.style.format.utils import generate_comment
from lookout.style.reporter import Reporter
from lookout.style.typos import IdTyposAnalyzer
from lookout.style.typos.analyzer import TypoFix


class TyposAnalyzerSpy(IdTyposAnalyzer):
    """
    The Analyzer which returns fixes found by IdTyposAnalyzer for all file content as JSON \
    structures.
    """

    def run(self, ptr: ReferencePointer, data_service: DataService) -> Iterable[TypoFix]:
        """
        Run `generate_typos_fixes` for all lines and all files in ptr_from revision.

        :param ptr: Git repository state pointer to the revision that should be analyzed.
        :param data_service: Connection to the Lookout data retrieval service to get the files.
        :return: Generator of fixes for each file.
        """
        files = request_files(data_service.get_data(), ptr,
                              contents=True, uast=True, unicode=False)
        return self.generate_typos_fixes([
            Change(base=f, head=File(path=f.path, language=f.language)) for f in files])

    def analyze(self, ptr_from: ReferencePointer, ptr_to: ReferencePointer,
                data_service: DataService, **data) -> List[Comment]:
        """
        Return the list of `TypoFix`-es as Comments.

        `TypoFix`-es are generated with `run()` method.

        :param ptr_from: The Git revision to analyze.
        :param ptr_to: Not used. ptr_from is used for both model training and analysis.
        :param data_service: The channel to the data service in Lookout server to query for \
                             UASTs, file contents, etc.
        :param data: Extra data passed into the method. Used by the decorators to simplify \
                     the data retrieval.
        :return: List of Typo Fixes information in a JSON format required for further analysis.
        """
        return [generate_comment(
            filename=typo_fix.head_file.path,
            line=typo_fix.line_number,
            text=json.dumps(typo_fix._asdict()),
            confidence=100) for typo_fix in self.run(ptr_from, data_service)]


class TypoCommitsReporter(Reporter):
    """
    Report system for Typos Analyser.
    """

    inspected_analyzer_type = IdTyposAnalyzer

    @classmethod
    def get_report_names(cls) -> Tuple[str, ...]:
        """
        Get all available report names.

        :return: Tuple with report names.
        """
        return ("report", )

    def _generate_reports(self, dataset_row: Dict[str, Any], fixes: Sequence[TypoFix],
                          ) -> Dict[str, str]:
        """
        Generate reports for the dataset row.

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
        Generate commit dataset report for the dataset row.

        :param dataset_row: Dataset row which triggered the analyze method of the analyzer.
        :param fixes: List of `TypoFix`-es provided by the `TyposAnalyzerSpy.analyze()` method.
        :return: Dictionary with report names as keys and report string as values.
        """
        metrics = self.get_metrics_billet()

        correct_fix = dataset_row["correct id"]
        wrong_identifier = dataset_row["wrong id"]
        for fix in fixes:
            if fix.identifier == wrong_identifier:
                if correct_fix in fix.identifier_candidates:
                    metrics.detection_true_positive += 1
                    metrics.fix_accuracy += 1
            else:
                metrics.detection_false_positive += 1
        return json.dumps(metrics.to_dict())

    def _trigger_review_event(self, dataset_row: Dict[str, Any]) -> Sequence[TypoFix]:
        comments = self._analyzer_context_manager.review(
            dataset_row["commit"], "HEAD", git_dir=dataset_row["repo"], bblfsh=self._bblfsh,
            log_level="info", config_json=self._config)
        typo_fixes = []
        for comment in comments:
            typo_fix_dict = json.loads(comment.text)
            typo_fixes.append(TypoFix(head_file=typo_fix_dict["head_file"],
                                      line_number=typo_fix_dict["line_number"],
                                      candidates=typo_fix_dict["candidates"],
                                      token=typo_fix_dict["token"]))
        return typo_fixes

    def _finalize(self, reports: Iterable[Dict[str, str]]) -> Iterator[Dict[str, str]]:
        """
        Summarize all individual reports to the final one.

        :param reports: Reports generated by `TypoCommitsReporter.generate_commit_dataset_report()`
        :return: Summarized final report
        """
        # TODO(zurk): Add a proper report
        sum_metric = self.get_metrics_billet()
        for report in reports:
            sum_metric += pandas.Series(json.loads(report)["report"])
        yield json.dumps(sum_metric.to_dict())

    @staticmethod
    def get_metrics_billet() -> pandas.Series:
        """
        Generate pandas series with TypoCommitsReporter metrics.

        `detection_` prefix relates metric to typo detection and `fix_` to a metrics for founded
        typos. Support is a number of analyzed identifiers.
        """
        metrics = ((
            ("detection_true_positive", 0.0),
            ("detection_false_positive", 0.0),
            ("fix_accuracy", 0.0),
            ("support", 0.0),
        ))
        index, defaults = zip(*metrics)
        return pandas.Series(data=defaults, index=index)


def generate_typos_report_entry(dataset: str, output: str, bblfsh: str, config: dict,
                                database: Optional[str] = None, fs: Optional[str] = None,
                                repos_cache: Optional[str] = None) -> None:
    """
    Entry point for command line interface to generate typos quality report for the given data.

    :param dataset: csv file with commits to make report. Should contain repo, commit, file, \
                    line, wrong id and correct id columns.
    :param output: Directory where to save report.
    :param bblfsh: bblfsh address to use by lookout-sdk.
    :param config: config for IdTypoAnalyzer.
    :param database: sqlite3 database path to store the models. Temporary file is used if not set.
    :param fs: Model repository file system root. Temporary directory is used if not set.
    :param repos_cache: Directory where to download repositories from the dataset. It is strongly \
                        recommended to set this parameter if there are more then 20 repositories \
                        required for report generation. Temporary directory is used if not set.
    """
    log = logging.getLogger("TyposReporter")
    os.makedirs(output, exist_ok=True)
    dataset = list(csv.DictReader(handle_input_arg(dataset)))
    repositories = list(set(row["repo"] for row in dataset))
    log.info("Generate report for dataset with %d entries", len(dataset))
    repositories_path = Cloner(repos_cache).clone_repositories(repositories)
    local_dataset = []
    for entry in dataset:
        if entry["repo"] in repositories_path:
            local_dataset.append(dict(entry))
            local_dataset[-1]["repo"] = repositories_path[entry["repo"]]
    with TypoCommitsReporter(config, bblfsh, database, fs) as reporter:
        reports = list(reporter.run(local_dataset))
        for report in reports:
            for report_name in reporter.get_report_names():
                with open(os.path.join(output, "%s_%s_report.md" % (
                        report["repo"], report_name)), "w") as f:
                    f.write(report[report_name])
