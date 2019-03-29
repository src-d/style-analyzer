"""Measure quality on several top repositories."""
from collections import OrderedDict
import csv
import functools
import json
import logging
import logging.handlers
import os
import subprocess
import sys
import tempfile
from typing import Dict, Iterable, Iterator, Mapping, NamedTuple, Optional, Sequence, Union

from dulwich import porcelain
from lookout.core.helpers.analyzer_context_manager import AnalyzerContextManager
import numpy
from smart_open import smart_open
from tabulate import tabulate

from lookout.style.common import huge_progress_bar, merge_dicts
from lookout.style.format.benchmarks.general_report import QualityReportAnalyzer
from lookout.style.format.feature_extractor import FeatureExtractor


# TODO: add to ./benchmarks/data/quality_report_repos.csv after bblfsh python client v3 is released and we use it  # noqa: E501
# https://github.com/vuejs/vue,b7105ae8c9093e36ec89a470caa3b78bda3ef467,db1d0474997e9e60b8b0d39a3b7c2af55cfd0d4a",  # noqa: E501
# https://github.com/vuejs/vuex,2e62705d4bce4ebcb8eca23df8c7b849125fc565,1ac16a95c574f6b1386016fb6d4f00cfd2ee1d60",  # noqa: E501

FLOAT_PRECISION = ".3f"


def get_repo_name(url: str) -> str:
    """
    Extract name of repository from URL.

    :param url: URL for repository.
    :return: name of repository.
    """
    return url.split("/")[-1]


def ensure_repo(repository: str, storage_dir: str) -> str:
    """
    Clones repository if it is an url and returns repository path.

    :param repository: Repository url or directory in the file system.
    :param storage_dir: Clone repository to this directory if it is an url.
    :return: Repository path.
    """
    if os.path.exists(repository):
        return repository
    # clone repository
    git_dir = os.path.join(storage_dir, get_repo_name(repository))  # location for code
    porcelain.clone(repository, git_dir)
    return git_dir


class RestartReport(ValueError):
    """Exception raises if report collection should be restarted."""


def measure_quality(repository: str, from_commit: str, to_commit: str,
                    context: AnalyzerContextManager, config: dict, bblfsh: Optional[str],
                    vnodes_expected_number: Optional[int], restarts: int=3) -> Dict[str, str]:
    """
    Generate `QualityReport` for a repository. If it fails it returns empty reports.

    :param repository: URL of repository.
    :param from_commit: Hash of the base commit.
    :param to_commit: Hash of the head commit.
    :param context: LookoutSDK instance to query analyzer.
    :param config: config for FormatAnalyzer.
    :param bblfsh: Babelfish server address to use. Specify None to use the default value.
    :param vnodes_expected_number: Specify number for expected number of vnodes if known. \
                                   report collection will be restarted if number of extracted \
                                   vnodes does not match.
    :param restarts: Number of restarts if number of extracted vnodes does not match.
    :return: Dictionary with all QualityReport reports.
    """
    log = logging.getLogger("QualityAnalyzer")

    # This dirty hack should be removed as soon as
    # https://github.com/src-d/style-analyzer/issues/557 resolved.
    sum_vnodes_number = 0
    call_numbers = 0

    _convert_files_to_xy_backup = FeatureExtractor._convert_files_to_xy

    def _convert_files_to_xy(self, parsed_files):
        nonlocal sum_vnodes_number, call_numbers
        call_numbers += 1
        sum_vnodes_number += sum(len(vn) for vn, _, _ in parsed_files)
        # sum_vnodes_number + 1 because of whatever reason if you extract test and train
        # separately you have -1 vnode
        # TODO (zurk): investigate ^
        if call_numbers == 2 and sum_vnodes_number + 1 != vnodes_expected_number:
            raise RestartReport("VNodes number does not match to expected: %d != %d:" % (
                sum_vnodes_number, vnodes_expected_number))
        log.info("VNodes number match to expected %d. ", vnodes_expected_number)
        return _convert_files_to_xy_backup(self, parsed_files)

    reports = {}

    def capture_reports(func):
        @functools.wraps(func)
        def wrapped_capture_quality_reports(*args, **kwargs):
            nonlocal reports
            if reports:
                raise RuntimeError("generate_reports should be called only one time.")
            result = func(*args, **kwargs)
            reports = result
            return result
        wrapped_capture_quality_reports.original = func
        return wrapped_capture_quality_reports

    try:
        QualityReportAnalyzer.generate_reports = \
            capture_reports(QualityReportAnalyzer.generate_reports)
        if vnodes_expected_number:
            log.info("Vnodes expected number is equal to %d", vnodes_expected_number)
            FeatureExtractor._convert_files_to_xy = _convert_files_to_xy
        with tempfile.TemporaryDirectory(prefix="top-repos-quality-repos-") as tmpdirname:
            git_dir = ensure_repo(repository, tmpdirname)
            for attempt_number in range(restarts):
                sum_vnodes_number = -1
                try:
                    context.push(fr=from_commit, to=to_commit, git_dir=git_dir,
                                 log_level="warning", bblfsh=bblfsh, config_json=config)
                    break
                except subprocess.CalledProcessError:
                    # Assume that we failed because VNodes number does not match to expected one
                    log.warning("%d/%d try to train the model failed.", attempt_number, restarts)
            else:
                raise RuntimeError("Run out of %d attempts. Failed to train proper model for %s." %
                                   (restarts, repository))
            context.review(fr=from_commit, to=to_commit, git_dir=git_dir, log_level="warning",
                           bblfsh=bblfsh, config_json=config)
    finally:
        QualityReportAnalyzer.generate_reports = QualityReportAnalyzer.generate_reports.original
        if vnodes_expected_number:
            FeatureExtractor._convert_files_to_xy = _convert_files_to_xy_backup
    return reports


def calc_weighted_avg(arr: Union[Sequence[Sequence], numpy.ndarray], col: int,
                      weight_col: int = 5) -> float:
    """Calculate average value in `col` weighted by column `weight_col`."""
    arr = numpy.array(arr)
    weights_ = arr[:, weight_col].astype(float)
    col_ = arr[:, col].astype(float)
    numerator = (col_ * weights_).sum()
    denominator = weights_.sum()
    if denominator == 0:
        return 1
    return numerator / denominator


def calc_avg(arr: Union[Sequence[Sequence], numpy.ndarray], col: int) -> float:
    """Calculate average value in `col`."""
    return numpy.array(arr)[:, col].astype(float).sum() / len(arr)


Metrics = NamedTuple("Metrics", (
    ("precision", float),
    ("recall", float),
    ("full_recall", float),
    ("f1", float),
    ("full_f1", float),
    ("ppcr", float),
    ("support", int),
    ("full_support", int),
))
Metrics.__doc__ = """Metrics for the quality report. Metrics are calculated on the samples
 subset where predictions were made. `full_` prefix means that metric was calculated on all
 available samples. Without `full_` means that metric was calculated only on samples where it has
 prediction from the model. `ppcr` means predicted positive condition rate and shows the
 ratio of samples where the model was able to predict.
"""


def _get_metrics(report: str) -> Metrics:
    """Extract avg / total precision, recall, f1 score, support from report."""
    data = _get_json_data(report)
    avg = data["cl_report"]["micro avg"]
    avg_full = data["cl_report_full"]["micro avg"]
    return Metrics(
        precision=avg["precision"], recall=avg["recall"], full_recall=avg_full["recall"],
        f1=avg["f1-score"], full_f1=avg_full["f1-score"], ppcr=data["ppcr"],
        support=avg["support"], full_support=avg_full["support"])


def _get_model_summary(report: str) -> (int, float):
    """Extract model summary - number of rules and avg. len."""
    data = _get_json_data(report)
    # TODO(vmarkovtsev): address this embarrasing hardcode
    return data["javascript"]["num_rules"], data["javascript"]["avg_rule_len"]


def _get_json_data(report: str) -> dict:
    start_anchor = "```json\n"
    mrr_start = report.find(start_anchor, report.rfind("</summary>"))
    if mrr_start < 0:
        raise ValueError("malformed report")
    mrr_start += len(start_anchor)
    mrr_end = report.find("\n```", mrr_start)
    data = json.loads(report[mrr_start:mrr_end])
    return data


def handle_input_arg(input_arg: str, log: Optional[logging.Logger] = None) -> Iterator[str]:
    """
    Process input argument and return an iterator over input data.

    :param input_arg: file to process or `-` to get data from stdin.
    :param log: Logger if you want to log handling process.
    :return: An iterator over input files.
    """
    log = log.info if log else (lambda *x: None)
    if input_arg == "-":
        log("Reading file paths from stdin.")
        for line in sys.stdin:
            yield line
    else:
        with smart_open(input_arg, "r") as f:
            for line in f:
                yield line


def _generate_report_summary(reports: Iterable[Mapping[str, str]], report_name: str) -> str:
    # precision, recall, f1, support, n_rules, avg_len stats
    additional_fields = ("Rules Number", "Average Rule Len")
    table = []
    fields2id = OrderedDict()
    for repo, report in reports:
        metrics = _get_metrics(report[report_name])
        if not table:
            table.append(("repo",) + metrics._fields + additional_fields)
            for i, field in enumerate(table[0]):
                fields2id[field] = i
        n_rules, avg_len = _get_model_summary(report["model"])
        table.append((get_repo_name(repo),) + metrics + (n_rules, avg_len))
    avgvals = tuple(calc_avg(table[1:], fields2id[field]) for field in Metrics._fields)
    average = tuple(("%" + FLOAT_PRECISION) % v for v in avgvals[:-2])
    average += tuple("%d" % v for v in avgvals[-2:])  # support, full_support
    average += tuple(("%d", "%.1f")[i] % calc_avg(table[1:], fields2id[field])
                     for i, field in enumerate(additional_fields))
    fields_to_weight = (
        ("precision", "support"), ("recall", "support"),
        ("full_recall", "full_support"), ("f1", "support"),
        ("full_f1", "full_support"), ("ppcr", "support"),
    )
    weighted_average = []
    for field, weight_field in fields_to_weight:
        weighted_average.append(("%" + FLOAT_PRECISION) % calc_weighted_avg(
            table[1:], col=fields2id[field], weight_col=fields2id[weight_field]))
    table.append(("average",) + average)
    table.append(("weighted average",) + tuple(weighted_average))
    float_fields = ("precision", "recall", "full_recall", "f1", "full_f1", "ppcr")
    floatfmts = []
    for field in fields2id:
        if field in float_fields:
            floatfmts.append(FLOAT_PRECISION)
        elif field == "Average Rule Len":
            floatfmts.append(".1f")
        else:
            floatfmts.append("g")

    return tabulate(table, tablefmt="pipe", headers="firstrow", floatfmt=floatfmts)


def generate_quality_report(input: str, output: str, force: bool, bblfsh: str, config: dict,
                            database: Optional[str] = None, fs: Optional[str] = None) -> None:
    """
    Generate quality report for the given data. Entry point for command line interface.

    :param input: csv file with repositories to make report. Should contain url, to and from \
                  columns.
    :param output: Directory where to save results.
    :param force: force to overwrite results stored in output directory if True. \
                  Stored results will be used if False.
    :param bblfsh: bblfsh address to use.
    :param config: config for FormatAnalyzer.
    :param database: sqlite3 database path to store the models. Temporary file is used if not set.
    :param fs: Model repository file system root. Temporary directory is used if not set.
    :return:
    """
    os.makedirs(output, exist_ok=True)
    assert os.path.isdir(output), "Output should be a directory"
    log = logging.getLogger("QualityAnalyzer")
    handler = logging.handlers.RotatingFileHandler(os.path.join(output, "errors.txt"))
    handler.setLevel(logging.ERROR)
    log.addHandler(handler)
    reports = []
    config = {QualityReportAnalyzer.name: merge_dicts(config, {"aggregate": True})}
    repositories = list(csv.DictReader(handle_input_arg(input)))
    with tempfile.TemporaryDirectory() as tmpdirname:
        database = database if database else os.path.join(tmpdirname, "db.sqlite3")
        fs = fs if fs else os.path.join(tmpdirname, "models")
        os.makedirs(fs, exist_ok=True)
        with AnalyzerContextManager(QualityReportAnalyzer, db=database, fs=fs,
                                    init=False) as context:
            for row in huge_progress_bar(repositories, log, lambda row: row["url"]):
                path_tmpl = os.path.join(output, get_repo_name(row["url"])) + "-%s_report.md"
                try:
                    if force or not any(os.path.exists(path_tmpl % name)
                                        for name in QualityReportAnalyzer.get_report_names()):
                        vnodes_expected_number = int(row["vnodes_number"]) \
                            if "vnodes_number" in row else None
                        report = measure_quality(
                            row["url"], to_commit=row["to"], from_commit=row["from"],
                            context=context, config=config, bblfsh=bblfsh,
                            vnodes_expected_number=vnodes_expected_number)
                        for report_name in report:
                            with open(path_tmpl % report_name, "w", encoding="utf-8") as f:
                                f.write(report[report_name])
                        reports.append((row["url"], report))
                    else:
                        report = {}
                        log.info("Found existing reports for %s in %s", row["url"], output)
                        for report_name in QualityReportAnalyzer.get_report_names():
                            report_path = path_tmpl % report_name
                            if not os.path.exists(report_path):
                                log.warning(
                                    "skipped %s. %s report is missing", row["url"], report_name)
                                break
                            with open(path_tmpl % report_name, encoding="utf-8") as f:
                                report[report_name] = f.read()
                        else:
                            reports.append((row["url"], report))
                except Exception:
                    log.exception("-" * 20 + "\nFailed to process %s repo", row["url"])
                    continue

        for report_name in ("train", "test"):
            summary = _generate_report_summary(reports, report_name)
            log.info("\n%s\n%s", report_name, summary)
            summary_loc = os.path.join(output, "summary-%s_report.md" % report_name)
            with open(summary_loc, "w", encoding="utf-8") as f:
                f.write(summary)
