"""General utilities to generate performance reports for analyzers."""

import logging
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Dict, Iterable, Iterator, NamedTuple, Optional, Sequence, Tuple

from dulwich import porcelain
import dulwich.repo
from lookout.core.analyzer import Analyzer
from lookout.core.helpers.analyzer_context_manager import AnalyzerContextManager

import lookout.style


class Reporter:
    """
    Base class to create performance reports for the analyzer.

    To create a reporter for your Analyzer you should make two steps.
    1. Inherit SpyAnalyzer from an Analyzer you want to evaluate. SpyAnalyzer's `analyze` \
       function should be overridden to return all the information you need for the following \
       evaluation of the `Comment`-s. Refer to `TyposAnalyzerSpy` as an example.
    2. Inherit MyReporter from this Reporter class. Set created SpyAnalyzer to \
       `inspected_analyzer_type` attribute. You should have a dataset that you feed to \
       `Reporter.run()`. The dataset rows are passed to `_trigger_review_event` to trigger your \
       analyzer's `analyze()`. The result is passed to `_generate_reports()`. If you need to \
       summarize your reports, override `_finalize` method.

       If you want to create several reports (e.g. separate train and test reports) you should \
       override both `get_report_names()` and `_generate_reports()`.
    """

    _log = logging.getLogger("Reporter")

    inspected_analyzer_type = None  # type: Type[Analyzer]

    def __init__(self, config: Optional[dict] = None, bblfsh: Optional[str] = None,
                 database: Optional[str] = None, fs: Optional[str] = None):
        """
        Initialize a new `Reporter` instance.

        You should provide `database` and `fs` in order to re-use existing models (no training).

        :param config: Analyzer configuration for push and review events. The analyzer uses \
                       default config if not provided.
        :param bblfsh: Babelfish endpoint to use by lookout-sdk.
        :param database: Database endpoint to use to read and store information about models. \
            Sqlite3 database in a temporary file is used if not provided.
        :param fs: Model repository file system root. Temporary directory is used if not provided.
        """
        if self.inspected_analyzer_type is None or \
                not issubclass(self.inspected_analyzer_type, Analyzer):
            raise AttributeError(
                "inspected_analyzer_type attribute must be set to an Analyzer subclass in %s,"
                " got %s." % (type(self), repr(self.inspected_analyzer_type)))
        self._config = config
        self._bblfsh = bblfsh
        self._database = database
        self._fs = fs
        self._fails = {}

    def __enter__(self) -> "Reporter":
        self._tmpdir = tempfile.mkdtemp("reporter-") \
            if self._database is None or self._fs is None else None
        if self._database is None:
            self._database = os.path.join(self._tmpdir, "db.sqlite3")
        if self._fs is None:
            self._fs = os.path.join(self._tmpdir, "models")
        os.makedirs(self._fs, exist_ok=True)
        self._analyzer_context_manager = AnalyzerContextManager(
            self.inspected_analyzer_type, db=self._database, fs=self._fs, init=False)
        self._analyzer_context_manager.__enter__()
        return self

    def __exit__(self, exc_type=None, exc_val=None, exc_tb=None):
        self._analyzer_context_manager.__exit__()
        if self._tmpdir:
            shutil.rmtree(self._tmpdir)

    def run(self, dataset: Sequence[Dict[str, Any]]) -> Iterator[Dict[str, str]]:
        """
        Run report generation.

        :param dataset: The dataset for the report generation. The format is a list of data rows. \
                        The row is a Dictionary with a mapping from the column name to its content.

        :return: Iterator through generated reports. Each Generated report is extended with the \
                 corresponding row data from the dataset.
        """
        self._fails = {}

        def _run(dataset) -> Iterator[Dict[str, str]]:
            for index, row in enumerate(dataset, start=1):
                self._log.info("processing %d / %d (%s)", index, len(dataset), row)
                try:
                    fixes = self._trigger_review_event(row)
                    reports = self._generate_reports(row, fixes)
                    reports.update(row)
                    yield reports
                except Exception:
                    self._log.exception("failed to generate report %d / %d (%s)",
                                        index, len(dataset), row)
                    self._fails[index] = row

        yield from self._finalize(_run(dataset))

    @classmethod
    def get_report_names(cls) -> Tuple[str, ...]:
        """
        Get all available report names.

        :return: Tuple with report names.
        """
        raise NotImplementedError()

    def _generate_reports(self, dataset_row: Dict[str, Any], fixes: Sequence[NamedTuple],
                          ) -> Dict[str, str]:
        """
        Generate reports for a dataset row.

        :param dataset_row: Dataset row which triggered the analyze method of the analyzer.
        :param fixes: List of data provided by the analyze method of spied analyzer.
        :return: Dictionary with report names as keys and report string as values.
        """
        raise NotImplementedError()

    def _trigger_review_event(self, dataset_row: Dict[str, Any]) -> Sequence[NamedTuple]:
        """
        Trigger review event and convert provided comments to an internal representation.

        It is required to call `Reporter._analyzer_context_manager.review()` in this function \
        with arguments you need and convert provided comments to a Sequence of NamedTuple-s for \
        the report generation.

        :param dataset_row: Dataset row with information required to run \
                            `analyzer_context_manager.review()`.
        :return: Sequence of data extracted from comments to generate report.
        """
        raise NotImplementedError()

    def _finalize(self, reports: Iterable[Dict[str, str]]) -> Iterator[Dict[str, str]]:
        """
        Extend or summarize the generated reports.

        The function does not change the reports by default.

        :param reports: Iterable with generated reports.
        :return: New finalized reports.
        """
        yield from reports

    @staticmethod
    def _get_package_version():
        """Return lookout-style package version or "local" if it is a git repository."""
        if (Path(__file__).parents[2] / ".git").exists():
            return "local"
        else:
            return lookout.style.__version__

    @staticmethod
    def _get_commit():
        """Return current head commit hash if you run inside git repository."""
        if Reporter._get_package_version() != "local":
            return "N/A"
        clean_status = porcelain.GitStatus(
            staged={"delete": [], "add": [], "modify": []}, unstaged=[], untracked=[])
        repo_path = str(Path(__file__).parents[2])
        head = dulwich.repo.Repo(repo_path).head().decode()
        if porcelain.status(repo_path) == clean_status:
            return head
        else:
            return "%s (dirty)" % head
