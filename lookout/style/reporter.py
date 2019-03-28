"""General utilities to generate performance reports for analyzers."""

import logging
import os
import shutil
import tempfile
from typing import Any, Dict, Iterable, Iterator, NamedTuple, Optional, Sequence, Tuple

from lookout.core.analyzer import Analyzer
from lookout.core.helpers.analyzer_context_manager import AnalyzerContextManager


class Reporter:
    """
    Base class to create performance reports for the analyzer.

    To create a reporter for yours Analyzer you should inherit two classes.
    1. Inherit SpyAnalyzer from Analyzer you want to estimate. SpyAnalyzer `analyze` function
       should be overwritten to return all information you need for the next performance analysis
       in the Comments with a JSON object. See `TyposAnalyzerSpy` as an example.
    2. Inherit MyReporter from this Reporter class. It is expected that you have a dataset that
       you feed to `Reprter.run()` method. After that dataset rows are passed to
       `_trigger_review_event` to trigger Your analyzer analyze method and send a result to
       `_generate_reports` method. If you need to summarize your reports or make a reduced report
       override `_finalize` method.

       Note, that if you want to create several reports for the data
       (like test, train reports for example) you should properly override both
       `get_report_names()` and `_generate_reports()` functions.
    """

    _log = logging.getLogger("Reporter")

    # Reporter can work only with analyzers that provides json in output message.
    inspected_analyzer_type = None  # type: Type[Analyzer]

    def __init__(self, config: Optional[dict] = None, bblfsh: Optional[str] = None,
                 database: Optional[str] = None, fs: Optional[str] = None):
        """
        Get new reporter instance.

        If you want to use existing models and do not retrain them you should provide `database`
        and 'fs' arguments.

        :param config: Analyzer config to feed during push and review events. The analyzer uses \
                       default config if not provided.
        :param bblfsh: Babelfish endpoint to use by lookout-sdk.
        :param database: Database endpoint to use to read and store information about models. \
            Sqlite3 database in the temporary file is used if not provided.
        :param fs: Model repository file system root. Temporary directory is used if not provided.
        """
        if not issubclass(self.inspected_analyzer_type, Analyzer):
            raise AttributeError(
                "inspected_analyzer_type attribute should we set to Analyzer class in %s reporter,"
                " got %s." % (type(self), type(self.inspected_analyzer_type)))
        self._config = config
        self._bblfsh = bblfsh
        self._database = database
        self._fs = fs

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
        def _run(dataset) -> Iterator[Dict[str, str]]:
            for index, row in enumerate(dataset):
                self._log.info("processing %d / %d (%s)", index, len(dataset), row)
                try:
                    fixes = self._trigger_review_event(row)
                    reports = self._generate_reports(row, fixes)
                    reports.update(row)
                    yield reports
                except Exception:
                    self._log.exception("Failed to generate report %d / %d (%s)",
                                        index, len(dataset), row)

        self._log.info("processing %d entries", len(dataset))
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
        Generate reports for the dataset row.

        :param dataset_row: Dataset row which triggered the analyze method of the analyzer.
        :param fixes: List of data provided by the analyze method of spied analyzer.
        :return: Dictionary with report names as keys and report string as values.
        """
        raise NotImplementedError()

    def _trigger_review_event(self, dataset_row: Dict[str, Any]) -> Sequence[NamedTuple]:
        """
        Trigger review event and convert provided comments to internal representation.

        It is required to call `Reporter._analyzer_context_manager.review()` in this function with
        arguments you need and convert provided comments to a Sequence of NamedTuple-s for the
        report generation.

        :param dataset_row: Dataset row which triggers the analyze method of the analyzer.
        :return: Sequence of data extracted from comments to generate report.
        """
        raise NotImplementedError()

    def _finalize(self, reports: Iterable[Dict[str, str]]) -> Iterator[Dict[str, str]]:
        """
        Extend or Summarize generated reports if required.

        Function provides reports as is by default.

        :param reports: Iterable with generated reports.
        :return: New finalized reports.
        """
        yield from reports
