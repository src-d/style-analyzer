import json
from typing import Any, Dict, NamedTuple, Sequence
import unittest
from unittest import mock

from lookout.core.analyzer import Analyzer, AnalyzerModel, DummyAnalyzerModel, ReferencePointer
from lookout.core.data_requests import DataService
from lookout.sdk.service_analyzer_pb2 import Comment


from lookout.style.reporter import Reporter

DummyFix = NamedTuple("DummyFix", (("value", str), ("fr", str), ("to", str)))


class FakeDummyAnalyzerSpy(Analyzer):
    version = 1
    model_type = DummyAnalyzerModel
    name = "fake.analyzer.FakeDummyAnalyzerSpy"
    vendor = "source{d}"

    def __init__(self, model: AnalyzerModel, url: str, config: dict):
        super().__init__(model, url, config)

    def analyze(self, ptr_from: str, ptr_to: str, data_service: DataService, **data) -> [Comment]:
        return [Comment(text=json.dumps(DummyFix("#1", ptr_from, ptr_to)._asdict())),
                Comment(text=json.dumps(DummyFix("#2", ptr_from, ptr_to)._asdict()))]

    @classmethod
    def train(cls, ptr: ReferencePointer, config: dict, data_service: DataService, **data) \
            -> AnalyzerModel:
        return DummyAnalyzerModel()


def fake_review(fr, to):
    yield from FakeDummyAnalyzerSpy(DummyAnalyzerModel(), "", {}).analyze(fr, to, None)


class DummyReporter(Reporter):
    inspected_analyzer_type = FakeDummyAnalyzerSpy

    def get_report_names(cls):
        return "report1", "report2"

    def _generate_reports(self, dataset_row: Dict[str, Any], fixes: Sequence[DummyFix],
                          ) -> Dict[str, str]:
        reports = {}
        for i in range(1, 3):
            key = "report%d" % i
            value = ["%d. Row value: %s" % (i, dataset_row["value"])]
            for fix in fixes:
                value.append("Fix: %s" % repr(fix))
            value = " ".join(value)
            reports[key] = value

        return reports

    def _trigger_review_event(self, dataset_row: Dict[str, Any]) -> Sequence[DummyFix]:
        comments = self._analyzer_context_manager.review(dataset_row["fr"], dataset_row["to"])
        dummy_fixes = []
        for comment in comments:
            dummy_fix_dict = json.loads(comment.text)
            dummy_fixes.append(DummyFix(value=dummy_fix_dict["value"],
                                        fr=dummy_fix_dict["fr"],
                                        to=dummy_fix_dict["to"]))
        return dummy_fixes


class ReporterTests(unittest.TestCase):
    @mock.patch("lookout.core.helpers.analyzer_context_manager.AnalyzerContextManager.review",
                side_effect=fake_review)
    def test_run(self, func):
        with DummyReporter() as reporter:
            reports = list(reporter.run([{"fr": "00", "to": "XX", "value": "aaa"},
                                         {"fr": "11", "to": "FF", "value": "bbb"}]))
            correct_reports = [{"fr": "00",
                                "report1": "1. Row value: aaa "
                                           "Fix: DummyFix(value='#1', fr='00', to='XX') "
                                           "Fix: DummyFix(value='#2', fr='00', to='XX')",
                                "report2": "2. Row value: aaa "
                                           "Fix: DummyFix(value='#1', fr='00', to='XX') "
                                           "Fix: DummyFix(value='#2', fr='00', to='XX')",
                                "to": "XX",
                                "value": "aaa"},
                               {"fr": "11",
                                "report1": "1. Row value: bbb "
                                           "Fix: DummyFix(value='#1', fr='11', to='FF') "
                                           "Fix: DummyFix(value='#2', fr='11', to='FF')",
                                "report2": "2. Row value: bbb "
                                           "Fix: DummyFix(value='#1', fr='11', to='FF') "
                                           "Fix: DummyFix(value='#2', fr='11', to='FF')",
                                "to": "FF",
                                "value": "bbb"}]
            self.assertEqual(reports, correct_reports)


if __name__ == "__main__":
    unittest.main()
