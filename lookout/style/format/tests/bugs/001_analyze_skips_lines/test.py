import logging
import os
from typing import List
import unittest

from bblfsh import BblfshClient
from lookout.core import slogging
from lookout.core.analyzer import ReferencePointer
from lookout.core.api.service_analyzer_pb2 import Comment
from lookout.core.data_requests import DataService
from lookout.core.lib import parse_files

from lookout.style.format.analyzer import FileFix, LineFix  # noqa: F401
from lookout.style.format.benchmarks.general_report import analyze_files, FormatAnalyzerSpy


class AnalyzeSkipsLines(unittest.TestCase):
    def setUp(self):
        slogging.setup("DEBUG", False)
        self.bblfsh_endpoint = "0.0.0.0:9432"

    def test(self):
        fixes = []  # type: FileFix
        bblfsh_client = BblfshClient(self.bblfsh_endpoint)
        basedir = os.path.dirname(__file__)
        base_files = parse_files(
            filepaths=[os.path.join(basedir, "find_chrome_base.js")],
            line_length_limit=500,
            overall_size_limit=5 << 20,
            client=bblfsh_client,
            language="javascript")
        base_files[0].path = os.path.join(basedir, "find_chrome_head.js")

        class Runner(FormatAnalyzerSpy):
            def analyze(self, ptr_from: ReferencePointer, ptr_to: ReferencePointer,
                        data_service: DataService, **data) -> List[Comment]:
                """
                We run the analysis on the single pair of files: `find_chrome.js`.
                """
                class FakeStub:
                    def GetFiles(self, *args, **kwargs):
                        return base_files

                class FakeDataService:
                    def get_data(self):
                        return FakeStub()

                fixes.extend(self.run(ptr_from, data_service, FakeDataService()))
                return []

        log = logging.getLogger(type(self).__name__)
        analyze_files(
            Runner,
            {},
            os.path.join(basedir, "style.format.analyzer.FormatAnalyzer_1.asdf"),
            "javascript",
            self.bblfsh_endpoint,
            os.path.join(basedir, "find_chrome_head.js"),
            log)
        self.assertEqual(len(fixes), 1)
        self.assertEqual(len(fixes[0].line_fixes), 1)
        fix = fixes[0].line_fixes[0]  # type: LineFix
        self.assertEqual(fix.line_number, 22)
        self.assertEqual(
            fix.suggested_code, "const execFileSync = require('child_process').execFileSync;")


if __name__ == "__main__":
    unittest.main()
