import logging
import os
from pathlib import Path
import re
import tarfile
import tempfile
import unittest

from lookout.core.test_helpers import server
from modelforge import slogging

from lookout.style.format.analyzer import FormatAnalyzer
from lookout.style.format.benchmarks.analyzer_context_manager import AnalyzerContextManager
from lookout.style.format.tests import long_test

FROM_COMMIT = "HEAD" + "^" * 9
TO_COMMIT = "HEAD"


class BaseAnalyzerIntegrationTests(unittest.TestCase):
    def setUp(self, fs=None):
        self.port = server.find_port()
        self.db = tempfile.NamedTemporaryFile(dir=self.base_dir)
        if fs is None:
            self.fs = tempfile.TemporaryDirectory(dir=self.base_dir)
        else:
            self.fs = fs

        self.analyzer = AnalyzerContextManager(
            FormatAnalyzer, port=self.port, db=self.db.name, fs=self.fs.name).__enter__()
        self.logs = logs = []

        class ShadowHandler(logging.Handler):
            def emit(self, record):
                logs.append(logging.getLogger().handlers[0].format(record))

        self.log_handler = ShadowHandler()
        logging.getLogger().addHandler(self.log_handler)

        if not os.path.exists(str(server.exefile)):
            server.fetch()

    def tearDown(self, fs_cleanup=True):
        if fs_cleanup:
            self.fs.cleanup()
        self.analyzer.__exit__()
        logging.getLogger().removeHandler(self.log_handler)


@long_test
class AnalyzerIntegrationTests(BaseAnalyzerIntegrationTests):
    @classmethod
    def setUpClass(cls):
        slogging.setup("DEBUG", False)
        parent = Path(__file__).parent.resolve()
        cls.base_dir_ = tempfile.TemporaryDirectory()
        cls.base_dir = cls.base_dir_.name
        cls.jquery_dir = os.path.join(cls.base_dir, "jquery")
        # str() is needed for Python 3.5
        with tarfile.open(str(parent / "jquery.tar.xz")) as tar:
            tar.extractall(path=cls.base_dir)

    @classmethod
    def tearDownClass(cls):
        cls.base_dir_.cleanup()

    def test_review(self):
        server.run(
            "review",
            FROM_COMMIT,
            TO_COMMIT,
            self.port, git_dir=self.jquery_dir)
        matches = re.search(r"FormatAnalyzer: (\d+) comments", "".join(self.logs))
        self.assertTrue(matches)
        self.assertGreater(int(matches.group(1)), 0)


if __name__ == "__main__":
    unittest.main()
