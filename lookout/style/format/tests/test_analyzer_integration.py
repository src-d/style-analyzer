import logging
import os
from pathlib import Path
import re
import tarfile
import tempfile
import threading
from threading import Thread
from typing import Sequence, Union
import unittest
from unittest.mock import patch

from lookout.core.event_listener import EventListener
from lookout.core.test_helpers.server import find_port, run as launch_server

# PyCharm has a bug and inserts a new line here
from lookout.__main__ import main as launch_analyzer  # noqa: I202, I100

FROM_COMMIT = "HEAD" + "^" * 9
TO_COMMIT = "HEAD"


class TestAnalyzer:
    """Context manager for launching analyzer."""
    def __init__(self, port: int, db: str, fs: str, config: str = "",
                 analyzer: Union[str, Sequence] = "lookout.style.format"):
        """
        :param port: port to use for analyzer.
        :param db: database location.
        :param fs: location where to store results of launched analyzer.
        :param config: Path to the configuration file with option defaults. If empty - skip.
        :param analyzer: analyzer(s) to use.
        """
        self.port = port
        self.db = db
        self.fs = fs
        self.config = config
        self.analyzer = analyzer if type(analyzer) is str else " ".join(analyzer)

    def __enter__(self):
        command = "analyzer init --db sqlite:///%s --fs %s" % (self.db, self.fs)
        with patch("sys.argv", command.split(" ")):
            launch_analyzer()

        command = "analyzer run %s " % self.analyzer
        if self.config:
            command += " --config %s " % self.config
        command += "--db sqlite:///%s --server localhost:%d --fs %s --log-level DEBUG"  \
                   % (self.db, self.port, self.fs)
        self.logs = logs = []

        class ShadowHandler(logging.Handler):
            def emit(self, record):
                logs.append(logging.getLogger().handlers[0].formatter.format(record))

        self.log_handler = ShadowHandler()
        logging.getLogger().addHandler(self.log_handler)

        def main():
            with patch("sys.argv", command.split(" ")):
                launch_analyzer()

        sync = threading.Event()
        block = EventListener.block

        def sync_block(this):
            self.listener = this
            sync.set()
            return block(this)

        EventListener.block = sync_block
        self.process = Thread(target=main)
        try:
            self.process.start()
            sync.wait()
        finally:
            EventListener.block = block

        return self

    def __exit__(self, exc_type=None, exc_val=None, exc_tb=None):
        logging.getLogger().removeHandler(self.log_handler)
        self.listener.stop()
        self.process.join()


class BaseAnalyzerIntegrationTests(unittest.TestCase):
    def setUp(self, fs=None):
        self.port = find_port()
        self.db = tempfile.NamedTemporaryFile(dir=self.base_dir)
        if fs is None:
            self.fs = tempfile.TemporaryDirectory(dir=self.base_dir)
        else:
            self.fs = fs

        self.analyzer = TestAnalyzer(port=self.port, db=self.db.name, fs=self.fs.name).__enter__()

    def tearDown(self, fs_cleanup=True):
        if fs_cleanup:
            self.fs.cleanup()
        self.analyzer.__exit__()


@unittest.skipUnless(os.getenv("LONG_TESTS", False),
                     "Time-consuming tests are skipped by default.")
class AnalyzerIntegrationTests(BaseAnalyzerIntegrationTests):
    @classmethod
    def setUpClass(cls):
        parent = Path(__file__).parent.resolve()
        cls.base_dir_ = tempfile.TemporaryDirectory(dir=str(parent))
        cls.base_dir = cls.base_dir_.name
        cls.jquery_dir = os.path.join(cls.base_dir, "jquery")
        # str() is needed for Python 3.5
        with tarfile.open(str(parent / "jquery.tar.xz")) as tar:
            tar.extractall(path=cls.base_dir)

    @classmethod
    def tearDownClass(cls):
        cls.base_dir_.cleanup()

    def test_review(self):
        launch_server(
            "review",
            FROM_COMMIT,
            TO_COMMIT,
            self.port, git_dir=self.jquery_dir)
        matches = re.search(r"FormatAnalyzer: (\d+) comments", "".join(self.analyzer.logs))
        self.assertTrue(matches)
        self.assertGreater(int(matches.group(1)), 0)


if __name__ == "__main__":
    unittest.main()
