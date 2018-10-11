import logging
import os
from pathlib import Path
import re
import shutil
import tarfile
import tempfile
import threading
from threading import Thread
import unittest
from unittest.mock import patch

from lookout.__main__ import main as launch_analyzer
from lookout.core.event_listener import EventListener
from lookout.core.tests.server import find_port, run as launch_server


class TestAnalyzer:
    """Context manager for launching analyzer."""
    def __init__(self, port: int, db: str, fs: str, analyzer: str = "lookout.style.format"):
        """
        :param port: port to use for analyzer.
        :param db: database location.
        :param fs: location where to store results of launched analyzer.
        :param analyzer: analyzer to use
        """
        self.port = port
        self.db = db
        self.fs = fs
        self.analyzer = analyzer

    def __enter__(self):
        command = "analyzer init --db sqlite:///%s --fs %s" % (self.db, self.fs)
        with patch("sys.argv", command.split(" ")):
            launch_analyzer()
        command = ("analyzer run %s --db sqlite:///%s "
                   "--server localhost:%d --fs %s --log-level DEBUG"
                   ) % (self.analyzer, self.db, self.port, self.fs)
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

    def __exit__(self, exc_type, exc_val, exc_tb):
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
        shutil.rmtree(self.jquery_dir)
        if fs_cleanup:
            self.fs.cleanup()
        self.analyzer.__exit__()


@unittest.skipUnless(os.getenv("LONG_TESTS", False),
                     "Time-consuming tests are skipped by default.")
class AnalyzerIntegrationTests(BaseAnalyzerIntegrationTests):
    @classmethod
    def setUpClass(cls):
        parent = Path(__file__).parent.resolve()
        cls.base_dir = str(parent)
        cls.jquery_dir = str(parent / "jquery")
        # str() is needed for Python 3.5
        with tarfile.open(str(parent / "jquery.tar.xz")) as tar:
            tar.extractall(path=cls.base_dir)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.jquery_dir)

    def test_review(self):
        launch_server(
            "review",
            "fbd32214d1b08e09278f39c77979658cabde6c4d",
            "931442679ad04ebadad2b70a9f938fb0bc64d537",
            self.port, git_dir=self.jquery_dir)
        matches = re.search(r"FormatAnalyzer: (\d+) comments", "".join(self.logs))
        self.assertTrue(matches)
        self.assertGreater(int(matches.group(1)), 0)


if __name__ == "__main__":
    unittest.main()
