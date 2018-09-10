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


@unittest.skipUnless(os.getenv("LONG_TESTS", False),
                     "Time-consuming tests are skipped by default.")
class AnalyzerIntegrationTests(unittest.TestCase):
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

    def setUp(self):
        self.port = find_port()
        self.db = tempfile.NamedTemporaryFile(dir=self.base_dir)
        self.fs = tempfile.TemporaryDirectory(dir=self.base_dir)
        command = "analyzer init --db sqlite:///%s --fs %s" % (self.db.name, self.fs.name)
        with patch("sys.argv", command.split(" ")):
            launch_analyzer()
        command = ("analyzer run lookout.style.format --db sqlite:///%s "
                   "--server localhost:%d --fs %s --log-level DEBUG"
                   ) % (self.db.name, self.port, self.fs.name)
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

    def tearDown(self):
        logging.getLogger().removeHandler(self.log_handler)
        self.listener.stop()
        self.process.join()
        self.db.close()
        self.fs.cleanup()

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
