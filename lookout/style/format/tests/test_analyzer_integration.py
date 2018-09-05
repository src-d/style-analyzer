from pathlib import Path
import queue
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
from threading import Thread
import unittest
from unittest.mock import patch

from lookout.__main__ import main as analyzer
from lookout.core.tests.server import find_port, run


class AnalyzerIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base_path = Path(__file__).parent.resolve()
        cls.base_dir = str(base_path)
        print(cls.base_dir)
        print(cls.base_dir)
        cls.jquery_dir = str(base_path / "jquery")
        # str() is needed for Python 3.5
        with tarfile.open(str(base_path / "jquery.tar.xz")) as tar:
            tar.extractall(path=str(cls.base_dir))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.jquery_dir)

    def _wait_ready(self):
        for line in iter(self.analyzer.stdout.readline, b""):
            line = line.decode("utf-8", "replace")
            if line.find(":run:Listening localhost:%d" % self.port) != -1:
                return True

    def _tee(self):
        for line in iter(self.analyzer.stdout.readline, b""):
            line = line.decode("utf-8", "replace")
            line = line.rstrip()
            print(line, file=sys.stderr)
            self.analyzer_output.put(line)

    def setUp(self):
        self.port = find_port()
        self.db = tempfile.NamedTemporaryFile(dir=self.base_dir)
        self.fs = tempfile.TemporaryDirectory(dir=self.base_dir)
        command = "analyzer init --db sqlite:///%s --fs %s" % (self.db.name, self.fs.name)
        with patch("sys.argv", command.split(" ")):
            analyzer()
        command = ("python3 -m lookout run lookout.style.format --db sqlite:///%s "
                   "--server localhost:%d --fs %s --log-level DEBUG"
                   ) % (self.db.name, self.port, self.fs.name)
        self.analyzer = subprocess.Popen(
            command.split(" "), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        waiter = Thread(target=self._wait_ready)
        waiter.start()
        waiter.join(5)
        if waiter.is_alive():
            raise RuntimeError("couldn't start the analyzer correctly")
        self.analyzer_output = queue.Queue()
        self.tee = Thread(target=self._tee)
        self.tee.start()

    def tearDown(self):
        self.analyzer.send_signal(signal.SIGINT)
        self.analyzer.wait()
        self.analyzer.stdout.close()
        self.tee.join()
        self.db.close()
        self.fs.cleanup()

    def test_review(self):
        run("review", "fbd32214d1b08e09278f39c77979658cabde6c4d",
            "931442679ad04ebadad2b70a9f938fb0bc64d537", self.port, git_dir=self.jquery_dir)
        while True:
            try:
                line = self.analyzer_output.get_nowait()
            except queue.Empty:
                break
            matches = re.search(r"FormatAnalyzer: (\d+) comments", line)
            if matches:
                self.assertGreater(int(matches.group(1)), 0)
                return
        self.fail("did not output any comment")


if __name__ == "__main__":
    unittest.main()
