import signal
import subprocess
import sys
import tempfile
import time
import unittest

from lookout.core.tests.server import find_port, run


class AnalyzerIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.port = find_port()
        self.db = tempfile.NamedTemporaryFile()
        command = "python3 -m lookout init --db sqlite:///%s --fs /tmp" % self.db.name
        subprocess.check_call(command.split(" "))
        command = "python3 -m lookout run lookout.style.format --db sqlite:///%s " \
            "--server localhost:%d --fs /tmp" % (self.db.name, self.port)
        self.analyzer = subprocess.Popen(
            command.split(" "), stdout=sys.stdout, stderr=sys.stderr)
        # FIXME(@m09): is there a better way?
        time.sleep(3)

    def tearDown(self):
        self.db.close()
        self.analyzer.send_signal(signal.SIGINT)
        # FIXME(@m09): lib/python3.6/subprocess.py:766: ResourceWarning: subprocess 5891 is still
        # running
        # ResourceWarning, source=self)

    def test_review(self):
        run("review",
            "4984b98b0e2375e9372fbab4eb4c9cd8f0c289c6",
            "5833b4ba94154cf1ed07f37c32928c7b4411b36b",
            self.port)


if __name__ == "__main__":
    unittest.main()
