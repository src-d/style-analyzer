from pathlib import Path
import threading
import unittest

import bblfsh

import lookout
from lookout.core.api.event_pb2 import PushEvent, ReviewEvent
from lookout.core.api.service_analyzer_pb2 import EventResponse
from lookout.core.api.service_data_pb2_grpc import DataStub
from lookout.core.data_requests import with_changed_uasts, with_changed_uasts_and_contents, \
    with_uasts, with_uasts_and_contents, DataService
from lookout.core.event_listener import EventListener, EventHandlers
from lookout.core.tests import server


class DataRequestsTests(unittest.TestCase, EventHandlers):
    def setUp(self):
        self.setUpEvent = threading.Event()
        self.tearDownEvent = threading.Event()
        self.port = server.find_port()
        self.listener = EventListener("localhost:%d" % self.port, self).start()
        self.server_thread = threading.Thread(target=self.run_data_service)
        self.server_thread.start()
        self.data_service = DataService("localhost:10301")
        self.url = "file://" + str(Path(lookout.__file__).parent.absolute())
        self.setUpEvent.wait()

    def tearDown(self):
        self.data_service.shutdown()
        self.tearDownEvent.set()
        self.listener.stop()
        self.server_thread.join()

    def process_review_event(self, request: ReviewEvent) -> EventResponse:
        self.setUpEvent.set()
        self.tearDownEvent.wait()
        return EventResponse()

    def process_push_event(self, request: PushEvent) -> EventResponse:
        self.setUpEvent.set()
        self.tearDownEvent.wait()
        return EventResponse()

    def run_data_service(self):
        server.run("push",
                   "4984b98b0e2375e9372fbab4eb4c9cd8f0c289c6",
                   "5833b4ba94154cf1ed07f37c32928c7b4411b36b",
                   self.port)

    def test_with_changed_uasts(self):
        def func(imposter, commit_from: str, commit_to: str,
                 data_request_stub: DataStub, **data):
            changes = list(data["changes"])
            self.assertEqual(len(changes), 1)
            change = changes[0]
            self.assertEqual(change.base.content, b"")
            self.assertEqual(change.head.content, b"")
            self.assertEqual(type(change.base.uast).__module__, bblfsh.Node.__module__)
            self.assertEqual(type(change.head.uast).__module__, bblfsh.Node.__module__)
            self.assertEqual(change.base.path, change.head.path)
            self.assertEqual(change.base.path, "lookout/core/manager.py")
            self.assertEqual(change.base.language, "Python")
            self.assertEqual(change.head.language, "Python")

        func = with_changed_uasts(func)
        func(self,
             "4984b98b0e2375e9372fbab4eb4c9cd8f0c289c6",
             "5833b4ba94154cf1ed07f37c32928c7b4411b36b",
             self.data_service.get())

    def test_with_changed_uasts_and_contents(self):
        def func(imposter, commit_from: str, commit_to: str,
                 data_request_stub: DataStub, **data):
            changes = list(data["changes"])
            self.assertEqual(len(changes), 1)
            change = changes[0]
            self.assertEqual(len(change.base.content), 5548)
            self.assertEqual(len(change.head.content), 5542)
            self.assertEqual(type(change.base.uast).__module__, bblfsh.Node.__module__)
            self.assertEqual(type(change.head.uast).__module__, bblfsh.Node.__module__)
            self.assertEqual(change.base.path, change.head.path)
            self.assertEqual(change.base.path, "lookout/core/manager.py")
            self.assertEqual(change.base.language, "Python")
            self.assertEqual(change.head.language, "Python")

        func = with_changed_uasts_and_contents(func)
        func(self,
             "4984b98b0e2375e9372fbab4eb4c9cd8f0c289c6",
             "5833b4ba94154cf1ed07f37c32928c7b4411b36b",
             self.data_service.get())

    def test_with_uasts(self):
        def func(imposter, url: str, commit: str, config: dict,
                 data_request_stub: DataStub, **data):
            files = list(data["files"])
            self.assertEqual(len(files), 61)
            for file in files:
                self.assertEqual(file.content, b"")
                self.assertEqual(type(file.uast).__module__, bblfsh.Node.__module__)
                self.assertTrue(file.path)
                self.assertIn(file.language, ("Python", ""))

        func = with_uasts(func)
        func(self,
             self.url,
             "5833b4ba94154cf1ed07f37c32928c7b4411b36b",
             None,
             self.data_service.get())

    def test_with_uasts_and_contents(self):
        def func(imposter, url: str, commit: str, config: dict,
                 data_request_stub: DataStub, **data):
            files = list(data["files"])
            self.assertEqual(len(files), 61)
            for file in files:
                if not file.path.endswith("__init__.py"):
                    self.assertGreater(len(file.content), 0, file.path)
                self.assertEqual(type(file.uast).__module__, bblfsh.Node.__module__)
                self.assertTrue(file.path)
                self.assertIn(file.language, ("Python", ""))

        func = with_uasts_and_contents(func)
        func(self,
             self.url,
             "5833b4ba94154cf1ed07f37c32928c7b4411b36b",
             None,
             self.data_service.get())


if __name__ == "__main__":
    unittest.main()
