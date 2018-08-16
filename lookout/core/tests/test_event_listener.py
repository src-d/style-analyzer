import random
import socket
import unittest

from lookout.core.api.event_pb2 import ReviewEvent, PushEvent
from lookout.core.api.service_analyzer_pb2 import EventResponse
from lookout.core.event_listener import EventListener, EventHandlers
from lookout.core.tests import server


class Handlers(EventHandlers):
    def __init__(self):
        self.request = None

    def process_review_event(self, request: ReviewEvent) -> EventResponse:
        self.request = request
        return EventResponse()

    def process_push_event(self, request: PushEvent) -> EventResponse:
        self.request = request
        return EventResponse()


class EventListenerTests(unittest.TestCase):
    def setUp(self):
        self.handlers = Handlers()
        attempts = 100
        while True:
            attempts -= 1
            if attempts == 0:
                self.fail("cannot find an open port")
            self.port = random.randint(1024, 32768)
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                s.connect(("localhost", self.port))
            except ConnectionRefusedError:
                break
            finally:
                try:
                    s.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                s.close()

    def test_review(self):
        listener = EventListener("localhost:%d" % self.port, self.handlers).start()
        server.run("review",
                   "4984b98b0e2375e9372fbab4eb4c9cd8f0c289c6",
                   "5833b4ba94154cf1ed07f37c32928c7b4411b36b",
                   port=self.port)
        self.assertIsInstance(self.handlers.request, ReviewEvent)
        del listener

    def test_push(self):
        listener = EventListener("localhost:%d" % self.port, self.handlers).start()
        server.run("push",
                   "4984b98b0e2375e9372fbab4eb4c9cd8f0c289c6",
                   "5833b4ba94154cf1ed07f37c32928c7b4411b36b",
                   port=self.port)
        self.assertIsInstance(self.handlers.request, PushEvent)
        del listener


if __name__ == "__main__":
    unittest.main()
