import logging

from lookout.core.analyzer import Analyzer
from lookout.core.api.event_pb2 import ReviewEvent, PushEvent
from lookout.core.api.service_analyzer_pb2 import EventResponse


class AnalyzerManager:
    _log = logging.getLogger("AnalyzerManager")

    def __init__(self):
        self._registered = set()
        self.version = "<unknown>"

    def register(self, cls: Analyzer):
        self._registered.add(cls)

    def process_review_event(self, request: ReviewEvent):
        response = EventResponse()
        response.analyzer_version = self.version
        comments = []
        for analyzer in self._registered:
            self._log.debug("running %s", analyzer.__name__)
            results = analyzer.analyze(request)
            self._log.info("%s: %d comments", analyzer.__name__, len(results))
            comments.extend(results)
        response.comments.extend(comments)
        return response

    def process_push_event(self, request: PushEvent):
        for analyzer in self._registered:
            analyzer.push(request)


manager = AnalyzerManager()


def register(cls: Analyzer):
    manager.register(cls)
    return cls
