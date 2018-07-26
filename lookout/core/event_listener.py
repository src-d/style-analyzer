from concurrent.futures import ThreadPoolExecutor
import functools
import logging
from threading import Event

import grpc
import stringcase

from lookout.core.api.service_analyzer_pb2_grpc import \
    AnalyzerServicer, add_AnalyzerServicer_to_server
from lookout.core.api.event_pb2 import ReviewEvent, PushEvent
from lookout.core import slogging


def extract_review_event_context(request: ReviewEvent):
    return {
        "type": "ReviewEvent",
        "url_from": request.commit_revision.base.internal_repository_url,
        "url_to": request.commit_revision.head.internal_repository_url,
        "commit_from": request.commit_revision.base.Hash,
        "commit_to": request.commit_revision.head.Hash,
    }


def extract_push_event_context(request: PushEvent):
    return {
        "type": "PushEvent",
        "url": request.commit_revision.head.internal_repository_url,
        "head": request.commit_revision.head.Hash,
        "count": request.distinct_commits,
    }


request_log_context_extractors = {
    ReviewEvent: extract_review_event_context,
    PushEvent: extract_push_event_context,
}


class EventListener(AnalyzerServicer):
    def __init__(self, address: str, handlers, n_workers: int=1):
        self._server = grpc.server(ThreadPoolExecutor(max_workers=n_workers),
                                   maximum_concurrent_rpcs=n_workers)
        add_AnalyzerServicer_to_server(self, self._server)
        self.handlers = handlers
        self._server.add_insecure_port(address)
        self._stop_event = Event()
        self._log = logging.getLogger(type(self).__name__)

    def start(self):
        self._server.start()
        return self

    def block(self):
        self._stop_event.clear()
        try:
            self._stop_event.wait()
        except KeyboardInterrupt:
            pass

    def stop(self, cancel_running=False):
        self._stop_event.set()
        self._server.stop(None if cancel_running else 0)

    def set_logging_context(func):
        @functools.wraps(func)
        def wrapped_set_logging_context(self, request, context: grpc.ServicerContext):
            obj = request_log_context_extractors[type(request)](request)
            meta = {}
            for md in context.invocation_metadata():
                meta[md.key] = md.value
            obj["meta"] = meta
            obj["peer"] = context.peer()
            slogging.set_context(obj)
            self._log.info("new %s", type(request).__name__)
            return func(self, request, context)

        return wrapped_set_logging_context

    def log_exceptions(func):
        @functools.wraps(func)
        def wrapped_catch_them_all(self, request, context: grpc.ServicerContext):
            try:
                return wrapped_catch_them_all(self, request, context)
            except Exception as e:
                self._log.exception("")
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details("%s: %s" % (type(e), e))
                raise e from None

        return wrapped_catch_them_all

    def handle(func):
        @functools.wraps(func)
        def wrapped_handle(self, request, context: grpc.ServicerContext):
            method_name = "process_" + stringcase.snakecase(type(request).__name__)
            return getattr(self.handlers, method_name)(request)

        return wrapped_handle

    @set_logging_context
    @log_exceptions
    @handle
    def NotifyReviewEvent(self, request: ReviewEvent, context: grpc.ServicerContext):
        pass

    @set_logging_context
    @log_exceptions
    @handle
    def NotifyPushEvent(self, request: PushEvent, context: grpc.ServicerContext):
        pass

    set_logging_context = staticmethod(set_logging_context)
    log_exceptions = staticmethod(log_exceptions)
    handle = staticmethod(handle)
