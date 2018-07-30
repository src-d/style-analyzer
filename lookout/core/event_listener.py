from concurrent.futures import ThreadPoolExecutor
import functools
import logging
from threading import Event
import time

import grpc
import stringcase

from lookout.core.api.service_analyzer_pb2_grpc import \
    AnalyzerServicer, add_AnalyzerServicer_to_server
from lookout.core.api.event_pb2 import ReviewEvent, PushEvent
from lookout.core import slogging


def extract_review_event_context(request: ReviewEvent):
    return {
        "type": "ReviewEvent",
        "url_base": request.commit_revision.base.internal_repository_url,
        "url_head": request.commit_revision.head.internal_repository_url,
        "commit_base": request.commit_revision.base.hash,
        "commit_head": request.commit_revision.head.hash,
    }


def extract_push_event_context(request: PushEvent):
    return {
        "type": "PushEvent",
        "url": request.commit_revision.head.internal_repository_url,
        "head": request.commit_revision.head.hash,
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
        self._server.address = address
        self._server.n_workers = n_workers
        add_AnalyzerServicer_to_server(self, self._server)
        self.handlers = handlers
        self._server.add_insecure_port(address)
        self._stop_event = Event()
        self._log = logging.getLogger(type(self).__name__)

    def __str__(self) -> str:
        return "EventListener(%s, %d workers)" % (self._server.address, self._server.n_workers)

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

    def timeit(func):
        @functools.wraps(func)
        def wrapped_timeit(self, request, context: grpc.ServicerContext):
            start_time = time.perf_counter_ns()
            context.start_time = start_time
            result = func(self, request, context)
            delta = time.perf_counter_ns() - start_time
            self._log.info("OK %d", delta // 1000)
            return result

        return wrapped_timeit

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
                return func(self, request, context)
            except Exception as e:
                start_time = getattr(request, "start_time", None)
                if start_time is not None:
                    delta = time.perf_counter_ns() - start_time
                    self._log.exception("FAIL %d", delta // 1000)
                else:
                    self._log.exception("FAIL ?")
                context.abort(grpc.StatusCode.INTERNAL, "%s: %s" % (type(e), e))

        return wrapped_catch_them_all

    def handle(func):
        @functools.wraps(func)
        def wrapped_handle(self, request, context: grpc.ServicerContext):
            method_name = "process_" + stringcase.snakecase(type(request).__name__)
            return getattr(self.handlers, method_name)(request)

        return wrapped_handle

    @set_logging_context
    @timeit
    @log_exceptions
    @handle
    def NotifyReviewEvent(self, request: ReviewEvent, context: grpc.ServicerContext):
        pass

    @set_logging_context
    @timeit
    @log_exceptions
    @handle
    def NotifyPushEvent(self, request: PushEvent, context: grpc.ServicerContext):
        pass

    timeit = staticmethod(timeit)
    set_logging_context = staticmethod(set_logging_context)
    log_exceptions = staticmethod(log_exceptions)
    handle = staticmethod(handle)
