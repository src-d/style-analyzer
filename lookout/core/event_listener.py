from concurrent.futures import ThreadPoolExecutor
import functools
import logging
from threading import Event
import time

import grpc
import stringcase

from lookout.core.api.service_analyzer_pb2 import EventResponse
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


class EventHandlers:
    """
    Interface of the classes which process Lookout gRPC events.
    """
    def process_review_event(self, request: ReviewEvent) -> EventResponse:
        raise NotImplementedError

    def process_push_event(self, request: PushEvent) -> EventResponse:
        raise NotImplementedError


class EventListener(AnalyzerServicer):
    """
    gRPC ninja which listens to the events coming from the Lookout server.
    So far it supports two events: NotifyReviewEvent and NotifyPushEvent. Both receivers are
    decorated heavily to reduce the amount of duplicated code to zero.

    Usage:

    >>> handlers = EventHandlers()
    >>> EventListener("0.0.0.0:1234", handlers).start().block()

    gRPC calls are operated in a separate thread pool. Thus the main thread has nothing to do
    and needs to be suspended.
    """
    def __init__(self, address: str, handlers: EventHandlers, n_workers: int=1):
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
        """
        Starts the gRPC server. Does *not* block.

        :return: self
        """
        self._server.start()
        return self

    def block(self):
        """
        Blocks the calling thread until a KeyboardInterrupt is triggered.

        :return: None
        """
        self._stop_event.clear()
        try:
            self._stop_event.wait()
        except KeyboardInterrupt:
            pass

    def stop(self, cancel_running=False):
        """
        Forces the gRPC server to terminate.

        :param cancel_running: If True, performs a very impolite and sudden termination of all \
                               the threads in the thread pool.
        :return: None
        """
        self._stop_event.set()
        self._server.stop(None if cancel_running else 0)

    def timeit(func):
        """
        Decorator which measures the elapsed time via `time.perf_counter()`.

        :return: The decorated function.
        """
        @functools.wraps(func)
        def wrapped_timeit(self, request, context: grpc.ServicerContext):
            start_time = time.perf_counter()
            context.start_time = start_time
            result = func(self, request, context)
            delta = time.perf_counter() - start_time
            self._log.info("OK %.3f", delta)
            return result

        return wrapped_timeit

    def set_logging_context(func):
        """
        Assigns the metadata of the current gRPC call to the thread-local logging context.
        Thus it is associated with the running thread in the pool until a new event is fired.

        :return: The decorated function.
        """
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
        """
        Performs the top-level exception handling. In case of an error, catches it gracefully
        and converts into a nicer gRPC error message.

        :return: The decorated function.
        """
        @functools.wraps(func)
        def wrapped_catch_them_all(self, request, context: grpc.ServicerContext):
            try:
                return func(self, request, context)
            except Exception as e:
                start_time = getattr(request, "start_time", None)
                if start_time is not None:
                    delta = time.perf_counter() - start_time
                    self._log.exception("FAIL %.3f", delta)
                else:
                    self._log.exception("FAIL ?")
                context.abort(grpc.StatusCode.INTERNAL, "%s: %s" % (type(e), e))

        return wrapped_catch_them_all

    def handle(func):
        """
        Runs the corresponding callback from `handlers`.

        :return: The decorated function.
        """
        @functools.wraps(func)
        def wrapped_handle(self, request, context: grpc.ServicerContext):
            method_name = "process_" + stringcase.snakecase(type(request).__name__)
            return getattr(self.handlers, method_name)(request)

        return wrapped_handle

    @set_logging_context
    @timeit
    @log_exceptions
    @handle
    def NotifyReviewEvent(self, request: ReviewEvent, context: grpc.ServicerContext) \
            -> EventResponse:
        """
        Fired on `ReviewEvent`-s. Returns `EventResponse`. See
        lookout/core/server/sdk/event.proto and lookout/core/server/sdk/service_analyzer.proto.

        The actual result is returned in `handle()` decorator.

        Called in a thread from the thread pool.
        """
        pass

    @set_logging_context
    @timeit
    @log_exceptions
    @handle
    def NotifyPushEvent(self, request: PushEvent, context: grpc.ServicerContext) -> EventResponse:
        """
        Fired on `PushEvent`-s. Returns nothing - we are not supposed to answer anything.

        The actual work is done in `handle()` decorator.

        Called in a thread from the thread pool.
        """
        pass

    timeit = staticmethod(timeit)
    set_logging_context = staticmethod(set_logging_context)
    log_exceptions = staticmethod(log_exceptions)
    handle = staticmethod(handle)
