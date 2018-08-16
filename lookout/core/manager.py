import functools
import logging
import threading
from typing import Iterable, Type, Sequence

import grpc

from lookout.core.analyzer import Analyzer
from lookout.core.api.event_pb2 import ReviewEvent, PushEvent
from lookout.core.api.service_analyzer_pb2 import EventResponse
from lookout.core.api.service_data_pb2_grpc import DataStub
from lookout.core.event_listener import EventHandlers
from lookout.core.model_repository import ModelRepository


class AnalyzerManager(EventHandlers):
    """
    Manages several `Analyzer`-s: runs them and trains the models.

    Relies on a `ModelRepository` to retrieve and update the models. Also requires the address
    of the data (UAST, contents) gRPC service, typically running in the same Lookout server.
    """
    _log = logging.getLogger("AnalyzerManager")

    def __init__(self, model_repository: ModelRepository, analyzers: Iterable[Type[Analyzer]],
                 data_request_address: str):
        """
        Initializes a new instance of the AnalyzerManager class.

        :param model_repository: Injected implementor of the `ModelRepository` interface.
        :param analyzers: Analyzer types to manage (not instances!).
        :param data_request_address: gRPC address of the data retrieval service.
        """
        self._model_repository = model_repository
        analyzers = [(a.__name__, a) for a in analyzers]
        analyzers.sort()
        self._analyzers = [a[1] for a in analyzers]
        self._data_request_stub = threading.local()
        self._data_request_address = data_request_address

    def __str__(self) -> str:
        return "AnalyzerManager(%s)" % self.version

    @property
    def version(self) -> str:
        """
        Version depends on all the managed analyzers.
        """
        return " ".join(self._model_id(a) for a in self._analyzers)

    def with_data_request_stub(func):
        """
        Lazily creates the gRPC data retrieval API instance (stub). We use the thread-local
        storage as this method is called from `EventListener`.

        :return: The decorated function.
        """
        @functools.wraps(func)
        def wrapped_with_data_request_stub(self, request):
            if not hasattr(self._data_request_stub, "stub"):
                channel = grpc.insecure_channel(self._data_request_address, options=[
                    ("grpc.max_send_message_length", 100 * 1024 * 1024),
                    ("grpc.max_receive_message_length", 100 * 1024 * 1024),
                ])
                self._data_request_stub.stub = DataStub(channel)
            return func(self, request)

        return wrapped_with_data_request_stub

    @with_data_request_stub
    def process_review_event(self, request: ReviewEvent) -> EventResponse:
        base_url = request.commit_revision.base.internal_repository_url
        external_url = request.commit_revision.head.internal_repository_url
        commit_head = request.commit_revision.head.hash
        commit_base = request.commit_revision.base.hash
        configuration = request.configuration
        response = EventResponse()
        response.analyzer_version = self.version
        comments = []
        for analyzer in self._analyzers:
            mycfg = getattr(configuration, analyzer.__name__, {})
            model, cache_miss = self._model_repository.get(
                self._model_id(analyzer), analyzer.model_type, base_url)
            if cache_miss:
                self._log.info("cache miss: %s", analyzer.__name__)
            if model is None:
                self._log.info("training: %s", analyzer.__name__)
                model = analyzer.train(base_url, commit_base, mycfg, self._data_request_stub.stub)
                self._model_repository.set(self._model_id(analyzer), base_url, model)
            self._log.debug("running %s", analyzer.__name__)
            results = analyzer(model, external_url, mycfg).analyze(
                commit_base, commit_head, self._data_request_stub.stub)
            self._log.info("%s: %d comments", analyzer.__name__, len(results))
            comments.extend(results)
        response.comments.extend(comments)
        return response

    @with_data_request_stub
    def process_push_event(self, request: PushEvent) -> EventResponse:
        url = request.commit_revision.head.internal_repository_url
        commit = request.commit_revision.head.hash
        configuration = request.configuration
        for analyzer in self._analyzers:
            self._log.debug("training %s", analyzer.__name__)
            mycfg = configuration.fields.get(analyzer.__name__, {})
            model = analyzer.train(url, commit, mycfg, self._data_request_stub.stub)
            self._model_repository.set(self._model_id(analyzer), url, model)
        return EventResponse()

    def warmup(self, urls: Sequence[str]):
        """
        Warms up the model cache (which supposedly exists in the injected `ModelRepository`).
        We get the models corresponding to the managed analyzers and the specified list of
        repositories.

        :param urls: The list of Git repositories for which to fetch the models.
        """
        self._log.info("warming up on %d urls", len(urls))
        for url in urls:
            for analyzer in self._analyzers:
                self._model_repository.get(self._model_id(analyzer), analyzer.model_type, url)

    with_data_request_stub = staticmethod(with_data_request_stub)

    @staticmethod
    def _model_id(analyzer: Type[Analyzer]) -> str:
        return "%s/%s" % (analyzer.__name__, analyzer.version)
