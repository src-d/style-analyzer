import logging
from typing import Iterable, Sequence

from lookout.core.analyzer import Analyzer, ReferencePointer
from lookout.core.api.event_pb2 import ReviewEvent, PushEvent
from lookout.core.api.service_analyzer_pb2 import EventResponse
from lookout.core.data_requests import DataService
from lookout.core.event_listener import EventHandlers
from lookout.core.model_repository import ModelRepository
from lookout.core.ports import Type


class AnalyzerManager(EventHandlers):
    """
    Manages several `Analyzer`-s: runs them and trains the models.

    Relies on a `ModelRepository` to retrieve and update the models. Also requires the address
    of the data (UAST, contents) gRPC service, typically running in the same Lookout server.
    """
    _log = logging.getLogger("AnalyzerManager")

    def __init__(self, analyzers: Iterable[Type[Analyzer]], model_repository: ModelRepository,
                 data_service: DataService):
        """
        Initializes a new instance of the AnalyzerManager class.

        :param analyzers: Analyzer types to manage (not instances!).
        :param model_repository: Injected implementor of the `ModelRepository` interface.
        :param data_service: gRPC data retrieval service to fetch UASTs and files.
        """
        self._model_repository = model_repository
        analyzers = [(a.__name__, a) for a in analyzers]
        analyzers.sort()
        self._analyzers = [a[1] for a in analyzers]
        self._data_service = data_service

    def __str__(self) -> str:
        return "AnalyzerManager(%s)" % self.version

    @property
    def version(self) -> str:
        """
        Version depends on all the managed analyzers.
        """
        return " ".join(self._model_id(a) for a in self._analyzers)

    def process_review_event(self, request: ReviewEvent) -> EventResponse:
        base_ptr = ReferencePointer.from_pb(request.commit_revision.base)
        head_ptr = ReferencePointer.from_pb(request.commit_revision.head)
        response = EventResponse()
        response.analyzer_version = self.version
        comments = []
        for analyzer in self._analyzers:
            try:
                mycfg = dict(request.configuration[analyzer.__name__])
                self._log.info("%s config: %s", analyzer.__name__, mycfg)
            except (KeyError, ValueError):
                mycfg = {}
                self._log.debug("no config was provided for %s", analyzer.__name__)
            model, cache_miss = self._model_repository.get(
                self._model_id(analyzer), analyzer.model_type, base_ptr.url)
            if cache_miss:
                self._log.info("cache miss: %s", analyzer.__name__)
            if model is None:
                self._log.info("training: %s", analyzer.__name__)
                model = analyzer.train(base_ptr, mycfg, self._data_service.get())
                self._model_repository.set(self._model_id(analyzer), base_ptr.url, model)
            self._log.debug("running %s", analyzer.__name__)
            results = analyzer(model, head_ptr.url, mycfg).analyze(
                base_ptr, head_ptr, self._data_service.get())
            self._log.info("%s: %d comments", analyzer.__name__, len(results))
            comments.extend(results)
        response.comments.extend(comments)
        return response

    def process_push_event(self, request: PushEvent) -> EventResponse:
        ptr = ReferencePointer.from_pb(request.commit_revision.head)
        for analyzer in self._analyzers:
            self._log.debug("training %s", analyzer.__name__)
            try:
                mycfg = dict(request.configuration[analyzer.__name__])
            except (KeyError, ValueError):
                mycfg = {}
            model = analyzer.train(ptr, mycfg, self._data_service.get())
            self._model_repository.set(self._model_id(analyzer), ptr.url, model)
        response = EventResponse()
        response.analyzer_version = self.version
        return response

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

    @staticmethod
    def _model_id(analyzer: Type[Analyzer]) -> str:
        return "%s/%s" % (analyzer.__name__, analyzer.version)
