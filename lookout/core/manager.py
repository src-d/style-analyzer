import logging
from typing import Iterable, Type, Sequence

from lookout.core.analyzer import Analyzer
from lookout.core.api.event_pb2 import ReviewEvent, PushEvent
from lookout.core.api.service_analyzer_pb2 import EventResponse
from lookout.core.model_repository import ModelRepository


class AnalyzerManager:
    _log = logging.getLogger("AnalyzerManager")

    def __init__(self, model_repository: ModelRepository, analyzers: Iterable[Type[Analyzer]]):
        self._model_repository = model_repository
        analyzers = [(a.__name__, a) for a in analyzers]
        analyzers.sort()
        self._analyzers = [a[1] for a in analyzers]

    @property
    def version(self):
        return " ".join(self._model_id(a) for a in self._analyzers)

    def process_review_event(self, request: ReviewEvent):
        base_url = request.commit_revision.base.internal_repository_url
        external_url = request.commit_revision.head.internal_repository_url
        commit_head = request.commit_revision.head.Hash
        commit_base = request.commit_revision.base.Hash
        configuration = request.configuration
        response = EventResponse()
        response.analyzer_version = self.version
        comments = []
        for analyzer in self._analyzers:
            mycfg = configuration.get(analyzer.__name__, {})
            model, cache_miss = self._model_repository.get(self._model_id(analyzer), base_url)
            if cache_miss:
                self._log.info("cache miss: %s", analyzer.__name__)
            if model is None:
                self._log.info("training: %s", analyzer.__name__)
                model = analyzer.train(base_url, commit_base, mycfg)
                self._model_repository.set(self._model_id(analyzer), base_url, model)
            self._log.debug("running %s", analyzer.__name__)
            results = analyzer(model, external_url, mycfg).analyze(commit_base, commit_head)
            self._log.info("%s: %d comments", analyzer.__name__, len(results))
            comments.extend(results)
        response.comments.extend(comments)
        return response

    def process_push_event(self, request: PushEvent):
        url = request.commit_revision.head.internal_repository_url
        commit = request.commit_revision.head.Hash
        configuration = request.configuration
        for analyzer in self._analyzers:
            self._log.debug("training %s", analyzer.__name__)
            mycfg = configuration.get(analyzer.__name__, {})
            model = analyzer.train(url, commit, mycfg)
            self._model_repository.set(self._model_id(analyzer), url, model)

    def warmup(self, urls: Sequence[str]):
        self._log.info("warming up on %d urls", len(urls))
        for url in urls:
            for analyzer in self._analyzers:
                self._model_repository.get(self._model_id(analyzer), url)

    @staticmethod
    def _model_id(analyzer: Type[Analyzer]) -> str:
        return "%s/%s" % (analyzer.__name__, analyzer.__version__)
