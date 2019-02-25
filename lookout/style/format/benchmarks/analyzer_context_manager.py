from argparse import Namespace
from typing import Type

from lookout.core.analyzer import Analyzer
from lookout.core.cmdline import create_model_repo_from_args
from lookout.core.data_requests import DataService
from lookout.core.event_listener import EventListener
from lookout.core.manager import AnalyzerManager


class AnalyzerContextManager:
    """Context manager for launching analyzer."""

    def __init__(self, analyzer: Type[Analyzer], port: int, db: str, fs: str, init: bool = True,
                 data_request_address: str = "localhost:10301"):
        """
        Init analyzer: model_repository, data_service, arguments, etc.

        :param port: port to use for analyzer.
        :param db: path to sqlite database location.
        :param fs: location where to store results of launched analyzer.
        :param analyzer: analyzer class to use.
        :param init: To run `analyzer init` or not. \
                     If you want to reuse existing database set False.
        :param data_request_address: DataService GRPC endpoint to use.
        """
        self.analyzer = analyzer
        self.port = port
        self.init = init
        self.data_request_address = data_request_address
        self._sql_alchemy_model_args = Namespace(
            db="sqlite:///%s" % db,
            fs=fs,
            cache_size="1G",
            cache_ttl="6h",
            db_kwargs={},
        )

    def __enter__(self) -> "AnalyzerContextManager":
        self.model_repository = create_model_repo_from_args(self._sql_alchemy_model_args)
        if self.init:
            self.model_repository.init()
        self.data_service = DataService(self.data_request_address)
        self.manager = AnalyzerManager(analyzers=[self.analyzer],
                                       model_repository=self.model_repository,
                                       data_service=self.data_service)
        self.listener = EventListener(address="0.0.0.0:%d" % self.port, handlers=self.manager,
                                      n_workers=1)
        self.listener.start()
        return self

    def __exit__(self, exc_type=None, exc_val=None, exc_tb=None):
        self.listener.stop()
        self.model_repository.shutdown()
        self.data_service.shutdown()
