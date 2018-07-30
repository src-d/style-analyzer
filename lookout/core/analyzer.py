from typing import Type  # noqa: F401

from modelforge import Model

from lookout.core.api.service_analyzer_pb2 import Comment
from lookout.core.api.service_data_pb2_grpc import DataStub


class Analyzer:
    version = None  # type: str
    model_type = None  # type: Type[Model]

    def __init__(self, model: Model, url: str, config: dict):
        self.model = model
        self.url = url
        self.config = config

    def analyze(self, commit_from: str, commit_to: str, data_request_stub: DataStub,
                **data) -> [Comment]:
        raise NotImplementedError

    @classmethod
    def train(cls, url: str, commit: str, config: dict, data_request_stub: DataStub,
              **data) -> Model:
        raise NotImplementedError
