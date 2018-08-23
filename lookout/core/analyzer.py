from typing import Type  # noqa: F401

from modelforge import Model

from lookout.core.api.service_analyzer_pb2 import Comment
from lookout.core.api.service_data_pb2_grpc import DataStub


class AnalyzerModel(Model):
    """
    All models used in `Analyzer`-s must derive from this base class.
    """
    def __init__(self, **kwargs):
        """
        Defines:
        `name` - name of the model. Corresponds to the bound analyzer's class name and version.
        `url` - Git repository on which the model was trained.
        `commit` - revision of the Git repository on which the model was trained.
        :param kwargs: passed to the upstream __init__.
        """
        super().__init__(**kwargs)
        self.name = "<unknown name>"
        self.url = "<unknown url>"
        self.commit = "<unknown commit>"

    def construct(self, analyzer: Type["Analyzer"], url: str, commit: str):
        """
        Initialization of the model (__init__ is empty to allow load()).

        :param analyzer: Bound type of the `Analyzer`. Not instance!
        :param url: Git repository on which the model was trained.
        :param commit: revision of the Git repository on which the model was trained.
        :return: self
        """
        assert isinstance(self, analyzer.model_type)
        self.name = analyzer.__name__
        self.version = [analyzer.version]
        self.url = url
        self.commit = commit
        return self

    def dump(self) -> str:
        """
        Implements the upstream abstract method.

        :return: summary text of the model.
        """
        return "%s/%s %s %s" % (self.name, self.version, self.url, self.commit)


class Analyzer:
    """
    Interface of all the analyzers. Each analyzer uses a model to run the analysis and generates
    a model as the result of the training.

    `version` allows to version the models. It is checked in the model repository and if it does
    not match, a new model is trained.
    `model_type` points to the specific derivative of AnalyzerModel - type of the model used
    in analyze() and generated in train().
    """
    version = None  # type: str
    model_type = None  # type: Type[AnalyzerModel]

    def __init__(self, model: AnalyzerModel, url: str, config: dict):
        """
        :param model: The instance of the model loaded from the repository or freshly trained.
        :param url: The analyzed project's Git remote.
        :param config: Configuration of the analyzer of unspecified structure.
        """
        self.model = model
        self.url = url
        self.__dict__.update(config)

    def analyze(self, commit_from: str, commit_to: str, data_request_stub: DataStub,
                **data) -> [Comment]:
        """
        This is called on Review events. It must return the list of `Comment`-s - found review
        suggestions.

        :param commit_from: The Git revision of the fork point. Exists in both original and \
                            forked repositories.
        :param commit_to: The Git revision to analyze. Exists only in the forked repository.
        :param data_request_stub: The channel to the data service in Lookout server to query for \
                                  UASTs, file contents, etc.
        :param data: Extra data passed into the method. Used by the decorators to simplify \
                     the data retrieval.
        :return: List of found review suggestions. Refer to \
                 lookout/core/server/sdk/service_analyzer.proto.
        """
        raise NotImplementedError

    @classmethod
    def train(cls, url: str, commit: str, config: dict, data_request_stub: DataStub,
              **data) -> AnalyzerModel:
        """
        Generates a new model on top of the specified source code.

        :param url: Git repository remote.
        :param commit: Hash to checkout.
        :param config: Configuration of the training of unspecified structure.
        :param data_request_stub: The channel to the data service in Lookout server to query for \
                                  UASTs, file contents, etc.
        :param data: Extra data passed into the method. Used by the decorators to simplify \
                     the data retrieval.
        :return: Instance of `AnalyzerModel` (`model_type`, to be precise).
        """
        raise NotImplementedError

    @classmethod
    def construct_model(cls, url: str, commit: str) -> AnalyzerModel:
        return cls.model_type().construct(cls, url, commit)
