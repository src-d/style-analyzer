from typing import Any, Dict, Mapping, NamedTuple

from modelforge import Model

from lookout.core.api.event_pb2 import ReferencePointer as ApiReferencePointer
from lookout.core.api.service_analyzer_pb2 import Comment
from lookout.core.api.service_data_pb2_grpc import DataStub
from lookout.core.ports import Type

# We redefine ReferencePointer because Protocol Buffers message objects suck.
ReferencePointer = NamedTuple("ReferencePointer", (("url", str), ("ref", str), ("commit", str)))
ReferencePointer.from_pb = lambda refptr: ReferencePointer(*[f[1] for f in refptr.ListFields()])
ReferencePointer.to_pb = lambda self: ApiReferencePointer(internal_repository_url=self.url,
                                                          reference_name=self.ref,
                                                          hash=self.commit)


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
        self.ptr = ReferencePointer("<unknown url>", "<unknown reference>", "<unknown commit>")

    def construct(self, analyzer: Type["Analyzer"], ptr: ReferencePointer):
        """
        Initialization of the model (__init__ is empty to allow load()).

        :param analyzer: Bound type of the `Analyzer`. Not instance!
        :param ptr: Git repository state pointer.
        :return: self
        """
        assert isinstance(self, analyzer.model_type)
        self.name = analyzer.name
        self.version = [analyzer.version]
        self.ptr = ptr
        return self

    def dump(self) -> str:
        """
        Implements the upstream abstract method.

        :return: summary text of the model.
        """
        return "%s/%s %s %s" % (self.name, self.version, self.ptr.url, self.ptr.commit)


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
    name = None  # type: str

    def __init__(self, model: AnalyzerModel, url: str, config: Mapping[str, Any]):
        """
        :param model: The instance of the model loaded from the repository or freshly trained.
        :param url: The analyzed project's Git remote.
        :param config: Configuration of the analyzer of unspecified structure.
        """
        self.model = model
        self.url = url
        self.config = config

    def analyze(self, ptr_from: ReferencePointer, ptr_to: ReferencePointer,
                data_request_stub: DataStub, **data) -> [Comment]:
        """
        This is called on Review events. It must return the list of `Comment`-s - found review
        suggestions.

        :param ptr_from: The Git revision of the fork point. Exists in both the original and \
                         the forked repositories.
        :param ptr_to: The Git revision to analyze. Exists only in the forked repository.
        :param data_request_stub: The channel to the data service in Lookout server to query for \
                                  UASTs, file contents, etc.
        :param data: Extra data passed into the method. Used by the decorators to simplify \
                     the data retrieval.
        :return: List of found review suggestions. Refer to \
                 lookout/core/server/sdk/service_analyzer.proto.
        """
        raise NotImplementedError

    @classmethod
    def train(cls, ptr: ReferencePointer, config: Dict[str, Any], data_request_stub: DataStub,
              **data) -> AnalyzerModel:
        """
        Generates a new model on top of the specified source code.

        :param ptr: Git repository state pointer.
        :param config: Configuration of the training of unspecified structure.
        :param data_request_stub: The channel to the data service in Lookout server to query for \
                                  UASTs, file contents, etc.
        :param data: Extra data passed into the method. Used by the decorators to simplify \
                     the data retrieval.
        :return: Instance of `AnalyzerModel` (`model_type`, to be precise).
        """
        raise NotImplementedError

    @classmethod
    def construct_model(cls, ptr: ReferencePointer) -> AnalyzerModel:
        return cls.model_type().construct(cls, ptr)
