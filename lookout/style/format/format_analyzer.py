import importlib
import logging

from lookout.core.analyzer import Analyzer
from lookout.core.analyzer_model import AnalyzerModel
from lookout.core.api.service_analyzer_pb2 import Comment
from lookout.core.api.service_data_pb2_grpc import DataStub
Node = importlib.import_module("lookout.core.api.gopkg.in.bblfsh.sdk.v1.uast.generated_pb2").Node
from lookout.core.data_requests import with_changed_uasts  # noqa: E401


class FormatModel(AnalyzerModel):
    def _generate_tree(self) -> dict:
        return {}

    def _load_tree(self, tree: dict) -> None:
        pass


class FormatAnalyzer(Analyzer):
    log = logging.getLogger("FormatAnalyzer")
    model_type = FormatModel
    version = "1"
    description = "Source code formatting: whitespace, new lines, quotes, braces."

    @with_changed_uasts
    def analyze(self, commit_from: str, commit_to: str, data_request_stub: DataStub,
                **data) -> [Comment]:
        changes = data["uast_changes"]
        comments = []
        for change in changes:
            comment = Comment()
            comment.file = change.head.path
            comment.text = "%d > %d" % (self.count_nodes(change.base.uast),
                                        self.count_nodes(change.head.uast))
            comment.line = 1
            comment.confidence = 100
            comments.append(comment)
        return comments

    @classmethod
    def train(cls, url: str, commit: str, config: dict, data_request_stub: DataStub, **data):
        cls.log.info("train %s %s %s", url, commit, data)
        return FormatModel().construct(cls, url, commit)

    @staticmethod
    def count_nodes(uast: Node):
        stack = [uast]
        count = 0
        while stack:
            node = stack.pop()
            count += 1
            stack.extend(node.children)
        return count
