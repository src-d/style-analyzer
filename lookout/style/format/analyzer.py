import logging

from bblfsh import Node

from lookout.core.analyzer import Analyzer, AnalyzerModel
from lookout.core.api.service_analyzer_pb2 import Comment
from lookout.core.api.service_data_pb2_grpc import DataStub
from lookout.core.data_requests import with_changed_uasts_and_contents, with_uasts_and_contents
from lookout.style.format.model import FormatModel


class FormatAnalyzer(Analyzer):
    log = logging.getLogger("FormatAnalyzer")
    model_type = FormatModel
    version = "1"
    description = "Source code formatting: whitespace, new lines, quotes, braces."

    @with_changed_uasts_and_contents
    def analyze(self, commit_from: str, commit_to: str, data_request_stub: DataStub,
                **data) -> [Comment]:
        changes = data["changes"]
        comments = []
        for change in changes:
            comment = Comment()
            comment.file = change.head.path
            comment.text = "%s %d > %d" % (change.head.language,
                                           self.count_nodes(change.base.uast),
                                           self.count_nodes(change.head.uast))
            comment.line = 1
            comment.confidence = 100
            comments.append(comment)
        return comments

    @classmethod
    @with_uasts_and_contents
    def train(cls, url: str, commit: str, config: dict, data_request_stub: DataStub,
              **data) -> AnalyzerModel:
        cls.log.info("train %s %s %s", url, commit, data)
        files = data["files"]
        for file in files:
            cls.log.info("%s %s %d", file.path, file.language, len(file.uast.children))

        """
        Plan:

        1. Load all the File-s - they are streamed and we need to read them all into a list
        2. Sort by language
        3. Loop per language group:
        4. extract_features()
        5. train the base model - tree or forest
        6. train the rules
        7. generate the Modelforge model
        """

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
