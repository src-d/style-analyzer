import logging

import bblfsh
from sourced.ml.algorithms import uast2sequence

from lookout.core.analyzer import Analyzer, AnalyzerModel, ReferencePointer
from lookout.core.api.service_analyzer_pb2 import Comment
from lookout.core.api.service_data_pb2_grpc import DataStub
from lookout.core.data_requests import with_changed_uasts_and_contents, with_uasts_and_contents
from lookout.style.typos.model import IdentifiersTyposModel


class IdentifiersTyposAnalyzer(Analyzer):
    log = logging.getLogger("IdentifiersTyposAnalyzer")
    model_type = IdentifiersTyposModel
    version = "1"
    description = "Correcting typos in identifiers."

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @with_changed_uasts_and_contents
    def analyze(self, ptr_from: ReferencePointer, ptr_to: ReferencePointer,
                data_request_stub: DataStub, **data) -> [Comment]:
        changes = data["changes"]
        comments = []
        for change in changes:
            old_identifiers = set([node.token for node in uast2sequence(change.base.uast)
                                   if bblfsh.role_id("IDENTIFIER") in node.roles and
                                   bblfsh.role_id("IMPORT") not in node.roles and
                                   len(node.token) > 0])
            identifiers_nodes = [node for node in uast2sequence(change.head.uast)
                                 if bblfsh.role_id("IDENTIFIER") in node.roles and
                                 bblfsh.role_id("IMPORT") not in node.roles and
                                 len(node.token) > 0 and node.token not in old_identifiers]
            identifiers = [node.token for node in identifiers_nodes]

            if len(identifiers) > 0:
                suggestions = self.model.check_identifiers(identifiers)
                for index in suggestions.keys():
                    corrections = suggestions[index]
                    for token in corrections.keys():
                        comment = Comment()
                        comment.file = change.head.path
                        corrections_line = ""
                        for candidate in corrections[token]:
                            corrections_line += candidate[0] + \
                                                " (%d)," % \
                                                int(candidate[1] * 100)

                        comment.text = "Typo inside identifier '%s' " \
                                       "in token '%s'. " \
                                       "Possible corrections:" % \
                                       (identifiers[index], token) +\
                                       corrections_line[:-1]
                        comment.line = identifiers_nodes[index].start_position.line
                        comment.confidence = int(corrections[token][0][1] * 100)
                        comments.append(comment)
        return comments

    @classmethod
    @with_uasts_and_contents
    def train(cls, ptr: ReferencePointer, config: dict, data_request_stub: DataStub,
              **data) -> AnalyzerModel:
        cls.log.info("train %s %s %s", ptr.url, ptr.commit, data)
        files = data["files"]
        for file in files:
            cls.log.info("%s %s %d", file.path, file.language, len(file.uast.children))
        model = IdentifiersTyposModel().construct(cls, ptr)
        return model.train()
