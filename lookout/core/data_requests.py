import functools

from lookout.core.analyzer import Analyzer
from lookout.core.api.service_analyzer_pb2 import Comment
from lookout.core.api.service_data_pb2 import ChangesRequest
from lookout.core.api.service_data_pb2_grpc import DataStub


def with_changed_uasts(func):
    @functools.wraps(func)
    def wrapped_with_changed_uasts(self: Analyzer, commit_from: str, commit_to: str,
                                   data_request_stub: DataStub) -> [Comment]:
        changes = request_changes(data_request_stub, self.url, commit_from, commit_to,
                                  contents=False, uast=True)
        return func(self, commit_from, commit_to, data_request_stub, uast_changes=changes)

    return wrapped_with_changed_uasts


def request_changes(stub: DataStub, url: str, commit_from: str, commit_to: str,
                    contents: bool, uast: bool):
    request = ChangesRequest()
    request.base.internal_repository_url = url
    request.base.hash = commit_from
    request.head.internal_repository_url = url
    request.head.hash = commit_to
    request.exclude_vendored = True
    request.want_contents = contents
    request.want_uast = uast
    return stub.GetChanges(request)
