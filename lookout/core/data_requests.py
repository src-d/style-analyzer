import functools

import modelforge

from lookout.core.analyzer import Analyzer
from lookout.core.api.service_analyzer_pb2 import Comment
from lookout.core.api.service_data_pb2 import ChangesRequest, FilesRequest
from lookout.core.api.service_data_pb2_grpc import DataStub


def with_changed_uasts(func):
    @functools.wraps(func)
    def wrapped_with_changed_uasts(self: Analyzer, commit_from: str, commit_to: str,
                                   data_request_stub: DataStub, **data) -> [Comment]:
        changes = request_changes(data_request_stub, self.url, commit_from, commit_to,
                                  contents=False, uast=True)
        return func(self, commit_from, commit_to, data_request_stub, changes=changes, **data)

    return wrapped_with_changed_uasts


def with_uasts(func):
    @functools.wraps(func)
    def wrapped_with_uasts(cls, url: str, commit: str, config: dict,
                           data_request_stub: DataStub, **data) -> modelforge.Model:
        files = request_files(data_request_stub, url, commit, contents=False, uast=True)
        return func(cls, url, commit, config, data_request_stub, files=files, **data)

    return wrapped_with_uasts


def request_changes(stub: DataStub, url: str, commit_from: str, commit_to: str,
                    contents: bool, uast: bool):
    request = ChangesRequest()
    request.base.internal_repository_url = url
    request.base.hash = commit_from
    request.head.internal_repository_url = url
    request.head.hash = commit_to
    request.exclude_vendored = False
    # TODO(vmarkovtsev): change to True once https://github.com/src-d/lookout/pull/92 is merged
    request.want_contents = contents
    request.want_uast = uast
    return stub.GetChanges(request)


def request_files(stub: DataStub, url: str, commit: str, contents: bool, uast: bool):
    request = FilesRequest()
    request.revision.internal_repository_url = url
    request.revision.hash = commit
    request.exclude_vendored = False
    # TODO(vmarkovtsev): change to True once https://github.com/src-d/lookout/pull/92 is merged
    request.want_contents = contents
    request.want_uast = uast
    return stub.GetFiles(request)
