import functools
from typing import Iterable

import modelforge

from lookout.core.analyzer import Analyzer
from lookout.core.api.service_analyzer_pb2 import Comment
from lookout.core.api.service_data_pb2 import ChangesRequest, FilesRequest, Change, File
from lookout.core.api.service_data_pb2_grpc import DataStub


def with_changed_uasts(func):
    """
    Use this decorator to provide "changes" keyword argument to `**data` in `Analyzer.analyze()`.
    "changes" contain the list of `Change` - see lookout/core/server/sdk/service_data.proto.

    :param func: Method with the signature compatible with `Analyzer.analyze()`.
    :return: The decorated method.
    """
    @functools.wraps(func)
    def wrapped_with_changed_uasts(self: Analyzer, commit_from: str, commit_to: str,
                                   data_request_stub: DataStub, **data) -> [Comment]:
        changes = request_changes(data_request_stub, self.url, commit_from, commit_to,
                                  contents=False, uast=True)
        return func(self, commit_from, commit_to, data_request_stub, changes=changes, **data)

    return wrapped_with_changed_uasts


def with_uasts(func):
    """
    Use this decorator to provide "files" keyword argument to `**data` in `Analyzer.train()`.
    "files" are the list of `File`-s with all the UASTs for the passed Git repository URL and
    revision, see lookout/core/server/sdk/service_data.proto.

    :param func: Method with the signature compatible with `Analyzer.train()`.
    :return: The decorated method.
    """
    @functools.wraps(func)
    def wrapped_with_uasts(cls, url: str, commit: str, config: dict,
                           data_request_stub: DataStub, **data) -> modelforge.Model:
        files = request_files(data_request_stub, url, commit, contents=False, uast=True)
        return func(cls, url, commit, config, data_request_stub, files=files, **data)

    return wrapped_with_uasts


def request_changes(stub: DataStub, url: str, commit_from: str, commit_to: str,
                    contents: bool, uast: bool) -> Iterable[Change]:
    """
    Used by `with_changed_uasts()`.

    :return: The stream of the gRPC invocation results. In theory, `.result()` would turn this \
             into a synchronous call, but in practice, that function call hangs for some reason.
    """
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


def request_files(stub: DataStub, url: str, commit: str, contents: bool,
                  uast: bool) -> Iterable[File]:
    """
    Used by `with_uasts()`.

    :return: The stream of the gRPC invocation results.
    """
    request = FilesRequest()
    request.revision.internal_repository_url = url
    request.revision.hash = commit
    request.exclude_vendored = False
    # TODO(vmarkovtsev): change to True once https://github.com/src-d/lookout/pull/92 is merged
    request.want_contents = contents
    request.want_uast = uast
    return stub.GetFiles(request)
