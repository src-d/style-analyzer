import functools
import threading
from typing import Iterable

import grpc

from lookout.core.analyzer import Analyzer, AnalyzerModel
from lookout.core.api.service_analyzer_pb2 import Comment
from lookout.core.api.service_data_pb2 import ChangesRequest, FilesRequest, Change, File
from lookout.core.api.service_data_pb2_grpc import DataStub


class DataService:
    """
    Retrieves UASTs/files from the Lookout server.
    """
    GRPC_MAX_MESSAGE_SIZE = 100 * 1024 * 1024

    def __init__(self, address: str):
        self._data_request_local = threading.local()
        self._data_request_channels = []
        self._data_request_address = address

    def get(self) -> DataStub:
        """
        Returns a `DataStub` for the current thread.
        """
        stub = getattr(self._data_request_local, "stub", None)
        if stub is None:
            channel = grpc.insecure_channel(self._data_request_address, options=[
                ("grpc.max_send_message_length", self.GRPC_MAX_MESSAGE_SIZE),
                ("grpc.max_receive_message_length", self.GRPC_MAX_MESSAGE_SIZE),
            ])
            self._data_request_channels.append(channel)
            self._data_request_local.stub = stub = DataStub(channel)
        return stub

    def shutdown(self):
        """
        Closes all the open network connections.
        """
        for channel in self._data_request_channels:
            channel.close()


def with_changed_uasts(func):
    """
    Use this decorator to provide "changes" keyword argument to `**data` in `Analyzer.analyze()`.
    "changes" contain the list of `Change` - see lookout/core/server/sdk/service_data.proto.
    The changes will have only UASTs.

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


def with_changed_uasts_and_contents(func):
    """
    Use this decorator to provide "changes" keyword argument to `**data` in `Analyzer.analyze()`.
    "changes" contain the list of `Change` - see lookout/core/server/sdk/service_data.proto.
    The changes will have both UASTs and raw file contents.

    :param func: Method with the signature compatible with `Analyzer.analyze()`.
    :return: The decorated method.
    """
    @functools.wraps(func)
    def wrapped_with_changed_uasts_and_contents(
            self: Analyzer, commit_from: str, commit_to: str,
            data_request_stub: DataStub, **data) -> [Comment]:
        changes = request_changes(data_request_stub, self.url, commit_from, commit_to,
                                  contents=True, uast=True)
        return func(self, commit_from, commit_to, data_request_stub, changes=changes, **data)

    return wrapped_with_changed_uasts_and_contents


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
                           data_request_stub: DataStub, **data) -> AnalyzerModel:
        files = request_files(data_request_stub, url, commit, contents=False, uast=True)
        return func(cls, url, commit, config, data_request_stub, files=files, **data)

    return wrapped_with_uasts


def with_uasts_and_contents(func):
    """
    Use this decorator to provide "files" keyword argument to `**data` in `Analyzer.train()`.
    "files" are the list of `File`-s with all the UASTs and raw file contents for the passed Git
    repository URL and revision, see lookout/core/server/sdk/service_data.proto.

    :param func: Method with the signature compatible with `Analyzer.train()`.
    :return: The decorated method.
    """
    @functools.wraps(func)
    def wrapped_with_uasts_and_contents(cls, url: str, commit: str, config: dict,
                                        data_request_stub: DataStub, **data) -> AnalyzerModel:
        files = request_files(data_request_stub, url, commit, contents=True, uast=True)
        return func(cls, url, commit, config, data_request_stub, files=files, **data)

    return wrapped_with_uasts_and_contents


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
    request.exclude_vendored = True
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
    request.exclude_vendored = True
    request.want_contents = contents
    request.want_uast = uast
    return stub.GetFiles(request)
