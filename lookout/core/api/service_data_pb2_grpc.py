# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
import grpc

from lookout.core.api import service_data_pb2 as lookout_dot_core_dot_api_dot_service__data__pb2


class DataStub(object):
  """Data services exposes VCS repositories.
  """

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.GetChanges = channel.unary_stream(
        '/pb.Data/GetChanges',
        request_serializer=lookout_dot_core_dot_api_dot_service__data__pb2.ChangesRequest.SerializeToString,
        response_deserializer=lookout_dot_core_dot_api_dot_service__data__pb2.Change.FromString,
        )
    self.GetFiles = channel.unary_stream(
        '/pb.Data/GetFiles',
        request_serializer=lookout_dot_core_dot_api_dot_service__data__pb2.FilesRequest.SerializeToString,
        response_deserializer=lookout_dot_core_dot_api_dot_service__data__pb2.File.FromString,
        )


class DataServicer(object):
  """Data services exposes VCS repositories.
  """

  def GetChanges(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def GetFiles(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_DataServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'GetChanges': grpc.unary_stream_rpc_method_handler(
          servicer.GetChanges,
          request_deserializer=lookout_dot_core_dot_api_dot_service__data__pb2.ChangesRequest.FromString,
          response_serializer=lookout_dot_core_dot_api_dot_service__data__pb2.Change.SerializeToString,
      ),
      'GetFiles': grpc.unary_stream_rpc_method_handler(
          servicer.GetFiles,
          request_deserializer=lookout_dot_core_dot_api_dot_service__data__pb2.FilesRequest.FromString,
          response_serializer=lookout_dot_core_dot_api_dot_service__data__pb2.File.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'pb.Data', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))
