from concurrent.futures import ThreadPoolExecutor
from threading import Event

import grpc

from lookout.core.api.service_analyzer_pb2_grpc import \
    AnalyzerServicer, add_AnalyzerServicer_to_server
from lookout.core.api.event_pb2 import ReviewEvent, PushEvent


class EventListener(AnalyzerServicer):
    def __init__(self, address: str, n_workers: int=1):
        self._server = grpc.server(ThreadPoolExecutor(max_workers=n_workers),
                                   maximum_concurrent_rpcs=n_workers)
        add_AnalyzerServicer_to_server(self, self._server)
        self._server.add_insecure_port(address)
        self._stop_event = Event()

    def start(self):
        self._server.start()

    def block(self):
        self._stop_event.clear()
        try:
            self._stop_event.wait()
        except KeyboardInterrupt:
            pass

    def stop(self, cancel_running=False):
        self._stop_event.set()
        self._server.stop(None if cancel_running else 0)

    def NotifyReviewEvent(self, request: ReviewEvent, context: grpc.ServicerContext):
        # missing associated documentation comment in .proto file
        pass
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def NotifyPushEvent(self, request: PushEvent, context: grpc.ServicerContext):
        # missing associated documentation comment in .proto file
        pass
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')
