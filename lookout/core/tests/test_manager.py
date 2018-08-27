from typing import Type, Tuple
import unittest

from lookout.core.analyzer import AnalyzerModel, Analyzer, ReferencePointer
from lookout.core.api.event_pb2 import ReviewEvent, PushEvent
from lookout.core.api.service_analyzer_pb2 import Comment, EventResponse
from lookout.core.api.service_data_pb2_grpc import DataStub
from lookout.core.manager import AnalyzerManager
from lookout.core.model_repository import ModelRepository


class FakeModel(AnalyzerModel):
    def _generate_tree(self) -> dict:
        return {}

    def _load_tree(self, tree: dict) -> None:
        pass


class FakeAnalyzer(Analyzer):
    version = "1"
    model_type = FakeModel

    def __init__(self, model: AnalyzerModel, url: str, config: dict):
        super().__init__(model, url, config)
        FakeAnalyzer.instance = self

    def analyze(self, ptr_from: ReferencePointer, ptr_to: ReferencePointer,
                data_request_stub: DataStub, **data) -> [Comment]:
        comment = Comment()
        comment.text = "%s|%s" % (ptr_from.commit, ptr_to.commit)
        FakeAnalyzer.stub = data_request_stub
        return [comment]

    @classmethod
    def train(cls, ptr: ReferencePointer, config: dict, data_request_stub: DataStub, **data
              ) -> AnalyzerModel:
        cls.stub = data_request_stub
        return FakeModel()


class FakeDataService:
    def get(self) -> DataStub:
        return "XXX"

    def shutdown(self):
        pass


class FakeModelRepository(ModelRepository):
    def __init__(self):
        self.get_calls = []
        self.set_calls = []

    def get(self, model_id: str, model_type: Type[AnalyzerModel], url: str
            ) -> Tuple[AnalyzerModel, bool]:
        self.get_calls.append((model_id, model_type, url))
        return FakeModel(), True

    def set(self, model_id: str, url: str, model: AnalyzerModel):
        self.set_calls.append((model_id, url, model))

    def init(self):
        pass

    def shutdown(self):
        pass


class AnalyzerManagerTests(unittest.TestCase):
    def setUp(self):
        self.data_service = FakeDataService()
        self.model_repository = FakeModelRepository()
        self.manager = AnalyzerManager(
            [FakeAnalyzer, FakeAnalyzer], self.model_repository, self.data_service)
        FakeAnalyzer.stub = None

    def test_process_review_event(self):
        request = ReviewEvent()
        request.configuration.update({"FakeAnalyzer": {"one": "two"}})
        request.commit_revision.base.internal_repository_url = "foo"
        request.commit_revision.base.reference_name = "refs/heads/master"
        request.commit_revision.base.hash = "00" * 20
        request.commit_revision.head.internal_repository_url = "bar"
        request.commit_revision.head.reference_name = "refs/heads/master"
        request.commit_revision.head.hash = "ff" * 20
        response = self.manager.process_review_event(request)
        self.assertIsInstance(response, EventResponse)
        self.assertEqual(response.analyzer_version, "FakeAnalyzer/1 FakeAnalyzer/1")
        self.assertEqual(len(response.comments), 2)
        self.assertEqual(*response.comments)
        self.assertEqual(response.comments[0].text, "%s|%s" % ("00" * 20, "ff" * 20))
        self.assertEqual(self.model_repository.get_calls,
                         [("FakeAnalyzer/1", FakeModel, "foo")] * 2)
        self.assertEqual(FakeAnalyzer.instance.one, "two")
        self.assertEqual(FakeAnalyzer.stub, "XXX")

    def test_process_push_event(self):
        request = PushEvent()
        request.commit_revision.head.internal_repository_url = "wow"
        request.commit_revision.head.reference_name = "refs/heads/master"
        request.commit_revision.head.hash = "80" * 20
        response = self.manager.process_push_event(request)
        self.assertIsInstance(response, EventResponse)
        self.assertEqual(response.analyzer_version, "FakeAnalyzer/1 FakeAnalyzer/1")
        self.assertEqual(len(response.comments), 0)
        self.assertEqual(len(self.model_repository.set_calls), 2)
        self.assertEqual(self.model_repository.set_calls[0][:2], ("FakeAnalyzer/1", "wow"))
        self.assertIsInstance(self.model_repository.set_calls[0][2], FakeModel)
        self.assertEqual(self.model_repository.set_calls[1][:2], ("FakeAnalyzer/1", "wow"))
        self.assertIsInstance(self.model_repository.set_calls[1][2], FakeModel)
        self.assertEqual(FakeAnalyzer.stub, "XXX")


if __name__ == "__main__":
    unittest.main()
