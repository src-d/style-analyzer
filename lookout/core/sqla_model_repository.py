import modelforge

from lookout.core.model_repository import ModelRepository


class SQLAlchemyModelRepository(ModelRepository):
    def __init__(self):
        pass

    def get(self, model_id: str, url) -> modelforge.Model:
        raise NotImplementedError

    def set(self, model_id: str, url, model: modelforge.Model):
        raise NotImplementedError
