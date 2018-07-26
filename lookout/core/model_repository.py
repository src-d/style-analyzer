import modelforge


class ModelRepository:
    def get(self, model_id: str, url) -> modelforge.Model:
        raise NotImplementedError

    def set(self, model_id: str, url, model: modelforge.Model):
        raise NotImplementedError
