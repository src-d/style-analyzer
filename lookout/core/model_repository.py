from typing import Tuple, Type

import modelforge


class ModelRepository:
    def get(self, model_id: str, model_type: Type[modelforge.Model],
            url: str) -> Tuple[modelforge.Model, bool]:
        raise NotImplementedError

    def set(self, model_id: str, url: str, model: modelforge.Model):
        raise NotImplementedError

    def init(self):
        raise NotImplementedError
