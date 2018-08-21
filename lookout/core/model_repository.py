from typing import Tuple, Type

from lookout.core.analyzer_model import AnalyzerModel


class ModelRepository:
    """
    Interface to retrieve and update the ML models. Injected into `AnalyzerManager`.
    """
    def get(self, model_id: str, model_type: Type[AnalyzerModel],
            url: str) -> Tuple[AnalyzerModel, bool]:
        """
        Returns the model for the specified key (`model_id`) and the repository (`url`).
        `model_type` is used to return the correct class instance.

        :param model_id: The key of the model (based on the bound analyzer name and version).
        :param model_type: Class of the model to return the instance of.
        :param url: Git repository remote.
        :return: an `AnalyzerModel` instance and the boolean which indicates whether \
                 the cache miss happened.
        """
        raise NotImplementedError

    def set(self, model_id: str, url: str, model: AnalyzerModel):
        """
        Puts the new model into the storage for the specified key (`model_id`) and the
        repository (`url`).

        :param model_id: The key of the model (based on the bound analyzer name and version).
        :param url: Git repository remote.
        :param model: The instance of the model to store.
        :return: None
        """
        raise NotImplementedError

    def init(self):
        """
        Initializes the persistent data structures of this storage.
        :return: None
        """
        raise NotImplementedError

    def shutdown(self):
        """
        Frees the resources allocated by the storage.
        """
        raise NotImplementedError
