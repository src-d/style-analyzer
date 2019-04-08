"""
Defines TyposCorrectorManager - the storage of TyposCorrector instances which is intended \
to be used globally.
"""
from typing import Iterable

import modelforge.backends

from lookout.style.typos.corrector import TyposCorrector


class TyposCorrectorManager:
    """
    Maintains the mapping from model sources to the corresponding loaded TyposCorrector objects.
    """

    def __init__(self):
        """Initialize a new instance of TyposCorrectorManager."""
        self.correctors = {}

    def get(self, source: str, processes_number: int) -> TyposCorrector:
        """
        Return the loaded TyposCorrector model for the specified model source.

        :param source: UUID or file path.
        :param processes_number: Number of processes for multiprocessing.
        :return: TyposCorrector.
        """
        obj = self.correctors.get(source)
        if obj is None:
            obj = TyposCorrector().load(source, backend=modelforge.backends.create_backend())
            obj.processes_number = processes_number
            self.correctors[source] = obj
        return obj

    def warmup(self, sources: Iterable[str]):
        """
        Force load several models specified in the arguments.

        :param sources: Model sources.
        :return: Nothing
        """
        for source in sources:
            self.get(source)
