"""
Defines TyposCorrectorManager - the storage of TyposCorrector instances which is intended \
to be used globally.
"""

from typing import Iterable

from lookout.style.typos.corrector import TyposCorrector


class TyposCorrectorManager:
    """
    Maintains the mapping from model sources to the corresponding loaded TyposCorrector objects.
    """

    def __init__(self):
        """Initialize a new instance of TyposCorrectorManager."""
        self.correctors = {}

    def get(self, source: str) -> TyposCorrector:
        """
        Return the loaded TyposCorrector model for the specified model source.

        :param source: UUID or file path.
        :return: TyposCorrector.
        """
        obj = self.correctors.get(source)
        if obj is None:
            obj = TyposCorrector(threads_number=1).load(source)
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
