"""Identifier typo correction model."""

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from lookout.core.analyzer import AnalyzerModel
import pandas
from sourced.ml.algorithms import TokenParser

from lookout.style.typos.corrector import TyposCorrector
from lookout.style.typos.utils import flatten_data, SPLIT_COLUMN, TYPO_COLUMN

NODE_ID_COLUMN = "node_id"


class IdTyposModel(AnalyzerModel):
    """
    Model for the typo corrector in source code identifiers.
    """

    NAME = "typos"
    VENDOR = "source{d}"

    DEFAULT_N_CANDIDATES = 3
    DEFAULT_CONFIDENCE_THRESHOLD = 0

    corrector = TyposCorrector(threads_number=4)
    corrector.load(str(Path(__file__).parent / "small_corrector.asdf"))

    def __init__(self, confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
                 n_candidates: int = DEFAULT_N_CANDIDATES):
        """
        Initialize a new instance of IdTyposModel.

        :param confidence_threshold: Confidence threshold.
        :param n_candidates: Number of candidates.
        """
        super().__init__()
        self.confidence_threshold = confidence_threshold
        self.n_candidates = n_candidates
        self.parser = TokenParser()

    def check_identifiers(self, identifiers: List[str]
                          ) -> Dict[int, Dict[str, List[Tuple[str, float]]]]:
        """
        Check tokens from identifiers for typos.

        :param identifiers: List of identifiers to check.
        :return: Dictionary of corrections grouped by ids of corresponding identifier \
                 in 'identifiers' and typoed tokens which have correction suggestions.
        """
        splits = [" ".join(list(self.parser.split(identifier)))
                  for identifier in identifiers]

        test_df = pandas.DataFrame(columns=[NODE_ID_COLUMN, SPLIT_COLUMN])
        test_df[NODE_ID_COLUMN] = range(len(identifiers))
        test_df[SPLIT_COLUMN] = splits
        test_df = flatten_data(test_df, new_column_name=TYPO_COLUMN)

        suggestions = self.corrector.suggest(test_df, n_candidates=3, return_all=False)
        suggestions = self.filter_suggestions(test_df, suggestions)
        return self.group_by_node_id(test_df, suggestions)

    def filter_suggestions(self, test_df: pandas.DataFrame,
                           suggestions: Dict[int, List[Tuple[str, float]]]
                           ) -> Dict[int, List[Tuple[str, float]]]:
        """
        Filter suggestions based on the repo specifics and confidence threshold.

        :param test_df: DataFrame with info about tested tokens.
        :param suggestions: Dictionary of correction suggestions grouped by \
                            typoed token index in test_df.
        :return: Dictionary of filtered suggestions grouped by typoed token index in test_df.
        """
        filtered_suggestions = {}
        tokens = test_df.typo
        for index, candidates in suggestions.items():
            filtered_candidates = []
            for candidate in candidates:
                if candidate[0] == tokens[index] or candidate[1] < self.confidence_threshold:
                    break
                filtered_candidates.append(candidate)

            if len(filtered_candidates):
                filtered_suggestions[index] = filtered_candidates

        return filtered_suggestions

    @staticmethod
    def group_by_node_id(test_df: pandas.DataFrame,
                         suggestions: Dict[int, List[Tuple[str, float]]]
                         ) -> Dict[int, Dict[str, List[Tuple[str, float]]]]:
        """
        Group corrections by nodes to which the typoed tokens belong.

        :param test_df: DataFrame with info about tested identifiers.
        :param suggestions: Dictionary of correction suggestions grouped by \
                            typoed token index in test_df.
        :return: Dictionary of correction suggestions grouped by nodes \
                 to which typoed tokens belong and the corrected tokens.
        """
        grouped_suggestions = defaultdict(dict)
        for index, row in test_df.iterrows():
            if index in suggestions.keys():
                grouped_suggestions[row[NODE_ID_COLUMN]][row[TYPO_COLUMN]] = suggestions[index]

        return grouped_suggestions

    @property
    def confidence_threshold(self) -> float:
        """Return the confidence threshold."""
        return self._confidence_threshold

    @property
    def n_candidates(self) -> int:
        """Return the number of candidates."""
        return self._n_candidates

    @confidence_threshold.setter
    def confidence_threshold(self, confidence_threshold: float):
        self._confidence_threshold = confidence_threshold

    @n_candidates.setter
    def n_candidates(self, n_candidates):
        self._n_candidates = n_candidates

    def dump(self) -> str:
        """Model.__str__ callback."""
        return "Typos correcting model with vocabulary size %d" %\
               len(self.corrector.generator.tokens)

    def _generate_tree(self) -> dict:
        return {"n_candidates": self.n_candidates,
                "confidence_threshold": self.confidence_threshold}

    def _load_tree(self, tree: dict) -> None:
        self.n_candidates = tree["n_candidates"]
        self.confidence_threshold = tree["confidence_threshold"]
