"""Typo correction model."""
from itertools import chain
import logging
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from modelforge import Model
import pandas
from sourced.ml.core.models.license import DEFAULT_LICENSE
from tqdm import tqdm

from lookout.style.typos.generation import (CandidatesGenerator, get_candidates_features,
                                            get_candidates_metadata)
from lookout.style.typos.metrics import generate_report, get_scores
from lookout.style.typos.ranking import CandidatesRanker
from lookout.style.typos.utils import Candidate, Columns


class TyposCorrector(Model):
    """
    Model for correcting typos in tokens inside identifiers.
    """

    _log = logging.getLogger("TyposCorrector")

    NAME = "typos_correction"
    VENDOR = "source{d}"
    DESCRIPTION = "Model that suggests fixes to correct typos."
    LICENSE = DEFAULT_LICENSE

    def __init__(self, ranking_config: Optional[Mapping[str, Any]] = None, **kwargs):
        """
        Initialize a new instance of TyposCorrector class.

        :param ranking_config: Ranking configuration, options:
                                train_rounds: Number of training rounds (int).
                                early_stopping: Early stopping parameter (int).
                                boost_param: Boosting parameters (dict).
        :param kwargs: Extra keyword arguments which are consumed by Model.
        """
        super().__init__(**kwargs)
        self.generator = CandidatesGenerator()
        self.ranker = CandidatesRanker(ranking_config)

    @property
    def processes_number(self) -> int:
        """Return the number of processes for multiprocessing used to train and to predict."""
        return self.ranker.config["boost_param"]["nthread"]

    @processes_number.setter
    def processes_number(self, processes_number: int):
        """Set the number of processes for multiprocessing used to train and to predict."""
        self.ranker.config["boost_param"]["nthread"] = processes_number

    def initialize_generator(self, vocabulary_file: str, frequencies_file: str,
                             embeddings_file: str, config: Optional[Mapping[str, Any]] = None,
                             ) -> None:
        """
        Construct a new CandidatesGenerator.

        :param vocabulary_file: The path to the vocabulary.
        :param frequencies_file: The path to the frequencies.
        :param embeddings_file: The path to the embeddings.
        :param config: Candidates generation configuration, options:
                       neighbors_number: Number of neighbors of context and typo embeddings \
                                         to consider as candidates (int).
                       edit_dist_number: Number of the most frequent tokens among tokens at \
                                         equal edit distance from the typo to consider as \
                                         candidates (int).
                       max_distance: Maximum edit distance for symspell lookup for candidates \
                                    (int).
                       radius: Maximum edit distance from typo allowed for candidates (int).
                       max_corrected_length: Maximum length of prefix in which symspell lookup \
                                             for typos is conducted (int).
                       start_pool_size: Length of data, starting from which multiprocessing is \
                                        desired (int).
                       chunksize: Max size of a chunk for one process during multiprocessing (int).
        """
        self.generator.construct(vocabulary_file, frequencies_file, embeddings_file, config)
        self._log.debug("%s is initialized", repr(self.generator))

    def set_ranking_config(self, config: Mapping[str, Any]) -> None:
        """
        Update the ranking config - see XGBoost docs for details.

        :param config: Ranking configuration, options:
                       train_rounds: Number of training rounds (int).
                       early_stopping: Early stopping parameter (int).
                       boost_param: Boosting parameters (dict).
        """
        self.ranker.set_config(config)
        self._log.debug("%s is initialized", repr(self.ranker))

    def set_generation_config(self, config: Mapping[str, Any]) -> None:
        """
        Update the candidates generation config.

        :param config: Candidates generation configuration, options:
                       neighbors_number: Number of neighbors of context and typo embeddings \
                                         to consider as candidates (int).
                       edit_dist_number: Number of the most frequent tokens among tokens at \
                                         equal edit distance from the typo to consider as \
                                         candidates (int).
                       max_distance: Maximum edit distance for symspell lookup for candidates \
                                    (int).
                       radius: Maximum edit distance from typo allowed for candidates (int).
                       max_corrected_length: Maximum length of prefix in which symspell lookup \
                                             for typos is conducted (int).
                       start_pool_size: Length of data, starting from which multiprocessing is \
                                        desired (int).
                       chunksize: Max size of a chunk for one process during multiprocessing (int).
        """
        self.generator.set_config(config)
        self._log.debug("%s is initialized", repr(self.ranker))

    def expand_vocabulary(self, additional_tokens: Iterable[str]) -> None:
        """
        Add given tokens to the model's vocabulary.

        :param additional_tokens: Tokens to add to the vocabulary.
        """
        self.generator.expand_vocabulary(additional_tokens)

    def train(self, data: pandas.DataFrame, candidates: Optional[str] = None,
              save_candidates_file: Optional[str] = None) -> None:
        """
        Train corrector on tokens from the given dataset.

        :param data: DataFrame which contains columns Columns.Token, Columns.CorrectToken, \
                     and Columns.Split.
        :param candidates: A .csv.xz dump of a dataframe with precalculated candidates.
        :param save_candidates_file: Path to file where to save the candidates (.csv.xz).
        """
        self._log.info("train input shape: %s", data.shape)
        if candidates is None:
            self._log.info("candidates were not provided and will be generated")
            candidates = self.generator.generate_candidates(
                data, self.processes_number, save_candidates_file)
        else:
            candidates = pandas.read_csv(candidates, index_col=0, keep_default_na=False)
            self._log.info("loaded candidates from %s", candidates)
        self.ranker.fit(data[Columns.CorrectToken], get_candidates_metadata(candidates),
                        get_candidates_features(candidates))

    def train_on_file(self, data_file: str, candidates:  Optional[str] = None,
                      save_candidates_file: Optional[str] = None) -> None:
        """
        Train corrector on tokens from the given file.

        :param data_file: A .csv dump of a dataframe which contains columns Columns.Token, \
                          Columns.CorrectToken and Columns.Split.
        :param candidates: A .csv.xz dump of a dataframe with precalculated candidates.
        :param save_candidates_file: Path to file where to save the candidates (.csv.xz).
        """
        self.train(pandas.read_csv(data_file, index_col=0, keep_default_na=False), candidates,
                   save_candidates_file)

    def suggest(self, data: pandas.DataFrame, candidates:  Optional[str] = None,
                save_candidates_file: Optional[str] = None, n_candidates: int = 3,
                return_all: bool = True) -> Dict[int, List[Candidate]]:
        """
        Suggest corrections for the tokens from the given dataset.

        :param data: DataFrame which contains columns Columns.Token and Columns.Split.
        :param candidates: A .csv.xz dump of a dataframe with precalculated candidates.
        :param save_candidates_file: Path to file to save candidates to (.csv.xz).
        :param n_candidates: Number of most probable candidates to return.
        :param return_all: False to return suggestions only for corrected tokens.
        :return: Dictionary `{id : [(candidate, correctness_proba), ...]}`, candidates are sorted \
                 by correctness probability in a descending order.
        """
        if candidates is None:
            candidates = self.generator.generate_candidates(
                data, self.processes_number, save_candidates_file)
        else:
            candidates = pandas.read_csv(candidates, index_col=0, keep_default_na=False)
        return self.ranker.rank(get_candidates_metadata(candidates),
                                get_candidates_features(candidates), n_candidates, return_all)

    def suggest_on_file(self, data_file: str, candidates:  Optional[str] = None,
                        save_candidates_file: Optional[str] = None, n_candidates: int = 3,
                        return_all: bool = True) -> Dict[int, List[Candidate]]:
        """
        Suggest corrections for the tokens from the given file.

        :param data_file: A .csv dump of a DataFrame which contains columns Columns.Token \
                          and Columns.Split.
        :param candidates: A .csv.xz dump of a dataframe with precalculated candidates.
        :param save_candidates_file: Path to file to save candidates to (.csv.xz).
        :param n_candidates: Number of most probable candidates to return.
        :param return_all: False to return suggestions only for corrected tokens.
        :return: Dictionary `{id : [(candidate, correctness_proba), ...]}`, candidates are sorted \
                 by correctness probability in a descending order.
        """
        return self.suggest(pandas.read_csv(data_file, index_col=0, keep_default_na=False),
                            candidates, save_candidates_file, n_candidates, return_all)

    def suggest_by_batches(self, data: pandas.DataFrame, n_candidates: int = 3,
                           return_all: bool = True, batch_size: int = 2048,
                           ) -> Dict[int, List[Candidate]]:
        """
        Suggest corrections for the tokens from the given dataset by batches. \
        Does not support precalculated candidates.

        :param data: DataFrame which contains columns Columns.Token and Columns.Split.
        :param n_candidates: Number of most probable candidates to return.
        :param return_all: False to return suggestions only for corrected tokens.
        :param batch_size: Batch size.
        :return: Dictionary `{id : [(candidate, correctness_proba), ...]}`, candidates are sorted \
                 by correctness probability in a descending order.
        """
        all_suggestions = []
        for i in tqdm(range(0, len(data), batch_size)):
            suggestions = self.suggest(data.iloc[i:i + batch_size, :], n_candidates=n_candidates,
                                       return_all=return_all)
            all_suggestions.append(suggestions.items())
        return dict(chain.from_iterable(all_suggestions))

    def evaluate(self, test_data: pandas.DataFrame) -> Tuple[Dict[int, List[Candidate]], str]:
        """
        Evaluate the corrector on the given test dataset.

        Save the result metrics to the model metadata and print it to the standard output.
        :param test_data: DataFrame which contains column Columns.Token, \
                          column Columns.Split is optional, but used when present.
        :return: Suggestions for correction of tokens inside the `test_data` and the quality
                 report.
        """
        self._log.info("evaluate on test data with shape %s", test_data.shape)
        suggestions = self.suggest(test_data)
        report = generate_report(test_data, suggestions)
        self.metrics = get_scores(test_data, suggestions)
        self._log.info("evaluation report:\n%s", report)
        return suggestions, report

    def __eq__(self, other: "TyposCorrector") -> bool:
        return self.generator == other.generator and self.ranker == other.ranker

    def dump(self) -> str:
        """Model.__str__ to format the object."""
        return ("# Generator\n"
                "%s\n\n"
                "# Ranker\n"
                "%s" %
                (self.generator.dump(), self.ranker.dump()))

    def _generate_tree(self) -> dict:
        return {"generator": self.generator._generate_tree(),
                "ranker": self.ranker._generate_tree()}

    def _load_tree(self, tree: dict) -> None:
        self.generator._load_tree(tree["generator"])
        self.ranker._load_tree(tree["ranker"])
