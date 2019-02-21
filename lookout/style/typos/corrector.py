"""Typo correction model."""
from itertools import chain
from typing import Dict, List, Tuple

from modelforge import Model
import pandas
from sourced.ml.models.license import DEFAULT_LICENSE
from tqdm import tqdm

from lookout.style.typos.generation import (CandidatesGenerator, get_candidates_features,
                                            get_candidates_metadata)
from lookout.style.typos.ranking import CandidatesRanker
from lookout.style.typos.utils import Columns


class TyposCorrector(Model):
    """
    Model for correcting typos in tokens inside identifiers.
    """

    NAME = "typos_correction"
    VENDOR = "source{d}"
    DESCRIPTION = "Model that suggests fixes to correct typos."
    LICENSE = DEFAULT_LICENSE
    DEFAULT_RADIUS = 3
    DEFAULT_MAX_DISTANCE = 2
    DEFAULT_NEIGHBORS_NUMBER = 0
    DEFAULT_EDIT_CANDIDATES = 20
    DEFAULT_TRAIN_ROUNDS = 4000
    DEFAULT_EARLY_STOPPING = 200
    DEFAULT_BOOST_PARAM = {"max_depth": 6,
                           "eta": 0.03,
                           "min_child_weight": 2,
                           "silent": 1,
                           "objective": "binary:logistic",
                           "subsample": 0.5,
                           "colsample_bytree": 0.5,
                           "alpha": 1,
                           "eval_metric": ["error"],
                           "nthread": 0}

    def __init__(self, **kwargs):
        """
        Initialize a new instance of TyposCorrector class.

        :param kwargs: extra keyword arguments which are consumed by Model.
        """
        super().__init__(**kwargs)
        self.generator = CandidatesGenerator()
        self.ranker = CandidatesRanker()

    @property
    def threads_number(self) -> int:
        """Return the number of threads for multiprocessing used to train and to predict."""
        return self.ranker.boost_param["nthread"]

    @threads_number.setter
    def threads_number(self, threads_number: int):
        """Set the number of threads for multiprocessing used to train and to predict."""
        self.ranker.boost_param["nthread"] = threads_number

    def initialize_ranker(self, train_rounds: int = DEFAULT_TRAIN_ROUNDS,
                          early_stopping: int = DEFAULT_EARLY_STOPPING,
                          boost_params: dict = None) -> None:
        """
        Apply the ranking parameters - see XGBoost docs for details.

        :param train_rounds: Number of training rounds.
        :param early_stopping: Early stopping parameter.
        :param boost_params: Boosting parameters. The defaults are DEFAULT_BOOST_PARAM.
        :return: Nothing
        """
        boost_params = boost_params or self.DEFAULT_BOOST_PARAM
        self.ranker.construct(train_rounds, early_stopping, boost_params)

    def initialize_generator(self, vocabulary_file: str, frequencies_file: str,
                             embeddings_file: str = None,
                             neighbors_number: int = DEFAULT_NEIGHBORS_NUMBER,
                             edit_candidates: int = DEFAULT_EDIT_CANDIDATES,
                             max_distance: int = DEFAULT_MAX_DISTANCE,
                             radius: int = DEFAULT_RADIUS) -> None:
        """
        Construct a new CandidatesGenerator.

        :param vocabulary_file: The path to the vocabulary.
        :param frequencies_file: The path to the frequencies.
        :param embeddings_file: The path to the embeddings.
        :param neighbors_number: Number of neighbors of context and typo embeddings \
                                 to consider as candidates.
        :param edit_candidates: Number of the most frequent tokens among tokens on \
                                equal edit distance from the typo to consider as candidates.
        :param max_distance: Maximum edit distance for symspell lookup.
        :param radius: Maximum edit distance from typo allowed for candidates.
        """
        self.generator.construct(vocabulary_file, frequencies_file, embeddings_file,
                                 neighbors_number, edit_candidates, max_distance, radius)

    def train(self, data: pandas.DataFrame, candidates:  str = None,
              save_candidates_file: str = None) -> None:
        """
        Train corrector on tokens from the given dataset.

        :param data: DataFrame which contains columns Columns.Token, Columns.CorrectToken, \
                     and Columns.Split.
        :param candidates: A .csv.xz dump of a dataframe with precalculated candidates.
        :param save_candidates_file: Path to file where to save the candidates (.csv.xz).
        """
        if candidates is None:
            candidates = self.generator.generate_candidates(
                data, self.threads_number, save_candidates_file)
        else:
            candidates = pandas.read_csv(candidates, index_col=0)
        self.ranker.fit(data[Columns.CorrectToken], get_candidates_metadata(candidates),
                        get_candidates_features(candidates))

    def train_on_file(self, data_file: str, candidates:  str = None,
                      save_candidates_file: str = None) -> None:
        """
        Train corrector on tokens from the given file.

        :param data_file: A .csv dump of a dataframe which contains columns Columns.Token, \
                          Columns.CorrectToken and Columns.Split.
        :param candidates: A .csv.xz dump of a dataframe with precalculated candidates.
        :param save_candidates_file: Path to file where to save the candidates (.csv.xz).
        """
        self.train(pandas.read_csv(data_file, index_col=0), candidates, save_candidates_file)

    def suggest(self, data: pandas.DataFrame, candidates: str = None,
                save_candidates_file: str = None, n_candidates: int = 3,
                return_all: bool = True) -> Dict[int, List[Tuple[str, float]]]:
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
                data, self.threads_number, save_candidates_file)
        else:
            candidates = pandas.read_csv(candidates, index_col=0)
        return self.ranker.rank(get_candidates_metadata(candidates),
                                get_candidates_features(candidates), n_candidates, return_all)

    def suggest_on_file(self, data_file: str, candidates:  str = None,
                        save_candidates_file: str = None, n_candidates: int = 3,
                        return_all: bool = True) -> Dict[int, List[Tuple[str, float]]]:
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
        return self.suggest(pandas.read_csv(data_file, index_col=0), candidates,
                            save_candidates_file, n_candidates, return_all)

    def suggest_by_batches(self, data: pandas.DataFrame, n_candidates: int = None,
                           return_all: bool = True, batch_size: int = 2048,
                           ) -> Dict[int, List[Tuple[str, float]]]:
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
