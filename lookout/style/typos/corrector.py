"""Typo correction model."""

from itertools import chain
from typing import Dict, List, Tuple

from modelforge import Model
import pandas
from tqdm import tqdm

from lookout.style.typos.generation import (CandidatesGenerator, get_candidates_features,
                                            get_candidates_metadata)
from lookout.style.typos.ranking import CandidatesRanker
from lookout.style.typos.utils import CORRECT_TOKEN_COLUMN


class TyposCorrector(Model):
    """
    Model for correcting typos in tokens inside identifiers.
    """

    NAME = "typos_correction"
    VENDOR = "source{d}"

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
    def threads_number(self):
        """Return the number of threads used to train and to predict."""
        return self.ranker.boost_param["nthread"]

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
        :param frequencies_file: The path ot the frequencies.
        :param embeddings_file: The path to the embeddings.
        :param neighbors_number: Number of neighbors of context and typo embeddings \
                                 to consider as candidates.
        :param edit_candidates: Number of the most frequent tokens among tokens on \
                                equal edit distance from the typo to consider as candidates.
        :param max_distance: Maximum edit distance for symspell lookup.
        :param radius: Maximum edit distance from typo allowed for candidates.
        :return: Nothing
        """
        self.generator.construct(vocabulary_file, frequencies_file, embeddings_file,
                                 neighbors_number, edit_candidates, max_distance, radius)

    def train(self, typos: pandas.DataFrame, candidates: pandas.DataFrame = None,
              save_candidates_file: str = None) -> None:
        """
        Train corrector on the given dataset of typos inside identifiers.

        :param typos: DataFrame containing columns "typo" and "identifier",
                      column "token_split" is optional, but used when present.
        :param candidates: DataFrame with precalculated candidates.
        :param save_candidates_file: Path to file where to save the candidates.
        """
        if candidates is None:
            candidates = self.generator.generate_candidates(
                typos, self.threads_number, save_candidates_file)
        self.ranker.fit(typos[CORRECT_TOKEN_COLUMN], get_candidates_metadata(candidates),
                        get_candidates_features(candidates))

    def train_on_file(self, typos_file: str, candidates_file: str = None,
                      save_candidates_file: str = None) -> None:
        """
        Train corrector on the given dataset of typos inside identifiers.

        :param typos_file: CSV file with columns "typo" and "identifier",
                           column "token_split" is optional, but used when present.
        :param candidates_file: Pickle dump of pandas.DataFrame with precalculated \
                                candidates and features
        :param save_candidates_file: Path to file where to save the candidates.
        """
        typos = pandas.read_csv(typos_file, index_col=0)
        candidates = None
        if candidates_file is not None:
            candidates = pandas.read_pickle(candidates_file)
        self.train(typos, candidates, save_candidates_file)

    def suggest(self, typos: pandas.DataFrame, candidates: pandas.DataFrame = None,
                save_candidates_file: str = None, n_candidates: int = 3,
                return_all: bool = True) -> Dict[int, List[Tuple[str, float]]]:
        """
        Suggest corrections for given typos.

        :param typos: DataFrame containing column "typo", \
                      column "token_split" is optional, but used when present
        :param candidates: DataFrame with precalculated candidates
        :param n_candidates: Number of most probable candidates to return
        :param return_all: False to return suggestions only for corrected tokens
        :param save_candidates_file: Path to file to save candidates to
        :return: Dictionary {id : [[candidate, correctness_proba]]}, candidates are sorted \
                 by correctness probability in a descending order.
        """
        if candidates is None:
            candidates = self.generator.generate_candidates(typos, self.threads_number,
                                                            save_candidates_file)
        return self.ranker.rank(get_candidates_metadata(candidates),
                                get_candidates_features(candidates), n_candidates, return_all)

    def suggest_file(self, typos_file: str, candidates_file: str = None,
                     save_candidates_file: str = None, n_candidates: int = 3,
                     return_all: bool = True) -> Dict[int, List[Tuple[str, float]]]:
        """
        Suggest corrections for given typos.

        :param typos_file: csv file containing DataFrame with column "typo", \
                           column "token_split" is optional, but used when present
        :param candidates_file: pickle file containing DataFrame with precalculated \
                                candidates and features
        :param n_candidates: Number of most probable candidates to return
        :param return_all: False to return suggestions only for corrected tokens
        :param save_candidates_file: Path to file to save candidates to
        :return: Dictionary {id : [[candidate, correctness_proba]]}, candidates are sorted \
                 by correctness probability in a descending order.
        """
        typos = pandas.read_csv(typos_file, index_col=0)
        candidates = None
        if candidates_file is not None:
            candidates = pandas.read_pickle(candidates_file)
        return self.suggest(typos, candidates, save_candidates_file, n_candidates, return_all)

    def suggest_by_batches(self, typos: pandas.DataFrame, n_candidates: int = None,
                           return_all: bool = True, batch_size: int = 2048,
                           ) -> Dict[int, List[Tuple[str, float]]]:
        """
        Correct typos from dataset by batches. Does not support precalculated candidates.

        Suggest corrections for given typos
        :param typos: DataFrame containing column "typo", \
               column "token_split" is optional, but used when present
        :param n_candidates: Number of most probable candidates to return
        :param return_all: False to return suggestions only for corrected tokens
        :param batch_size: Batch size
        :return: Dictionary {id : [[candidate, correctness_proba]]}, candidates are sorted \
                 by correctness probability in a descending order.
        """
        all_suggestions = []
        for i in tqdm(range(0, len(typos), batch_size)):
            suggestions = self.suggest(typos.loc[typos.index[i]:
                                                 typos.index[min(len(typos) - 1,
                                                                 i + batch_size - 1)], :],
                                       n_candidates=n_candidates, return_all=return_all)
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
