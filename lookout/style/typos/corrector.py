from itertools import chain
from typing import Dict, List, Tuple

from modelforge import Model
import pandas
from tqdm import tqdm

from lookout.style.typos.generation import (CandidatesGenerator, get_candidates_features,
                                            get_candidates_tokens)
from lookout.style.typos.ranking import CandidatesRanker
from lookout.style.typos.utils import CORRECT_TOKEN_COLUMN


class TyposCorrector(Model):
    """
    Model for correcting typos in tokens inside identifiers
    """
    NAME = "typos_correction"
    VENDOR = "source{d}"

    DEFAULT_THREADS_NUMBER = 16

    DEFAULT_RADIUS = 4
    DEFAULT_MAX_DISTANCE = 3
    DEFAULT_NEIGHBORS_NUMBER = 20
    DEFAULT_TAKEN_FOR_DISTANCE = 10

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
                           "eval_metric": ["auc", "error"]}

    def __init__(self, threads_number: int = DEFAULT_THREADS_NUMBER, nn_file: str = None):
        super().__init__()
        self.generator = CandidatesGenerator()
        self.ranker = CandidatesRanker()
        self.nn_file = nn_file
        self.threads_number = threads_number
        self.set_ranker_params()

    def set_ranker_params(self, train_rounds: int = DEFAULT_TRAIN_ROUNDS,
                          early_stopping: int = DEFAULT_EARLY_STOPPING,
                          boost_param: dict = DEFAULT_BOOST_PARAM) -> None:
        boost_param["nthread"] = self.threads_number
        self.ranker.set_boost_params(train_rounds, early_stopping, boost_param)

    def create_model(self, vocabulary_file: str, frequencies_file: str,
                     embeddings_file: str = None,
                     neighbors_number: int = DEFAULT_NEIGHBORS_NUMBER,
                     taken_for_distance: int = DEFAULT_TAKEN_FOR_DISTANCE,
                     max_distance: int = DEFAULT_MAX_DISTANCE,
                     radius: int = DEFAULT_RADIUS) -> None:
        self.generator.construct(vocabulary_file, frequencies_file, embeddings_file,
                                 neighbors_number, taken_for_distance, max_distance, radius)

    def train(self, typos: pandas.DataFrame, candidates: pandas.DataFrame = None,
              save_candidates_file: str = None) -> None:
        """
        Train corrector on given dataset of typos inside identifiers
        :param typos: DataFrame containing columns "typo" and "identifier",
                      column "token_split" is optional, but used when present
        :param candidates: DataFrame with precalculated candidates
        :param save_candidates_file: Path to file to save candidates to
        """
        if candidates is None:
            candidates = self.generator.generate_candidates(typos, self.threads_number,
                                                            self.nn_file, save_candidates_file)
        self.ranker.fit(typos[CORRECT_TOKEN_COLUMN], get_candidates_tokens(candidates),
                        get_candidates_features(candidates))

    def train_on_file(self, typos_file: str, candidates_file: str = None,
                      save_candidates_file: str = None) -> None:
        """
        Train corrector on given dataset of typos inside identifiers
        :param typos_file: csv file containing pandas.DataFrame with
                           columns "typo" and "identifier", column "token_split" is optional,
                           but used when present
        :param candidates_file: Pickle dump of pandas.DataFrame with precalculated
                                candidates and features
        :param save_candidates_file: Path to file to save candidates to
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
        Suggest corrections for given typos
        :param typos: DataFrame containing column "typo",
                      column "token_split" is optional, but used when present
        :param candidates: DataFrame with precalculated candidates
        :param n_candidates: Number of most probable candidates to return
        :param return_all: False to return suggestions only for corrected tokens
        :param save_candidates_file: Path to file to save candidates to
        :return: Dictionary {id : [[candidate, correctness_proba]]}, candidates are sorted
                 by correctness probability in a descending order.
        """
        if candidates is None:
            candidates = self.generator.generate_candidates(typos, self.threads_number,
                                                            self.nn_file, save_candidates_file)
        return self.ranker.rank(get_candidates_tokens(candidates),
                                get_candidates_features(candidates), n_candidates, return_all)

    def suggest_file(self, typos_file: str, candidates_file: str = None,
                     save_candidates_file: str = None, n_candidates: int = 3,
                     return_all: bool = True) -> Dict[int, List[Tuple[str, float]]]:
        """
        Suggest corrections for given typos
        :param typos_file: csv file containing DataFrame with column "typo",
                           column "token_split" is optional, but used when present
        :param candidates_file: pickle file containing DataFrame with precalculated
                                candidates and features
        :param n_candidates: Number of most probable candidates to return
        :param return_all: False to return suggestions only for corrected tokens
        :param save_candidates_file: Path to file to save candidates to
        :return: Dictionary {id : [[candidate, correctness_proba]]}, candidates are sorted
                 by correctness probability in a descending order.
        """
        typos = pandas.read_csv(typos_file, index_col=0)
        candidates = None
        if candidates_file is not None:
            candidates = pandas.read_pickle(candidates_file)
        return self.suggest(typos, candidates, save_candidates_file, n_candidates, return_all)

    def suggest_by_batches(self, typos: pandas.DataFrame, n_candidates: int = None,
                           return_all: bool = True, batch_size: int = 2048
                           ) -> Dict[int, List[Tuple[str, float]]]:
        """
        Correct typos from dataset by batches. Does not support precalculated candidates.
        Suggest corrections for given typos
        :param typos: DataFrame containing column "typo",
               column "token_split" is optional, but used when present
        :param n_candidates: Number of most probable candidates to return
        :param return_all: False to return suggestions only for corrected tokens
        :param batch_size: Batch size
        :return: Dictionary {id : [[candidate, correctness_proba]]}, candidates are sorted
                 by correctness probability in a descending order.
        """
        all_suggestions = []
        for i in tqdm(range(0, len(typos), batch_size)):
            suggestions = self.suggest(typos.loc[typos.index[i]:
                                                 typos.index[min(len(typos) - 1,
                                                                 i + batch_size - 1)], :],
                                       n_candidates=n_candidates, return_all=return_all)
            all_suggestions.append(suggestions.items())

        return dict(list(chain.from_iterable(all_suggestions)))

    def dump(self) -> str:
        return ("Candidates and features generator parameters:\n%s"
                "XGBoost classifier is used for ranking candidates" %
                str(self.finder))

    def _generate_tree(self) -> dict:
        return {"generator": self.generator._generate_tree(),
                "ranker": self.ranker._generate_tree()}

    def _load_tree(self, tree: dict) -> None:
        self.generator._load_tree(tree["generator"])
        self.ranker._load_tree(tree["ranker"])
