import pickle
from typing import Dict, List, Tuple

import numpy
import pandas
import xgboost as xgb

from lookout.style.typos.utils import rank_candidates, CANDIDATE_COLUMN, ID_COLUMN


class CandidatesRanker:
    """
    Rank typos correcting candidates based on given features.
    XGBoost classifier used.
    """
    DEFAULT_TRAIN_ROUNDS = 4000
    DEFAULT_EARLY_STOPPING = 200
    DEFAULT_BOOST_PARAM = {"max_depth": 6,
                           "eta": 0.03,
                           "min_child_weight": 2,
                           "silent": 1,
                           "objective": "binary:logistic",
                           "nthread": 16,
                           "subsample": 0.5,
                           "colsample_bytree": 0.5,
                           "alpha": 1,
                           "eval_metric": ["error"]}

    def __init__(self):
        self.train_rounds = self.DEFAULT_TRAIN_ROUNDS
        self.early_stopping = self.DEFAULT_EARLY_STOPPING
        self.boost_param = self.DEFAULT_BOOST_PARAM
        self.bst = None

    def set_boost_params(self, train_rounds: int = DEFAULT_TRAIN_ROUNDS,
                         early_stopping: int = DEFAULT_EARLY_STOPPING,
                         boost_param: dict = DEFAULT_BOOST_PARAM) -> None:
        self.train_rounds = train_rounds
        self.early_stopping = early_stopping
        self.boost_param = boost_param

    def fit(self, identifiers: pandas.Series, candidates: pandas.DataFrame,
            features: numpy.ndarray, val_part: float = 0.1) -> None:
        """
        Train booster on the given data.
        :param identifiers: Series containing column right corrections and indexed in
                            correspondence with typos from which candidates were generated.
        :param candidates: DataFrame containing information about candidates for correction.
                           Columns are [ID_COLUMN, TYPO_COLUMN, CANDIDATE_COLUMN].
        :param features: Matrix of features for candidates.
        :param val_part: Part of data used for validation.
        """
        labels = self._create_labels(identifiers, candidates)

        edge = int(features.shape[0] * (1 - val_part))

        dtrain = xgb.DMatrix(features[:edge, :], label=labels[:edge])
        dval = xgb.DMatrix(features[edge:, :], label=labels[edge:])

        self.boost_param["scale_pos_weight"] = (1.0 * (edge - numpy.sum(labels[:edge])) /
                                                numpy.sum(labels[:edge]))

        evallist = [(dtrain, "train"), (dval, "validation")]
        self.bst = xgb.train(self.boost_param, dtrain, self.train_rounds, evallist,
                             early_stopping_rounds=self.early_stopping)

    def rank(self, candidates: pandas.DataFrame, features: numpy.ndarray, n_candidates: int = 3,
             return_all: bool = True) -> Dict[int, List[Tuple[str, float]]]:
        """
        Rank candidates.
        :param candidates: DataFrame containing information about candidates for correction.
                           Columns are [ID_COLUMN, TYPO_COLUMN, CANDIDATE_COLUMN].
        :param features: Matrix of features for candidates.
        :param n_candidates: Number of most probably correct candidates to return for each typo.
        :param return_all: False to return corrections only for typos corrected in the
                           first candidate.
        :return: Dictionary {id : [[candidate, correctness_proba]]}, candidates are sorted
                 by correctness probability in a descending order.
        """
        dtest = xgb.DMatrix(features)
        test_proba = self.bst.predict(dtest, ntree_limit=self.bst.best_ntree_limit)

        return rank_candidates(candidates, test_proba, n_candidates, return_all)

    @staticmethod
    def _create_labels(identifiers: pandas.Series, candidates: pandas.DataFrame) -> numpy.ndarray:
        labels = []
        for ind, row in candidates.iterrows():
            labels.append(int(row[CANDIDATE_COLUMN] == identifiers[row[ID_COLUMN]]))
        return numpy.array(labels)

    def _generate_tree(self) -> dict:
        tree = self.__dict__.copy()
        tree["bst"] = pickle.dumps(self.bst)
        if self.bst is not None:
            tree["bst_ntree_limit"] = self.bst.best_ntree_limit
        return tree

    def _load_tree(self, tree: dict) -> None:
        self.__dict__.update()
        self.bst = pickle.loads(tree["bst"])
        if self.bst is not None:
            self.bst.best_ntree_limit = tree["bst_ntree_limit"]
