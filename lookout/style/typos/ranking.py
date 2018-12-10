"""Ranking typo correction candidates using a GBT."""
import multiprocessing
from typing import Dict, List, Tuple

from modelforge import Model
import numpy
import pandas
import xgboost as xgb

from lookout.style.typos.utils import CANDIDATE_COLUMN, ID_COLUMN, rank_candidates


class CandidatesRanker(Model):
    """
    Rank typos correcting candidates based on given features. \
    XGBoost classifier is used.
    """

    NAME = "candidates_ranks"
    VENDOR = "source{d}"
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

    def __init__(self, **kwargs):
        """Initialize a new instance of CandidatesRanker class."""
        super().__init__(**kwargs)
        self.train_rounds = self.DEFAULT_TRAIN_ROUNDS
        self.early_stopping = self.DEFAULT_EARLY_STOPPING
        self.boost_param = self.DEFAULT_BOOST_PARAM
        self.bst = None  # type: xgb.Booster

    def construct(self, train_rounds: int = DEFAULT_TRAIN_ROUNDS,
                  early_stopping: int = DEFAULT_EARLY_STOPPING,
                  boost_param: dict = None) -> None:
        """
        Assign the training parameters. See XGBoost docs for the details.

        :param train_rounds: Number of training rounds.
        :param early_stopping: Early stopping parameter.
        :param boost_param: Boosting parameters. The actual default is DEFAULT_BOOST_PARAM.
        :return: Nothing.
        """
        self.train_rounds = train_rounds
        self.early_stopping = early_stopping
        self.boost_param = boost_param or self.DEFAULT_BOOST_PARAM
        self.boost_param["nthread"] = self.boost_param["nthread"] or multiprocessing.cpu_count()

    def fit(self, identifiers: pandas.Series, candidates: pandas.DataFrame,
            features: numpy.ndarray, val_part: float = 0.1) -> None:
        """
        Train booster on the given data.

        :param identifiers: Series containing column right corrections and indexed in \
                            correspondence with typos from which candidates were generated.
        :param candidates: DataFrame containing information about candidates for correction. \
                           Columns are [ID_COLUMN, TYPO_COLUMN, CANDIDATE_COLUMN].
        :param features: Matrix of features for candidates.
        :param val_part: Part of data used for validation.
        """
        labels = self._create_labels(identifiers, candidates)
        edge = int(features.shape[0] * (1 - val_part))
        data_train = xgb.DMatrix(features[:edge, :], label=labels[:edge])
        data_val = xgb.DMatrix(features[edge:, :], label=labels[edge:])
        self.boost_param["scale_pos_weight"] = float(
            1.0 * (edge - numpy.sum(labels[:edge])) / numpy.sum(labels[:edge]))
        evallist = [(data_train, "train"), (data_val, "validation")]
        self.bst = xgb.train(self.boost_param, data_train, self.train_rounds, evallist,
                             early_stopping_rounds=self.early_stopping, verbose_eval=False)

    def rank(self, candidates: pandas.DataFrame, features: numpy.ndarray, n_candidates: int = 3,
             return_all: bool = True) -> Dict[int, List[Tuple[str, float]]]:
        """
        Assign the correctness probability value for each of the candidates.

        :param candidates: DataFrame containing information about candidates for correction. \
                           Columns are [ID_COLUMN, TYPO_COLUMN, CANDIDATE_COLUMN].
        :param features: Matrix of features for candidates.
        :param n_candidates: Number of most probably correct candidates to return for each typo.
        :param return_all: False to return corrections only for typos corrected in the \
                           first candidate.
        :return: Dictionary {id : [[candidate, correctness_proba]]}, candidates are sorted \
                 by correctness probability in a descending order.
        """
        dtest = xgb.DMatrix(features)
        test_probs = self.bst.predict(dtest, ntree_limit=self.bst.best_ntree_limit)
        return rank_candidates(candidates, test_probs, n_candidates, return_all)

    def dump(self):
        """Describe the model for introspection."""
        return "Attributes: %s\nParameters: %s" % (
            sorted(self.bst.attributes().items()) if self.bst is not None else "<not trained>",
            sorted(self.boost_param.items()))

    def __eq__(self, other: "CandidatesRanker") -> bool:
        for k in ("train_rounds", "early_stopping", "boost_param"):
            if getattr(self, k) != getattr(other, k):
                return False
        if (self.bst is None) != (other.bst is None):
            return False
        if self.bst is None:
            return True
        if self.bst.best_ntree_limit != other.bst.best_ntree_limit:
            return False
        return self.bst.save_raw() == other.bst.save_raw()

    @staticmethod
    def _create_labels(identifiers: pandas.Series, candidates: pandas.DataFrame) -> numpy.ndarray:
        labels = []
        for _, row in candidates.iterrows():
            labels.append(int(row[CANDIDATE_COLUMN] == identifiers[row[ID_COLUMN]]))
        return numpy.array(labels)

    def _generate_tree(self) -> dict:
        tree = {k: getattr(self, k) for k in ("train_rounds", "early_stopping", "boost_param")}
        assert self.bst is not None
        tree["bst"] = numpy.array(self.bst.save_raw())
        tree["best_ntree_limit"] = self.bst.best_ntree_limit
        return tree

    def _load_tree(self, tree: dict) -> None:
        self.__dict__.update(tree)
        self.bst = xgb.Booster(model_file=tree["bst"].data)
        self.bst.best_ntree_limit = self.best_ntree_limit
        del self.best_ntree_limit
