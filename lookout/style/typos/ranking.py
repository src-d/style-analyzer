"""Ranking typo correction candidates using a GBT."""
import logging
from typing import Any, Dict, List, Mapping, Optional

from modelforge import Model
import numpy
import pandas
from sourced.ml.core.models.license import DEFAULT_LICENSE
import xgboost as xgb

from lookout.style.common import merge_dicts
from lookout.style.typos.config import DEFAULT_CORRECTOR_CONFIG
from lookout.style.typos.utils import Candidate, Columns, rank_candidates


class CandidatesRanker(Model):
    """
    Rank typos correcting candidates based on given features. \
    XGBoost classifier is used.
    """

    _log = logging.getLogger("CandidatesRanker")

    NAME = "candidates_ranks"
    VENDOR = "source{d}"
    DESCRIPTION = "Model that ranks candidates according to their probability to fix the typo."
    LICENSE = DEFAULT_LICENSE

    def __init__(self, config: Optional[Mapping[str, Any]] = None, **kwargs):
        """
        Initialize a new instance of CandidatesRanker class.

        :param config: Ranking configuration, options:
                       train_rounds: Number of training rounds (int).
                       early_stopping: Early stopping parameter (int).
                       boost_param: Boosting parameters (dict).
        :param kwargs: Extra keyword arguments which are consumed by Model.
        """
        super().__init__(**kwargs)
        self.config = DEFAULT_CORRECTOR_CONFIG["ranking"]
        self.set_config(config)
        self.bst = None  # type: xgb.Booster

    def set_config(self, config: Optional[Mapping[str, Any]] = None) -> None:
        """
        Update ranking configuration.

        :param config: Ranking configuration, options:
                       train_rounds: Number of training rounds (int).
                       early_stopping: Early stopping parameter (int).
                       boost_param: Boosting parameters (dict).
        """
        if config is None:
            config = {}
        self.config = merge_dicts(self.config, config)

    def fit(self, identifiers: pandas.Series, candidates: pandas.DataFrame,
            features: numpy.ndarray, val_part: float = 0.1) -> None:
        """
        Train booster on the given data.

        :param identifiers: Series containing column right corrections and indexed in \
                            correspondence with typos from which candidates were generated.
        :param candidates: DataFrame containing information about candidates for correction. \
                           Columns are [Columns.Id, Columns.Token, Columns.Candidate].
        :param features: Matrix of features for candidates.
        :param val_part: Part of data used for validation.
        """
        self._log.info("fitting has started")
        self._log.info("identifiers shape %s", identifiers.shape)
        self._log.info("candidates shape %s", candidates.shape)
        self._log.info("features shape %s", features.shape)
        labels = self._create_labels(identifiers, candidates)
        all_tokens = numpy.array(list(set(candidates[Columns.Token])))
        indices = numpy.zeros(len(all_tokens), dtype=bool)
        indices[numpy.random.choice(len(all_tokens),
                                    int((1 - val_part) * len(all_tokens)),
                                    replace=False)] = True
        train_token = {all_tokens[i]: indices[i] for i in range(len(all_tokens))}
        in_train = numpy.array(
            [train_token[row[Columns.Token]] for _, row in candidates.iterrows()], dtype=bool)
        data_train = xgb.DMatrix(features[in_train], label=labels[in_train])
        data_val = xgb.DMatrix(features[~in_train], label=labels[~in_train])
        self.config["boost_param"]["scale_pos_weight"] = float(
            1.0 * (numpy.sum(in_train) - numpy.sum(labels[in_train])) / numpy.sum(
                labels[in_train]))
        evallist = [(data_train, "train"), (data_val, "validation")]
        self.bst = xgb.train(self.config["boost_param"], data_train, self.config["train_rounds"],
                             evallist, early_stopping_rounds=self.config["early_stopping"],
                             verbose_eval=self.config["verbose_eval"])
        self._log.debug("successfully fitted")

    def rank(self, candidates: pandas.DataFrame, features: numpy.ndarray, n_candidates: int = 3,
             return_all: bool = True) -> Dict[int, List[Candidate]]:
        """
        Assign the correctness probability value for each of the candidates.

        :param candidates: DataFrame containing information about candidates for correction.
        :param features: Matrix of features for candidates.
        :param n_candidates: Number of most probably correct candidates to return for each typo.
        :param return_all: False to return corrections only for typos corrected in the \
                           first candidate.
        :return: Dictionary `{id : [(candidate, correctness_proba), ...]}`, candidates are sorted \
                 by correctness probability in a descending order.
        """
        dtest = xgb.DMatrix(features)
        test_probs = self.bst.predict(dtest, ntree_limit=self.bst.best_ntree_limit)
        return rank_candidates(candidates, test_probs, n_candidates, return_all)

    def dump(self):
        """Describe the model for introspection."""
        return "Attributes: %s\nParameters: %s" % (
            sorted(self.bst.attributes().items()) if self.bst is not None else "<not trained>",
            sorted(self.config["boost_param"].items()))

    def __eq__(self, other: "CandidatesRanker") -> bool:
        if self.config != other.config:
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
            labels.append(int(row[Columns.Candidate] == identifiers[row[Columns.Id]]))
        return numpy.array(labels)

    def _generate_tree(self) -> dict:
        tree = {"config": self.config}
        if self.bst is None:
            tree["bst"] = numpy.array([])
        else:
            tree["bst"] = numpy.array(self.bst.save_raw())
            tree["best_ntree_limit"] = self.bst.best_ntree_limit
        return tree

    def _load_tree(self, tree: dict) -> None:
        self.__dict__.update(tree)
        if self.bst.shape == (0,):
            self.bst = None
        else:
            self.bst = xgb.Booster(model_file=tree["bst"].data)
            self.bst.best_ntree_limit = self.best_ntree_limit
            del self.best_ntree_limit
