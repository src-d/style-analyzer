"""Optimize base model hyper-parameters."""
from functools import partial
from logging import getLogger
from threading import Thread
import time
from typing import Any, Mapping, Optional, Sequence, Tuple

from lookout.core.slogging import logs_are_structured
import numpy
from scipy.optimize import OptimizeResult
from scipy.sparse import csr_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from skopt import gp_minimize
from skopt.space import Categorical, Integer
from skopt.utils import use_named_args


class Optimizer:
    """Optimize base model hyper-parameters."""

    _log = getLogger("Optimizer")

    def __init__(self, cv: int, n_iter: int, n_jobs: Optional[int], random_state: int,
                 base_model_name_categories: Sequence[str],
                 max_depth_categories: Sequence[Optional[int]],
                 max_features_categories: Sequence[Optional[str]], min_samples_leaf_min: int,
                 min_samples_leaf_max: int, min_samples_split_min: int,
                 min_samples_split_max: int) -> None:
        """
        Construct an `Optimizer`.

        :param cv: Number of folds to use during cross-validation.
        :param n_iter: Number of optimization iterations. Minimum 10.
        :param n_jobs: Number of jobs to use. Passed on to cross_val_score.
        :param random_state: Random seed.
        :param base_model_name_categories: Base model names considered during search.
        :param max_depth_categories: Depths considered during search.
        :param max_features_categories: Features considered during search.
        :param min_samples_leaf_min: Minimum of the minimum of samples in a leaf considered \
                                     during search.
        :param min_samples_leaf_max: Maximum of the minimum of samples in a leaf considered \
                                     during search.
        :param min_samples_split_min: Minimum of the minimum of samples in a split considered \
                                      during search.
        :param min_samples_split_max: Maximum of the minimum of samples in a split considered \
                                      during search.
        """
        self.cv = cv
        if n_iter < 10:
            self._log.warning("n_iter values below 10 (%d) are considered as 10.", n_iter)
        self.n_iter = max(10, n_iter)
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.dimensions = [
            Categorical(name="base_model_name", categories=base_model_name_categories),
            Categorical(name="max_depth", categories=max_depth_categories),
            Categorical(name="max_features", categories=max_features_categories),
            Integer(name="min_samples_split", low=min_samples_split_min,
                    high=min_samples_split_max),
            Integer(name="min_samples_leaf", low=min_samples_leaf_min, high=min_samples_leaf_max),
        ]

    def optimize(self, X: csr_matrix, y: numpy.ndarray) -> Tuple[float, Mapping[str, Any]]:
        """
        Conduct hyper-parameters search to find the best base model given the data.

        :param X: Sparse feature matrix.
        :param y: Labels numpy array.
        :return: Best base model score and parameters.
        """
        cost_function = use_named_args(self.dimensions)(partial(self._cost, X=X, y=y))

        def _minimize() -> OptimizeResult:
            callback = _VerboseLogCallback(self._log)
            return gp_minimize(cost_function, self.dimensions, n_calls=self.n_iter,
                               random_state=self.random_state, callback=callback)

        if not logs_are_structured:
            # fool the check in joblib - everything still works without it
            # this trick allows to run parallel bscv.fit()
            from unittest.mock import patch
            with patch("threading._MainThread", Thread):
                self._log.debug("patched joblib")
                res = _minimize()
        else:
            res = _minimize()
        best_score = -res.fun
        best_params = {dim.name: x for x, dim in zip(res.x, self.dimensions)}
        return best_score, best_params

    def _cost(self, *, X: csr_matrix, y: numpy.ndarray, **params: Any) -> float:
        params_copy = params.copy()
        base_model_name = params_copy.pop("base_model_name")
        if base_model_name == "sklearn.tree.DecisionTreeClassifier":
            base_model_class = DecisionTreeClassifier
        elif base_model_name == "sklearn.ensemble.RandomForestClassifier":
            base_model_class = RandomForestClassifier
            params_copy["n_estimators"] = 10
            params_copy["n_jobs"] = -1
        params_copy["random_state"] = self.random_state
        base_model = base_model_class(**params_copy)
        cv = StratifiedKFold(self.cv, random_state=self.random_state)
        return -numpy.mean(cross_val_score(base_model, X, y, cv=cv, n_jobs=self.n_jobs))


class _VerboseLogCallback:
    """
    Callback to control the verbosity and log output properly.

    Adopted from skopt library, VerboseCallback class.
    """

    def __init__(self, log):
        """
        Init method.

        :param log: logger Instance to log optimization steps.
        """
        self._log = log
        self.iter_no = 1
        self._start_time = time.time()

    def __call__(self, res):
        """
        Call callback method.

        :param res: The optimization as a OptimizeResult object.
        :return: None
        """
        self._log.debug(
            "Iteration No: %.3d. Time taken: %0.4f. Function value obtained: %0.4f. Current"
            " minimum: %0.4f.", self.iter_no, time.time() - self._start_time, res.func_vals[-1],
            res.fun)
        self.iter_no += 1
        self._start_time = time.time()
