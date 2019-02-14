from typing import Dict, List, Set, Tuple

import pandas

from lookout.style.typos.utils import Columns


class Scores:
    """"
    Class to store scores of solutions of binary classification problems.

    tp: true positive, fp: false positive, tn: true negative, fn: false negative.
    """
    def __init__(self, tp: int = 0, fp: int = 0, tn: int = 0, fn: int = 0):
        self.tp = tp
        self.fp = fp
        self.tn = tn
        self.fn = fn

    def total(self) -> int:
        """Get total number of examples."""
        return self.tp + self.fp + self.tn + self.fn
    
    def accuracy(self) -> float:
        """Calculate accuracy"""
        return (self.tp + self.tn) / self.total()

    def precision(self) -> float:
        """Calculate precision."""
        return self.tp / (self.tp + self.fp)

    def recall(self) -> float:
        """Calculate recall."""
        return self.tp / (self.tp + self.fn)

    def f1(self) -> float:
        """Calculate f1 score."""
        return 2 / (1 / self.precision() + 1 / self.recall())

    def get_metrics(self) -> Dict[str, float]:
        return {"accuracy": self.accuracy(),
                "precision": self.precision(),
                "recall": self.recall(),
                "f1": self.f1()}

    def __eq__(self, other):
        return self.__dict__ == other.__dict__


def first_k_set(corrections: List[Tuple[str, float]], k: int) -> Set[str]:
    """
    Compose the set of k most probable correction candidates (tokens without probabilities).

    :param corrections: List of corrections, sorted by probability.
    :param k: Number of corrections to take.
    :return: Set of k most probable correction tokens.
    """
    first_k = set()
    for correction, _prob in corrections[:k]:
        first_k.add(correction)
    return first_k


def get_score(data: pandas.DataFrame, suggestions: Dict[int, List[Tuple[str, float]]],
              mode: str = "correction", k: int = 1) -> Scores:
    """
    Calculate the score of the solution of the specific typo correction problem.

    Token is considered corrected, when the first suggestion doesn't match the token.
    Supports three problems:
    'detection': Typo is detected right: token is corrected when and only when it is not typo-ed.
    'correction': The suggestions for typo correction are considered correct when there is \
                  a correct one among the first k.
    'on_corrected': Same as `correction`, but only the tokens, corrected by \
                               the suggestions, are taken into account.
    :param data: DataFrame which is indexed by Columns.Id and has columns Column.Token and \
                 Column.CorrectToken.
    :param suggestions: `{id : [(candidate, correct_prob)]}`, candidates are sorted \
                        by correct_prob in a descending order .
    :param mode: One of 'detection', 'correction', 'on_corrected'.
    :param k: Number of the first suggested corrections to check. Used in modes \
              'correction', 'on_corrected'.
    :return: Scores of the suggestions.
    """
    scores = Scores()
    for i in data.index:
        if mode == "on_corrected" and suggestions[i][0][0] != data.loc[i, Columns.Token]:
            continue
        if mode == "correction":
            corrected_right = (data.loc[i, Columns.CorrectToken] in first_k_set(suggestions[i], k))
        elif mode == "detection":
            corrected_right = (suggestions[i][0][0] != data.loc[i, Columns.Token])
        else:
            raise ValueError("Mode must be one either `detection`, `correction` or `on_corrected`")
        token_typoed = data.loc[i, Columns.Token] != data.loc[i, Columns.CorrectToken]

        if token_typoed and corrected_right:
            scores.tp += 1
        elif token_typoed and not corrected_right:
            scores.fn += 1
        elif not token_typoed and corrected_right:
            scores.tn += 1
        else:
            scores.fp += 1
    return scores


def print_all_scores(data: pandas.DataFrame, suggestions: Dict[int, List[Tuple[str, float]]],
                     path: str = None) -> None:
    """Print scores for suggestions in an easy readable way."""
    file = None if not path else open(path, "w")
    print("%-20s| %-10s| %-10s| %-10s| %-10s" %
          ("Metrics", "Accuracy", "Precision", "Recall", "F1"), file=file)
    print("-" * 20 + "|" + ("-" * 11 + "|") * 3 + "-" * 11, file=file)
    scores = [get_score(data, suggestions, mode="detection")]
    for mode in ["on_corrected", "correction"]:
        for k in [1, 2, 3]:
            scores.append(get_score(data, suggestions, mode=mode, k=k).get_metrics())
    for i, score_name in enumerate(["DETECTION SCORE", "TOP1 SCORE ON CORR",
                                    "TOP2 SCORE ON CORR", "TOP3 SCORE ON CORR",
                                    "TOP1 SCORE ALL", "TOP2 SCORE ALL", "TOP3 SCORE ALL"]):
        print("%-20s| %-10s| %-10s| %-10s| %-10s" % (
            score_name, scores[i]["accuracy"], scores[i]["precision"], scores[i]["recall"],
            scores[i]["f1"]), file=file)
    if path:
        file.close()
