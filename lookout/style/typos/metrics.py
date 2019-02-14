from functools import partial
from typing import Dict

import numpy
import pandas

from lookout.style.typos.utils import Columns


class Scores:
    """"
    Class to store scores of solutions for binary classification problems.

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


def detection_score(data: pandas.DataFrame, suggestions: dict) -> Scores:
    """
    Calculate score of solution for typo detection problem.

    Typo is detected right: token is corrected then and only then when the token is not typoed.
    Token is corrected: the first suggestion doesn't match the token.
    :param data: DataFrame which indexed by "id" and has columns "typo", "corrupted".
    :param suggestions: {id : [(candidate, correct_prob)]}, candidates are sorted
                        by correct_prob in a descending order .
    :return: Scores of suggestions on detection problem.
    """
    scores = Scores()
    for i in data.index:
        if data.loc[i, Columns.Token] != data.loc[i, Columns.CorrectToken]:
            if suggestions[i][0][0] != data.loc[i, Columns.Token]:
                scores.tp += 1
            else:
                scores.fn += 1
        else:
            if suggestions[i][0][0] == data.loc[i, Columns.Token]:
                scores.tn += 1
            else:
                scores.fp += 1
    return scores


def first_k_set(corrections: list, k: int) -> set:
    """
    Get set of k most probable correction tokens.

    :param corrections: List of corrections, sorted by probability.
    :param k: Number of corrections to take.
    :return: Set of k most probable correction tokens.
    """
    first_k = set()
    for correction, _prob in corrections[:k]:
        first_k.add(correction)
    return first_k


def score_at_k(data: pandas.DataFrame, suggestions: dict, k: int = 1) -> Scores:
    """
    Calculate score@k of solution for typo correction problem.

    The suggestions for typo correction are considered right
    when there is a right one among the first k.
    :param data: DataFrame which is indexed by "id" and
                 has columns "typo", "corrupted".
    :param suggestions: {id : [(candidate, correct_prob)]},
                        candidates inside one suggestions list are
                        sorted by correct_prob in a descending order.
    :param k: Number of the first suggested corrections to check.
    :return: Scores of suggestions for score@k problem.
    """
    scores = Scores()
    for i in data.index:
        if data.loc[i, Columns.Token] != data.loc[i, Columns.CorrectToken]:
            if data.loc[i, Columns.CorrectToken] in first_k_set(suggestions[i], k):
                scores.tp += 1
            else:
                scores.fn += 1
        else:
            if data.loc[i, Columns.CorrectToken] in first_k_set(suggestions[i], k):
                scores.tn += 1
            else:
                scores.fp += 1
    return scores


def score_at_k_on_corrections(data: pandas.DataFrame, suggestions: dict, k: int = 1) -> Scores:
    """
    Calculate score@k of solution for typo correction problem only on correcting suggestions.

    The suggestions for typo correction are considered right if there is a right one among
    the first k. Only the tokens, considered to be typoed, are taken into account
    (correcting suggestions).
    :param data: DataFrame which is indexed by "id" and has columns "typo", "corrupted".
    :param suggestions: {id : [(candidate, correct_prob)]},
                        candidates inside one suggestions list are
                        sorted by correct_prob in a descending order.
    :param k: Number of the first suggested corrections to check.
    :return: Scores of correcting suggestions for score@k problem .
    """
    scores = Scores()
    for i in data.index:
        if suggestions[i][0][0] != data.loc[i, Columns.Token]:
            if data.loc[i, Columns.Token] != data.loc[i, Columns.CorrectToken]:
                if data.loc[i, Columns.CorrectToken] in first_k_set(suggestions[i], k):
                    scores.tp += 1
                else:
                    scores.fn += 1
            else:
                if data.loc[i, Columns.CorrectToken] in first_k_set(suggestions[i], k):
                    scores.tn += 1
                else:
                    scores.fp += 1
    return scores


def print_score_metrics(score: Scores, file=None):
    """Print score metrics."""
    print(score, file=file)
    print("Accuracy:", score.accuracy, file=file)
    print("Precision:", score.precision, file=file)
    print("Recall:", score.recall, file=file)
    print("F1:", score.f1, file=file)


def print_suggestion_results(data: pandas.DataFrame, suggestions: dict, path: str = None) -> None:
    """Print suggestion results."""
    file = None if not path else open(path, "w")
    print("DETECTION SCORE\n", file=file)
    print_score_metrics(detection_score(data, suggestions), file=file)
    print("\nFIRST SUGGESTION SCORE\n", file=file)
    print_score_metrics(score_at_k(data, suggestions, 1), file=file)
    print("\nFIRST TWO SUGGESTIONS SCORE\n", file=file)
    print_score_metrics(score_at_k(data, suggestions, 2), file=file)
    print("\nFIRST THREE SUGGESTIONS SCORE\n", file=file)
    print_score_metrics(score_at_k(data, suggestions, 3), file=file)


def print_scores(data: pandas.DataFrame, suggestions: dict, path: str = None) -> None:
    """Print scores for suggestions in an easy readable way."""
    file = None if not path else open(path, "w")
    print("{:15s}|{:15s}|{:15s}|{:15s}|{:15s}|{:15s}|{:15s}|{:15s}".format("METRICS",
                                                                           "DETECTION SCORE",
                                                                           "TOP1 SCORE CORR",
                                                                           "TOP2 SCORE CORR",
                                                                           "TOP3 SCORE CORR",
                                                                           "TOP1 SCORE ALL",
                                                                           "TOP2 SCORE ALL",
                                                                           "TOP3 SCORE ALL"),
          file=file)
    print(("-" * 15 + "|") * 7 + "-" * 15, file=file)
    scores = []
    for score_func in [detection_score,
                       partial(score_at_k_on_corrections, k=1),
                       partial(score_at_k_on_corrections, k=2),
                       partial(score_at_k_on_corrections, k=3),
                       partial(score_at_k, k=1),
                       partial(score_at_k, k=2),
                       partial(score_at_k, k=3)]:
        score = score_func(data, suggestions)
        scores.append([score.accuracy,
                       score.precision,
                       score.recall,
                       score.f1])
    scores = numpy.array(scores).transpose()
    for i, metrics in enumerate(["Accuracy", "Precision", "Recall", "F1"]):
        print("{:15s}| {:13.3f} | {:13.3f} | {:13.3f} | {:13.3f} | {:13.3f} | {:13.3f} | {:13.3f} "
              "".format(metrics,
                        scores[i, 0],
                        scores[i, 1],
                        scores[i, 2],
                        scores[i, 3],
                        scores[i, 4],
                        scores[i, 5],
                        scores[i, 6]),
              file=file)
