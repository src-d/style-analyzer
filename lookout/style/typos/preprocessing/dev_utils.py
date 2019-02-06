import pandas
import random

from lookout.style.typos.utils import COLUMNS


def rand_bool(true_prob: float):
    """
    Returns True with probability true_prob
    """
    return random.uniform(0, 1) < true_prob


def detection_score(data: pandas.DataFrame, suggestions: dict) -> dict:
    """
    Calculates score of solution for typo detection problem.

    data: DataFrame which indexed by "id" and has columns "typo", "corrupted".

    suggestions: {id : [(candidate, correct_prob)]}, candidates are sorted
                 by correct_prob in a descending order .
    """
    scores = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    for i in data.index:
        if data.loc[i, COLUMNS["TOKEN"]] != data.loc[i, COLUMNS["CORRECT_TOKEN"]]:
            if suggestions[i][0][0] != data.loc[i, COLUMNS["TOKEN"]]:
                scores["tp"] += 1
            else:
                scores["fn"] += 1
        else:
            if suggestions[i][0][0] == data.loc[i, COLUMNS["TOKEN"]]:
                scores["tn"] += 1
            else:
                scores["fp"] += 1
    return scores


def first_k_set(corrections: list, k: int) -> set:
    first_k = set()
    for correction, prob in corrections[:k]:
        first_k.add(correction)
    return first_k


def score_at_k(data, suggestions, k):
    """
    Calculates score of solution for typo correction problem.
    The suggestions for typo correction are considered correct
    if there is a right one among the first k.

    data: DataFrame which is indexed by "id" and
           has columns "typo", "corrupted".

    suggestions: {id : [(candidate, correct_prob)]},
                 candidates inside one suggestions list are
                 sorted by correct_prob in a descending order.
    """
    scores = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    for i in data.index:
        if data.loc[i, COLUMNS["TOKEN"]] != data.loc[i, COLUMNS["CORRECT_TOKEN"]]:
            if data.loc[i, COLUMNS["CORRECT_TOKEN"]] in first_k_set(suggestions[i], k):
                scores["tp"] += 1
            else:
                scores["fn"] += 1
        else:
            if data.loc[i, COLUMNS["CORRECT_TOKEN"]] in first_k_set(suggestions[i], k):
                scores["tn"] += 1
            else:
                scores["fp"] += 1
    return scores


def accuracy(score):
    return (score["tp"] + score["tn"]) / sum(score.values())


def precision(score):
    return score["tp"] / (score["tp"] + score["fp"])


def recall(score):
    return score["tp"] / (score["tp"] + score["fn"])


def f1(score):
    return 2 / (1 / precision(score) + 1 / recall(score))


def print_score_metrics(score, file=None):
    print(score, file=file)
    print("Accuracy:", accuracy(score), file=file)
    print("Precision:", precision(score), file=file)
    print("Recall:", recall(score), file=file)
    print("F1:", f1(score), file=file)


def print_suggestion_results(data, suggestions, file=None):
    print("DETECTION SCORE\n", file=file)
    print_score_metrics(detection_score(data, suggestions), file=file)
    print("\nFIRST SUGGESTION SCORE\n", file=file)
    print_score_metrics(score_at_k(data, suggestions, 1), file=file)
    print("\nFIRST TWO SUGGESTIONS SCORE\n", file=file)
    print_score_metrics(score_at_k(data, suggestions, 2), file=file)
    print("\nFIRST THREE SUGGESTIONS SCORE\n", file=file)
    print_score_metrics(score_at_k(data, suggestions, 3), file=file)
