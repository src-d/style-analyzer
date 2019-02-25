from enum import Enum, unique
from typing import Dict, List, Optional, Set, TextIO, Tuple

import pandas
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from lookout.style.typos.utils import Columns


@unique
class ScoreMode(Enum):
    """"Modes for calculation scores of typos correction"""
    detection = 0
    correction = 1
    on_corrected = 2


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


def get_scores(data: pandas.DataFrame, suggestions: Dict[int, List[Tuple[str, float]]],
               mode: ScoreMode = ScoreMode.correction, k: int = 1) -> Dict[str, float]:
    """
    Calculate the score of the solution of the specific typo correction problem.

    Token is considered corrected, when the first suggestion doesn't match the token.
    Supports three problems:
    `ScoreMode.detection`: Typo is detected right: token is corrected when and only when \
                           it is not typo-ed.
    `ScoreMode.correction`: Correctly spelled tokens should not be corrected. Typo-ed tokens \
                            should  contain the right correction among first k suggestions.
    `ScoreMode.on_corrected`: Same as `correction`, but only the tokens, corrected by \
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
    y_true, y_pred = [], []
    for i in data.index:
        if mode == "on_corrected" and suggestions[i][0][0] == data.loc[i, Columns.Token]:
            continue
        typoed = data.loc[i, Columns.Token] != data.loc[i, Columns.CorrectToken]
        if mode == "detection" or not typoed:
            # If the word is not misspelled, model should not correct it in any mode
            corrected_right = (suggestions[i][0][0] != data.loc[i, Columns.Token])
        else:
            corrected_right = (data.loc[i, Columns.CorrectToken] in first_k_set(suggestions[i], k))
        y_pred.append(corrected_right)
        y_true.append(data.loc[i, Columns.Token] != data.loc[i, Columns.CorrectToken])

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    accuracy = accuracy_score(y_true, y_pred)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def print_all_scores(data: pandas.DataFrame, suggestions: Dict[int, List[Tuple[str, float]]],
                     file: Optional[TextIO]) -> None:
    """Print scores for suggestions in an easy readable way."""
    print("%-20s| %-10s| %-10s| %-10s| %-10s" %
          ("Metrics", "Accuracy", "Precision", "Recall", "F1"), file=file)
    print("-" * 20 + "|" + ("-" * 11 + "|") * 3 + "-" * 11, file=file)
    scores = [get_scores(data, suggestions, ScoreMode.detection)]
    for mode in [ScoreMode.on_corrected, ScoreMode.correction]:
        for k in [1, 2, 3]:
            scores.append(get_scores(data, suggestions, mode, k))
    for i, score_name in enumerate(["DETECTION SCORE", "TOP1 SCORE ON CORR",
                                    "TOP2 SCORE ON CORR", "TOP3 SCORE ON CORR",
                                    "TOP1 SCORE ALL", "TOP2 SCORE ALL", "TOP3 SCORE ALL"]):
        print("%-20s| %-10.3f| %-10.3f| %-10.3f| %-10.3f" % (
            score_name, scores[i]["accuracy"], scores[i]["precision"], scores[i]["recall"],
            scores[i]["f1"]), file=file)
