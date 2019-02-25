from enum import Enum, unique
import os
from typing import Dict, List, Set, Tuple

import pandas
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from lookout.style.common import load_jinja2_template
from lookout.style.typos.utils import Columns


@unique
class ScoreMode(Enum):
    """Modes for calculation scores of typos correction."""

    detection = "detection"
    correction = "correction"
    on_corrected = "on_corrected"


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
        if mode == ScoreMode.on_corrected and suggestions[i][0][0] == data.loc[i, Columns.Token]:
            continue
        typoed = data.loc[i, Columns.Token] != data.loc[i, Columns.CorrectToken]
        if mode == ScoreMode.detection or not typoed:
            # If the word is not misspelled, model should not correct it in any mode
            corrected_right = (suggestions[i][0][0] != data.loc[i, Columns.Token])
        else:
            corrected_right = (data.loc[i, Columns.CorrectToken] in first_k_set(suggestions[i], k))
        y_pred.append(corrected_right)
        y_true.append(typoed)

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    accuracy = accuracy_score(y_true, y_pred)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def generate_report(data: pandas.DataFrame, suggestions: Dict[int, List[Tuple[str, float]]],
                    ) -> str:
    """Print scores for suggestions in an easy readable way."""
    scores = {ScoreMode.detection.value: get_scores(data, suggestions, ScoreMode.detection)}
    for mode in [ScoreMode.on_corrected, ScoreMode.correction]:
        for k in [1, 2, 3]:
            scores["Top %i score %s" % (k, mode.value)] = get_scores(data, suggestions, mode, k)
    template = load_jinja2_template(os.path.join(os.path.dirname(__file__), "..", "templates"),
                                    "scores.md.jinja2")
    return template.render(scores=scores)
