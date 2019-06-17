from enum import Enum, unique
import os
from typing import Dict, List

import pandas
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from lookout.style.common import load_jinja2_template
from lookout.style.typos.utils import Candidate, Columns, TEMPLATE_DIR


@unique
class ScoreMode(Enum):
    """
    Modes for calculation scores of typos correction.

    `ScoreMode.detection`: Typo is detected right: token is corrected when and only when \
                           it is not typo-ed.
    `ScoreMode.correction`: Correctly spelled tokens should not be corrected. Typo-ed tokens \
                            should contain the right correction among first k suggestions.
    `ScoreMode.on_typoed`: Same as `correction`, but only the truly typo-ed tokens \
                           are taken into account.
    """

    detection = "detection"
    correction = "correction"
    on_typoed = "on_typoed"


def get_scores(data: pandas.DataFrame, suggestions: Dict[int, List[Candidate]],
               mode: ScoreMode = ScoreMode.correction, k: int = 1) -> Dict[str, float]:
    """
    Calculate the score of the solution of the specific typo correction problem.

    Token is considered corrected, when the first suggestion doesn't match the token.
    Supports three problems:
    `ScoreMode.detection`: Typo is detected right: token is corrected when and only when \
                           it is not typo-ed.
    `ScoreMode.correction`: Correctly spelled tokens should not be corrected. Typo-ed tokens \
                            should contain the right correction among first k suggestions.
    `ScoreMode.on_typoed`: Same as `correction`, but only the truly typo-ed tokens \
                           are taken into account.
    :param data: DataFrame which is indexed by Columns.Id and has columns Column.Token and \
                 Column.CorrectToken.
    :param suggestions: `{id : [(candidate, correct_prob)]}`, candidates are sorted \
                        by correct_prob in a descending order .
    :param mode: One of `ScoreMode.detection`, `ScoreMode.correction`, `ScoreMode.on_typoed`.
    :param k: Number of the first suggested corrections to check. Used in modes \
              `ScoreMode.correction`, `ScoreMode.on_typoed`.
    :return: Scores of the suggestions.
    """
    y_true, y_pred = [], []
    for i in data.index:
        token = data.loc[i, Columns.Token]
        correct_token = data.loc[i, Columns.CorrectToken]
        suggestion = suggestions.get(i, [Candidate(token, 1.0)])
        typoed = token != correct_token
        if mode == ScoreMode.on_typoed and not typoed:
            continue
        if mode == ScoreMode.detection or not typoed:
            # If the word is not misspelled, model should not correct it in any mode
            corrected = suggestion[0].token != token
        else:
            corrected = correct_token in {candidate.token for candidate in suggestion[:k]}
        y_pred.append(corrected)
        y_true.append(typoed)

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    accuracy = accuracy_score(y_true, y_pred)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def generate_report(data: pandas.DataFrame, suggestions: Dict[int, List[Candidate]],
                    ) -> str:
    """Print scores for suggestions in an easy readable way."""
    template = load_jinja2_template(os.path.join(TEMPLATE_DIR, "scores.md.jinja2"))
    return template.render(ScoreMode=ScoreMode, get_scores=get_scores, **locals())
