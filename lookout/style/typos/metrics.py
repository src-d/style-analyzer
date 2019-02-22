from typing import Dict, List, Set, Tuple

import pandas
from sklearn.metrics import classification_report

from lookout.style.typos.utils import Columns


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
    'correction': Correctly spelled tokens should not be corrected. Typo-ed tokens should \
                  contain the right correction among first k suggestions.
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
    y_true, y_pred = [], []
    for i in data.index:
        if mode == "on_corrected" and suggestions[i][0][0] == data.loc[i, Columns.Token]:
            continue
        corrected_right = (data.loc[i, Columns.CorrectToken] in first_k_set(suggestions[i], k))
        if mode == "detection":
            corrected_right = (suggestions[i][0][0] != data.loc[i, Columns.Token])
        y_pred.append(corrected_right)
        y_true.append(data.loc[i, Columns.Token] != data.loc[i, Columns.CorrectToken])
    return classification_report(y_true, y_pred, output_dict=True)
