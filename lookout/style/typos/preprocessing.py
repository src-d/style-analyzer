from typing import Set

import pandas

from lookout.style.typos.utils import Columns


def check_split(text: str, tokens: Set[str]) -> bool:
    """
    Check whether all tokens of the text belong to the given set.

    :param text: String of space-separated tokens.
    :param tokens: Set of tokens (reference vocabulary).
    :return: True when all tokens of the `text` belong to `tokens`.
    """
    for token in text.split():
        if token not in tokens:
            return False
    return True


def filter_splits(data: pandas.DataFrame, tokens: Set[str]) -> pandas.DataFrame:
    """
    Leave rows in a dataframe whose splits' tokens all belong to some vocabulary.

    :param data: Dataframe which contains column Columns.Split.
    :param tokens: Set of tokens (reference vocabulary).
    :return: Filtered dataframe.
    """
    return data[data[Columns.Split].apply(lambda x: check_split(x, tokens))]
