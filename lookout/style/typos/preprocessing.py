from typing import Set, Union

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


def filter_splits(data: Union[pandas.DataFrame, str], tokens: Set[str]) -> pandas.DataFrame:
    """
    Leave rows in a dataframe whose splits' tokens all belong to some vocabulary.

    :param data: Dataframe or its .csv dump, containing column Columns.Split.
    :param tokens: Set of tokens (reference vocabulary).
    :return: Filtered dataframe.
    """
    if isinstance(data, str):
        data = pandas.read_csv(data, index_col=0)
    return data.loc[data[Columns.Split].apply(lambda x: check_split(x, tokens))]
