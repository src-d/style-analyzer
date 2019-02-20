from typing import Set, Union

import pandas

from lookout.style.typos.utils import Columns


def check_split(split: str, tokens_set: Set[str]) -> bool:
    """
    Check whether all elements of the split are in tokens_set.

    :param split: String of space-separated tokens.
    :param tokens_set: Set of tokens (reference vocabulary).
    :return: True when all tokens of split belong to tokens_set.
    """
    if type(split) != str:
        return False

    for token in split.split():
        if token not in tokens_set:
            return False
    return True


def filter_splits(data: Union[pandas.DataFrame, str], tokens_set: Set[str]) -> pandas.DataFrame:
    """
    Leave rows in a dataframe whose splits' tokens all belong to some vocabulary.

    :param data: Dataframe or its .csv dump, containing column Columns.Split.
    :param tokens_set: Set of tokens (reference vocabulary).
    :return: Filtered dataframe.
    """
    token_split = list(data[Columns.Split])
    filter_array = list(map(lambda x: check_split(x, tokens_set), token_split))
    return data[filter_array]
