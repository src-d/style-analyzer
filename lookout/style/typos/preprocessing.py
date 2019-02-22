import csv
from typing import Set

import pandas
from smart_open import smart_open

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


def print_frequencies(tokens: Set[str], id_stats: pandas.DataFrame,
                      frequency_column: str, path: str) -> None:
    """
    Print frequencies of tokens to a file.

    Frequencies info is obtained from id_stats dataframe.
    :param tokens: Set of tokens, for which frequencies should be printed.
    :param id_stats: Dataframe with frequency information for tokens. \
                     It must be indexed by tokens and contain column `frequency_column`.
    :param frequency_column: Name of the column in `id_stats`, from which to take frequency info.
    :param path: Path to a .csv file to print frequencies to.
    """
    frequencies = id_stats.loc[tokens].dropna()[frequency_column]
    with smart_open(path, "w") as f:
        writer = csv.writer(f, delimiter=" ")
        for line in frequencies.items():
            writer.writerow(line)
