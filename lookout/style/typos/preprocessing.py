import csv
from typing import Set

import pandas
from smart_open import smart_open

from lookout.style.typos.utils import Columns


def filter_splits(data: pandas.DataFrame, tokens: Set[str]) -> pandas.DataFrame:
    """
    Leave rows in a dataframe whose splits' tokens all belong to some vocabulary.

    :param data: Dataframe which contains column Columns.Split.
    :param tokens: Set of tokens (reference vocabulary).
    :return: Filtered dataframe.
    """
    return data[data[Columns.Split].apply(lambda x: set(x.split()).issubset(tokens))]


def print_frequencies(tokens: Set[str], id_stats: pandas.DataFrame, path: str) -> None:
    """
    Print frequencies of tokens to a file.

    Frequencies info is obtained from id_stats dataframe.
    :param tokens: Set of tokens, for which frequencies should be printed.
    :param id_stats: Dataframe with frequency information for tokens. \
                     It must be indexed by tokens and contain column Columns.Frequency.
    :param path: Path to a .csv file to print frequencies to.
    """
    frequencies = id_stats.loc[tokens].dropna()[Columns.Frequency]
    with smart_open(path, "w") as f:
        writer = csv.writer(f)
        for line in frequencies.items():
            writer.writerow(line)
