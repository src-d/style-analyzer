import lzma
from typing import Set

import numpy
import pandas

from lookout.style.typos.utils import COLUMNS


def print_frequencies(tokens_set: Set[str], frequency_column: str, id_stats: pandas.DataFrame,
                      path: str) -> None:
    """
    Dump frequencies of tokens from tokens_set to a file.

    Frequencies info is obtained from id_stats dataframe.
    """
    frequencies = {}
    for token, row in id_stats.iterrows():
        if token in tokens_set:
            frequencies[token] = row[frequency_column]
    with lzma.open(path, "wt") as f:
        for token in list(tokens_set):
            print(token, frequencies[token], file=f)


def cli_print_frequencies(args):
    """CLI entry point for print_frequencies."""
    id_stats = pandas.read_csv(args.stats_file, index_col=COLUMNS["TOKEN"])
    tokens_set = set(numpy.load(args.vocabulary_file))
    print_frequencies(tokens_set, id_stats, args.out_file)
