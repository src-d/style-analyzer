from typing import Set

import numpy
import pandas

from lookout.style.typos.utils import COLUMNS


def print_frequencies(tokens_set: Set[str], id_stats: pandas.DataFrame, file: str) -> None:
    """
    Dump frequencies of tokens from tokens_set to a file. Frequencies info is obtained
    from id_stats dataframe.
    """
    frequencies = {}
    for token, row in id_stats.iterrows():
        if token in tokens_set:
            frequencies[token] = row.num_occ
    with open(file, "w") as f:
        for token in list(tokens_set):
            print(token, frequencies[token], file=f)


def get_frequencies(args):
    id_stats = pandas.read_csv(args.stats_file, index_col=COLUMNS["TOKEN"])
    tokens_set = set(numpy.load(args.vocabulary_file))
    print_frequencies(tokens_set, id_stats, args.out_file)
