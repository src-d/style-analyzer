from typing import Set

import numpy
import pandas

from lookout.style.typos.utils import COLUMNS


def check_split(split: str, tokens_set: Set[str]):
    if type(split) != str:
        return False

    for token in split.split():
        if token not in tokens_set:
            return False
    return True


def filter_splitted_identifiers(data: pandas.DataFrame, tokens_set: Set[str]) -> pandas.DataFrame:
    """
    Leave rows in a dataframe whose identifiers' tokens are all in tokens_set.
    """
    token_split = list(data[COLUMNS["SPLIT"]])
    filter_array = list(map(lambda x: check_split(x, tokens_set), token_split))
    return data[filter_array]


def filter_identifiers(args):
    data = pandas.read_pickle(args.id_file)
    tokens_set = set(numpy.load(args.vocabulary_file))
    filtered_data = filter_splitted_identifiers(data, tokens_set)
    filtered_data.to_csv(args.out_file)
