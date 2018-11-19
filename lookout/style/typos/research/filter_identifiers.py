import numpy
import pandas


def check_split(split, tokens_set):
    if type(split) != str:
        return False

    for token in split.split():
        if token not in tokens_set:
            return False
    return True


def filter_splitted_identifiers(id_info, tokens_set, out_file):
    """
    Leave rows in a dataframe whose identifiers' tokens are all in tokens_set
    """
    token_split = list(id_info.token_split)
    filter_array = list(map(lambda x: check_split(x, tokens_set), token_split))

    data = id_info.copy()
    data = data[filter_array]
    data.to_pickle(out_file)


def filter_identifiers(args):
    id_info = pandas.read_pickle(args.id_file)
    tokens_set = set(numpy.load(args.vocabulary_file))
    filter_splitted_identifiers(id_info, tokens_set, args.out_file)
