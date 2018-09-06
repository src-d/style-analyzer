import itertools
from itertools import chain

import numpy
import pandas


def concatenate_ones(matrix):
    return numpy.concatenate((matrix, numpy.ones((matrix.shape[0], 1))), axis=1)


def collect_embeddings(fasttext, tokens) -> numpy.ndarray:
    vecs = []
    for token in tokens:
        vecs.append(fasttext.get_word_vector(token))
    vecs = numpy.array(vecs)
    return vecs


def read_frequencies(file) -> dict:
    """
    Read token frequencies from file.
    :param file: path to file containing tokens with frequencies.
                Every line has a format "token count"
    :return: dict of tokens frequencies
    """
    frequencies = {}
    with open(file, "r") as f:
        for line in f:
            split = line.split()
            frequencies[split[0]] = int(split[1])
    return frequencies


def read_vocabulary(file) -> list:
    """
    Read tokens from frequencies file
    :param file: path to file containing tokens with frequencies.
                Every line has a format "token count"
    :return: list of tokens of vocabulary
    """
    tokens = []
    with open(file, "r") as f:
        for line in f:
            split = line.split()
            tokens.append(split[0])
    return tokens


def join_lists(lists) -> list:
    return list(chain.from_iterable([l for l in lists]))


def flatten(dataframe: pandas.DataFrame, column: str, new_column: str,
            apply_function=lambda x: x) -> pandas.DataFrame:
    """
    Flatten dataframe on 'column' with extracted elements put to 'new_column'
    """
    other_columns = list(dataframe.columns)
    flat_other = numpy.repeat(dataframe.loc[:, other_columns].values,
                              repeats=numpy.array(dataframe[column].apply(
                                  lambda x: len(apply_function(x))).tolist()),
                              axis=0)
    flat_column = list(itertools.chain.from_iterable(dataframe[column].apply(
        apply_function).tolist()))
    result = pandas.DataFrame(flat_other, columns=other_columns)
    result[new_column] = flat_column
    return result


def flatten_data(data: pandas.DataFrame, new_column_name="identifier"):
    apply_function = (lambda x: x) if isinstance(data.token_split[0], list) \
        else (lambda x: str(x).split())
    return flatten(data, 'token_split', new_column_name, apply_function=apply_function)


def add_context_info(data: pandas.DataFrame) -> pandas.DataFrame:
    if "before" in data.columns and "after" in data.columns:
        return data

    tokens = list(data.typo)

    if "token_split" in data.columns:
        token_split = list(data.token_split)
        before = []
        after = []

        for row_number in range(len(data)):
            if "token_split" in data.columns:
                split = token_split[row_number]
                if isinstance(split, str):
                    split = split.split()
                index = split.index(tokens[row_number])
                before.append(split[:index])
                after.append(split[index + 1:])

    else:
        before = [[] for _ in range(len(data))]
        after = [[] for _ in range(len(data))]

    data["before"] = before
    data["after"] = after

    return data


def rank_candidates(candidates, pred_proba, n_candidates=None, return_all=True):
    """
    Suggest typos corrections base on candidates
    and correctness probabilities.

    candidates: dataframe with columns "id", "typo", "candidate"
                and indexed by range(len(pred_proba)).

    Returns suggestions: {id : [(candidate, correct_prob)]},
                         candidates are sorted
                         by correct_prob in a descending order.
    """
    suggestions = {}
    corrections = []
    for i in range(len(pred_proba)):
        index = candidates.loc[i, "id"]
        corrections.append((candidates.loc[i, "candidate"], pred_proba[i]))

        if i < len(pred_proba) - 1 and candidates.loc[i + 1, "id"] == index:
            continue

        corrections = list(sorted(corrections, key=lambda x: -x[1]))
        typo = candidates.loc[i, "typo"]
        if corrections[0][0] != typo:
            suggestions[index] = (corrections if n_candidates is None
                                  else corrections[:n_candidates])
        elif return_all:
            suggestions[index] = [(typo, 1.0)]

        corrections = []

    return suggestions


def filter_suggestions(typos: pandas.DataFrame, suggestions: dict,
                       n_candidates: int=1, return_all: bool=False) -> dict:
    corrections = {}
    for index, row in typos.iterrows():
        if return_all or suggestions[index][0][0] != row['typo']:
            corrections[index] = suggestions[index][:n_candidates]

    return corrections


def suggestions_to_df(typos: pandas.DataFrame, suggestions: dict) -> pandas.DataFrame:
    suggestions_array = []
    index = []
    for index in suggestions.keys():
        correction = [(suggestion[0], suggestion[1]) for suggestion in suggestions[index]]
        index.append(index)
        suggestions_array.append([typos.loc[index, 'typo'], correction])

    return pandas.DataFrame(suggestions_array, columns=['typo', 'suggestions'], index=index)


def suggestions_to_flat_df(typos: pandas.DataFrame, suggestions: dict) \
        -> pandas.DataFrame:
    suggestions_df = suggestions_to_df(typos, suggestions)
    flat_df = flatten(suggestions_df, "suggestions", "suggestion")
    flat_df = flat_df.drop(columns=['suggestions'])
    flat_df["candidate"] = [suggestion[0]
                            for suggestion in flat_df.suggestion]
    flat_df["proba"] = [suggestion[1]
                        for suggestion in flat_df.suggestion]
    flat_df = flat_df.drop(columns=["suggestion"])
    return flat_df
