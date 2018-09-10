from itertools import chain

import numpy
import pandas


def collect_embeddings(fasttext, tokens) -> numpy.ndarray:
    return numpy.array([fasttext.get_word_vector(token) for token in tokens])


def read_frequencies(file: str) -> dict:
    """
    Read token frequencies from the file.
    :param file: Path to the file containing tokens with frequencies.
                Every line has format "token count"
    :return: Dictionary of tokens frequencies
    """
    frequencies = {}
    with open(file, "r") as f:
        for line in f:
            split = line.split()
            frequencies[split[0]] = int(split[1])
    return frequencies


def read_vocabulary(file: str) -> list:
    """
    Read vocabulary tokens from the frequencies file
    :param file: Path to the file containing tokens with frequencies.
                Every line has format "token count"
    :return: List of tokens from the vocabulary
    """
    with open(file, "r") as f:
        tokens = [line.split()[0] for line in f]
    return tokens


def flatten(data: pandas.DataFrame, column: str, new_column: str,
            apply_function=lambda x: x) -> pandas.DataFrame:
    """
    Flatten dataframe on "column" with extracted elements put to "new_column"
    """
    flat_column = data[column].apply(apply_function).tolist()
    flat_values = numpy.repeat(data.values,
                               repeats=numpy.array(list(map(lambda x: len(x), flat_column))),
                               axis=0)
    flat_column = list(chain.from_iterable(flat_column))
    result = pandas.DataFrame(flat_values, columns=data.columns)
    result[new_column] = flat_column
    return result.infer_objects()


def flatten_data(data: pandas.DataFrame, new_column_name="identifier") -> pandas.DataFrame:
    """
    Flatten identifiers data in column "token_split"
    :param data: DataFrame containing column "token_split" with splitted identifiers
        either as strings or as lists of tokens
    :param new_column_name: Name of column to put tokens from splits to
    :return:
    """
    apply_function = (lambda x: x) if isinstance(data.token_split.tolist()[0], list) \
        else (lambda x: str(x).split())
    return flatten(data, "token_split", new_column_name, apply_function=apply_function)


def add_context_info(data: pandas.DataFrame) -> pandas.DataFrame:
    """
    Split context of identifier on before and after part
    :param data: DataFrame. Column "token_split" will be used for
        creating context info if present
    :return: Provided data with added columns "before" and "after", containing lists
        of corresponding contexts tokens
    """
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

    return data.infer_objects()


def rank_candidates(candidates: pandas.DataFrame, pred_proba: list, n_candidates: int = None,
                    return_all: bool = True):
    """
    Rank candidates for typos correction based on correctness probabilities.
    :param candidates: Dataframe with columns "id", "typo", "candidate"
        and indexed by range(len(pred_proba))
    :param pred_proba: Array of probabilities of correctness of every candidate
    :param n_candidates: Number of most probably correct candidates to return for each typo
    :param return_all: False to return corrections only for
        corrected in the first candidate typos
    :return: Dictionary {id : [[candidate, correct_prob]]}, candidates are sorted
        by correct_prob in a descending order.
    """
    suggestions = {}
    corrections = []
    for i in range(len(pred_proba)):
        index = candidates.loc[i, "id"]
        corrections.append([candidates.loc[i, "candidate"], pred_proba[i]])

        if i < len(pred_proba) - 1 and candidates.loc[i + 1, "id"] == index:
            continue

        corrections = list(sorted(corrections, key=lambda x: -x[1]))
        typo = candidates.loc[i, "typo"]
        if corrections[0][0] != typo:
            suggestions[index] = (corrections if n_candidates is None
                                  else corrections[:n_candidates])
        elif return_all:
            suggestions[index] = [[typo, 1.0]]
        corrections = []

    return suggestions


def filter_suggestions(typos: pandas.DataFrame, suggestions: dict, n_candidates: int=1,
                       return_all: bool=False) -> dict:
    """
    Filter corrections suggestions
    :param typos: DataFrame, containing column "typo"
    :param suggestions: Dictionary of suggestions, keys correspond with typos.index
    :param n_candidates: Number of most probably correct candidates to return for each typo
    :param return_all: False to return corrections only for
        corrected in the first candidate typos
    :return: Dictionary of filtered suggestions
    """
    return {index: suggestions[index][:n_candidates] for index, row in typos.iterrows()
            if return_all or suggestions[index][0][0] != row["typo"]}


def suggestions_to_df(typos: pandas.DataFrame, suggestions: dict) -> pandas.DataFrame:
    """
    Convert suggestions from dictionary to pandas.DataFrame
    :param typos: DataFrame containing column "typo"
    :param suggestions: Dictionary of suggestions, keys correspond with typos.index
    :return: DataFrame with columns "typo", "suggestions", indexed by typos.index
    """
    suggestions_array = [[index, typos.loc[index, "typo"], corrections]
                         for index, corrections in suggestions.items()]
    return pandas.DataFrame(suggestions_array, columns=["id", "typo", "suggestions"],
                            index=typos.index).infer_objects()


def suggestions_to_flat_df(typos: pandas.DataFrame, suggestions: dict) -> pandas.DataFrame:
    """
    Convert suggestions from dictionary to pandas.DataFrame, flattened by suggestions column
    :param typos: DataFrame containing column "typo"
    :param suggestions: Dictionary of suggestions, keys correspond with typos.index
    :return: DataFrame with columns "typo", "candidate", "proba", indexed by typos.index
    """
    flat_df = flatten(suggestions_to_df(typos, suggestions), "suggestions", "suggestion")
    flat_df["candidate"] = [suggestion[0] for suggestion in flat_df.suggestion]
    flat_df["proba"] = [suggestion[1] for suggestion in flat_df.suggestion]
    flat_df = flat_df.drop(columns=["suggestions", "suggestion"])
    return flat_df.infer_objects()
