"""Various glue functions to work with the input dataset and the output from FastText."""
import csv
from itertools import chain
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy
import pandas
from smart_open import smart_open


Columns = NamedTuple(
    "Columns",
    [("Token", str), ("CorrectToken", str), ("Split", str), ("CorrectSplit", str), ("After", str),
     ("Before", str), ("Id", str), ("Candidate", str), ("Features", str), ("Probability", str),
     ("Suggestions", str), ("Frequency", str)])(
    "token", "correct_token", "token_split", "correct_token_split", "after", "before", "id",
    "candidate", "features", "proba", "suggestions", "freq")


def read_frequencies(file: str) -> Dict[str, int]:
    """
    Read token frequencies from the file.

    :param file: Path to the .csv file with space-separated word-frequency pairs one-per-line.
    :return: Dictionary of tokens frequencies.
    """
    frequencies = {}
    with smart_open(file, "r") as f:
        for line in csv.reader(f):
            frequencies[line[0]] = int(line[1])
    return frequencies


def read_vocabulary(file: str) -> List[str]:
    """
    Read vocabulary tokens from the text file.

    :param file: .csv file in which the vocabulary of corrections candidates is stored. \
                 First token in every line split-by-space is added to the vocabulary. \
    :return: List of tokens of the vocabulary.
    """
    with smart_open(file, "r") as f:
        tokens = [line[0] for line in csv.reader(f)]
    return tokens


def flatten_df_by_column(data: pandas.DataFrame, column: str, new_column: str,
                         apply_function=lambda x: x) -> pandas.DataFrame:
    """
    Flatten DataFrame by `column` with extracted elements put to `new_column`. \
    Operation runs out-of-place.

    :param data: DataFrame to flatten.
    :param column: Column to expand.
    :param new_column: Column to populate with elements from flattened column.
    :param apply_function: Function used to expand every element of flattened column.
    :return: Flattened DataFrame.
    """
    flat_column = data[column].apply(apply_function).tolist()
    flat_values = numpy.repeat(data.values,
                               repeats=numpy.array(list(map(lambda x: len(x), flat_column))),
                               axis=0)
    flat_column = list(chain.from_iterable(flat_column))
    result = pandas.DataFrame(flat_values, columns=data.columns)
    result[new_column] = flat_column
    return result.infer_objects()


def add_context_info(data: pandas.DataFrame) -> pandas.DataFrame:
    """
    Split context of identifier on before and after part and return new dataframe with the info.

    :param data: DataFrame, containing column Columns.Token. \
                 Column Columns.Split will be used for creating context info.
    :return: New dataframe with added columns Columns.Before and Columns.After, \
             containing lists of corresponding contexts tokens.
    """
    result_data = data.copy()
    if Columns.Before in result_data.columns and Columns.After in result_data.columns:
        return result_data

    tokens = list(result_data[Columns.Token])
    token_split = list(result_data[Columns.Split])
    before = []
    after = []
    for row_number in range(len(result_data)):
        split_list = token_split[row_number].split()
        index = split_list.index(tokens[row_number])
        before.append(" ".join(split_list[:index]))
        after.append(" ".join(split_list[index + 1:]))

    result_data.loc[:, Columns.Before] = before
    result_data.loc[:, Columns.After] = after
    return result_data


def rank_candidates(candidates: pandas.DataFrame, pred_probs: List[float],
                    n_candidates: Optional[int] = None, return_all: bool = True,
                    ) -> Dict[int, List[Tuple[str, float]]]:
    """
    Rank candidates for tokens' correction based on the correctness probabilities.

    :param candidates: DataFrame with columns Columns.Id, Columns.Token, Columns.Candidate \
                       and indexed by range(len(pred_proba)).
    :param pred_probs: Array of probabilities of correctness of every candidate.
    :param n_candidates: Number of most probably correct candidates to return for each typo.
    :param return_all: False to return corrections only for tokens corrected in the \
                       first candidate.
    :return: Dictionary `{id : [(candidate, correctness_proba), ...]}`, candidates are sorted \
             by correct_prob in a descending order.
    """
    suggestions = {}
    corrections = []
    for i in range(len(pred_probs)):
        index = candidates.loc[i, Columns.Id]
        corrections.append((candidates.loc[i, Columns.Candidate], pred_probs[i]))
        if i < len(pred_probs) - 1 and candidates.loc[i + 1, Columns.Id] == index:
            continue

        corrections = list(sorted(corrections, key=lambda x: -x[1]))
        typo = candidates.loc[i, Columns.Token]
        if corrections[0][0] != typo:
            suggestions[index] = (corrections if n_candidates is None
                                  else corrections[:n_candidates])
        elif return_all:
            suggestions[index] = [(typo, 1.0)]
        corrections = []

    return suggestions


def suggestions_to_df(data: pandas.DataFrame, suggestions: Dict[int, List[Tuple[str, float]]],
                      ) -> pandas.DataFrame:
    """
    Convert suggestions from dictionary to pandas.DataFrame.

    :param data: DataFrame containing column Columns.Token.
    :param suggestions: Dictionary of suggestions, keys correspond with data.index.
    :return: DataFrame with columns Columns.Token, Columns.Suggestions, indexed by data.index.
    """
    suggestions_array = [[index, data.loc[index, Columns.Token], corrections]
                         for index, corrections in suggestions.items()]
    return pandas.DataFrame(suggestions_array,
                            columns=[Columns.Id, Columns.Token, Columns.Suggestions],
                            index=data.index).infer_objects()


def suggestions_to_flat_df(data: pandas.DataFrame,
                           suggestions: Dict[int, List[Tuple[str, float]]]) -> pandas.DataFrame:
    """
    Convert suggestions from dictionary to pandas.DataFrame, flattened by suggestions column.

    :param data: DataFrame containing column Columns.Token.
    :param suggestions: Dictionary of suggestions, keys correspond with data.index.
    :return: DataFrame with columns Columns.Token, Columns.Candidate, Columns.Probability,
             indexed by data.index.
    """
    flat_df = flatten_df_by_column(suggestions_to_df(data, suggestions),
                                   Columns.Suggestions, "suggestion")
    flat_df[Columns.Candidate] = [suggestion[0] for suggestion in flat_df.suggestion]
    flat_df[Columns.Probability] = [suggestion[1] for suggestion in flat_df.suggestion]
    flat_df = flat_df.drop(columns=[Columns.Suggestions, "suggestion"])
    return flat_df.infer_objects()
