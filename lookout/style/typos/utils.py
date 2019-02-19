"""Various glue functions to work with the input dataset and the output from FastText."""

from itertools import chain
import lzma
from typing import Dict, Iterable, List, NamedTuple, Tuple

from gensim.models.fasttext import FastText
import numpy
import pandas

Columns = NamedTuple(
    "Columns",
    [("Token", str), ("CorrectToken", str), ("Split", str), ("CorrectSplit", str), ("After", str),
     ("Before", str), ("Id", str), ("Candidate", str), ("Features", str), ("Probability", str),
     ("Suggestions", str)])(
    "token", "correct_token", "token_split", "correct_token_split", "after", "before", "id",
    "candidate", "features", "proba", "suggestions")


def extract_embeddings_from_fasttext(fasttext: FastText, tokens: Iterable[str]) -> numpy.ndarray:
    """
    Convert the embeddings from FastText to a dense matrix.

    :param fasttext: trained embeddings.
    :param tokens: list of tokens - axis Y of the returned matrix.
    :return: matrix with extracted embeddings.
    """
    return numpy.array([fasttext.wv[token] for token in tokens])


def read_frequencies(file: str) -> Dict[str, int]:
    """
    Read token frequencies from the file.

    :param file: Path to the file containing tokens with frequencies.
                 Every line has format "token count".
    :return: Dictionary of tokens frequencies.
    """
    frequencies = {}
    with lzma.open(file, "rt") as f:
        for line in f:
            split = line.split()
            frequencies[split[0]] = int(split[1])
    return frequencies


def read_vocabulary(file: str) -> List[str]:
    """
    Read vocabulary tokens from the text file.

    :param file: Text file used to generate vocabulary of corrections candidates.
                 First token in every line split is added to the vocabulary.
    :return: List of tokens of the vocabulary.
    """
    with lzma.open(file, "rt") as f:
        tokens = [line.split()[0] for line in f]
    return tokens


def flatten_df_by_column(data: pandas.DataFrame, column: str, new_column: str,
                         apply_function=lambda x: x) -> pandas.DataFrame:
    """
    Flatten DataFrame by `column` with extracted elements put to `new_column`. \
    Operation happens out-of-place.

    :param data: DataFrame to flatten
    :param column: Column to expand
    :param new_column: Column to populate with elements from flattened column
    :param apply_function: Function used to expand every element of flattened column
    :return: Flattened DataFrame
    """
    flat_column = data[column].apply(apply_function).tolist()
    flat_values = numpy.repeat(data.values,
                               repeats=numpy.array(list(map(lambda x: len(x), flat_column))),
                               axis=0)
    flat_column = list(chain.from_iterable(flat_column))
    result = pandas.DataFrame(flat_values, columns=data.columns)
    result[new_column] = flat_column
    return result.infer_objects()


def flatten_data(data: pandas.DataFrame, new_column_name=Columns.Token) -> pandas.DataFrame:
    """
    Flatten identifiers data in column `new_column_name`. \
    Operation happens out-of-place.

    :param data: DataFrame containing column `new_column_name` with splitted identifiers
                 either as strings or as lists of tokens.
    :param new_column_name: Name of column to put tokens from splits to.
    :return: Flattened DataFrame.
    """
    apply_function = (lambda x: x) if isinstance(data[Columns.Split].tolist()[0], list) \
        else (lambda x: str(x).split())
    return flatten_df_by_column(data, Columns.Split, new_column_name,
                                apply_function=apply_function)


def add_context_info(data: pandas.DataFrame) -> pandas.DataFrame:
    """
    Split context of identifier on before and after part.

    :param data: DataFrame. Column Columns.Split will be used for
                 creating context info if present.
    :return: Provided data with added columns Columns.Before and Columns.After, containing lists
             of corresponding contexts tokens.
    """
    if Columns.Before in data.columns and Columns.After in data.columns:
        return data

    tokens = list(data[Columns.Token])

    if Columns.Split in data.columns:
        token_split = list(data[Columns.Split])
        before = []
        after = []

        for row_number in range(len(data)):
            if Columns.Split in data.columns:
                split = token_split[row_number]
                if isinstance(split, str):
                    split = split.split()
                index = split.index(tokens[row_number])
                before.append(split[:index])
                after.append(split[index + 1:])

    else:
        before = [[] for _ in range(len(data))]
        after = [[] for _ in range(len(data))]

    data[Columns.Before] = before
    data[Columns.After] = after

    return data.infer_objects()


def rank_candidates(candidates: pandas.DataFrame, pred_probs: List[float],
                    n_candidates: int = None, return_all: bool = True,
                    ) -> Dict[int, List[Tuple[str, float]]]:
    """
    Rank candidates for typos correction based on the correctness probabilities.

    :param candidates: DataFrame with columns Columns.Id, Columns.Token, "candidate"
                       and indexed by range(len(pred_proba)).
    :param pred_probs: Array of probabilities of correctness of every candidate.
    :param n_candidates: Number of most probably correct candidates to return for each typo.
    :param return_all: False to return corrections only for typos corrected in the
                       first candidate.
    :return: Dictionary {id : [[candidate, correct_prob]]}, candidates are sorted
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


def filter_suggestions(typos: pandas.DataFrame, suggestions: Dict[int, List[Tuple[str, float]]],
                       n_candidates: int=1, return_all: bool=False,
                       ) -> Dict[int, List[Tuple[str, float]]]:
    """
    Filter correction suggestions.

    :param typos: DataFrame, containing column Columns.Token.
    :param suggestions: Dictionary of suggestions, keys correspond with typos.index.
    :param n_candidates: Number of most probably correct candidates to return for each typo.
    :param return_all: False to return corrections only for typos corrected in the
                       first candidate.
    :return: Dictionary of filtered suggestions.
    """
    return {index: suggestions[index][:n_candidates] for index, row in typos.iterrows()
            if return_all or suggestions[index][0][0] != row[Columns.Token]}


def suggestions_to_df(typos: pandas.DataFrame, suggestions: Dict[int, List[Tuple[str, float]]],
                      ) -> pandas.DataFrame:
    """
    Convert suggestions from dictionary to pandas.DataFrame.

    :param typos: DataFrame containing column Columns.Token.
    :param suggestions: Dictionary of suggestions, keys correspond with typos.index.
    :return: DataFrame with columns Columns.Token, Columns.Suggestions, indexed by typos.index.
    """
    suggestions_array = [[index, typos.loc[index, Columns.Token], corrections]
                         for index, corrections in suggestions.items()]
    return pandas.DataFrame(suggestions_array,
                            columns=[Columns.Id, Columns.Token, Columns.Suggestions],
                            index=typos.index).infer_objects()


def suggestions_to_flat_df(typos: pandas.DataFrame,
                           suggestions: Dict[int, List[Tuple[str, float]]]) -> pandas.DataFrame:
    """
    Convert suggestions from dictionary to pandas.DataFrame, flattened by suggestions column.

    :param typos: DataFrame containing column Columns.Token.
    :param suggestions: Dictionary of suggestions, keys correspond with typos.index.
    :return: DataFrame with columns Columns.Token, Columns.Candidate, Columns.Probability,
             indexed by typos.index.
    """
    flat_df = flatten_df_by_column(suggestions_to_df(typos, suggestions),
                                   Columns.Suggestions, "suggestion")
    flat_df[Columns.Candidate] = [suggestion[0] for suggestion in flat_df.suggestion]
    flat_df[Columns.Probability] = [suggestion[1] for suggestion in flat_df.suggestion]
    flat_df = flat_df.drop(columns=[Columns.Suggestions, "suggestion"])
    return flat_df.infer_objects()
