from functools import partial
import logging
from multiprocessing import Pool
import random
import string
from typing import NamedTuple, Tuple

import pandas
from tqdm import tqdm

Columns = NamedTuple(
    "Columns",
    [("Token", str), ("CorrectToken", str), ("Split", str), ("CorrectSplit", str), ("After", str),
     ("Before", str), ("Id", str), ("Candidate", str), ("Features", str), ("Probability", str),
     ("Suggestions", str), ("Frequency", str)])(
    "token", "correct_token", "token_split", "correct_token_split", "after", "before", "id",
    "candidate", "features", "proba", "suggestions", "freq")

letters = list(string.ascii_lowercase)


def rand_insert(token: str) -> str:
    """
    Add a random letter inside the token.
    """
    letter = random.choice(letters)
    if len(token) == 0:
        return letter
    pos = random.choice(range(len(token) + 1))
    if pos == len(token):
        return token + letter
    return token[:pos] + letter + token[pos:]


def rand_delete(token: str) -> str:
    """
    Delete a random symbol from the token.
    """
    if len(token) == 0:
        return token
    pos = random.choice(range(len(token)))
    return token[:pos] + token[pos + 1:]


def rand_substitution(token: str) -> str:
    """
    Substitute a random symbol with a letter inside the token.
    """
    if len(token) == 0:
        return token
    pos = random.choice(range(len(token)))
    return token[:pos] + random.choice([c for c in letters if c != token[pos]]) + token[pos + 1:]


def rand_swap(token: str) -> str:
    """
    Swap two random consequent symbols inside the token.
    """
    if len(token) < 2 or len(set(token)) == 1:
        return token
    pos = random.choice(range(len(token) - 1))
    while token[pos] == token[pos + 1]:
        pos = random.choice(range(len(token) - 1))
    return token[:pos] + token[pos + 1] + token[pos] + token[pos + 2:]


def _rand_typo(token_split: Tuple[str, str], typo_prob: float,
               add_typo_prob: float) -> Tuple[str, str]:
    token, split = token_split
    typoed_token = token
    if len(token) > 1 and random.uniform(0, 1) < typo_prob:
        typoed_token = ""
        while len(typoed_token) < 2:
            typoed_token = random.choice([rand_insert, rand_delete, rand_substitution,
                                          rand_swap])(token)
            while random.uniform(0, 1) < add_typo_prob:
                typoed_token = random.choice([rand_insert, rand_delete, rand_substitution,
                                              rand_swap])(typoed_token)
    if typoed_token != token:
        split = split.replace(token, typoed_token)
    return typoed_token, split


def corrupt_tokens_in_df(data: pandas.DataFrame, typo_prob: float,
                         add_typo_prob: float, threads_number: int = 16,
                         log_level: int = logging.DEBUG,
                         ) -> pandas.DataFrame:
    """
    Create artificial typos in tokens (identifiers) in a pandas DataFrame. \
    Augment some of the identifiers from the dataframe with `typo_prob`, \
    the consequent typos in the same word happen with `add_typo_prob` each. \
    Operations run out-of-place.

    :param data: Dataframe which contains columns Columns.Token and Columns.Split.
    :param typo_prob: Probability with which a token gets to be corrupted.
    :param add_typo_prob: Probability with which one more corruption happens to a \
                                 corrupted token.
    :param threads_number: Number of threads for multiprocessing
    :param log_level: Level of logging.
    :return: New dataframe with added columns Columns.CorrectToken and Columns.CorrectSplit, \
             which contain tokens and corresponding splits from the `data`. Columns.Token and \
             Columns.Split now contain partially corrupted tokens and corresponding splits.
    """
    tokens_splits = list(zip(data[Columns.Token].astype(str), data[Columns.Split].astype(str)))

    def _wrap(x):
        if log_level == logging.DEBUG:
            return tqdm(x, total=len(tokens_splits))
        else:
            return x

    with Pool(threads_number) as pool:
        typoed_tokens_splits = list(_wrap(pool.imap(
            partial(_rand_typo,
                    typo_prob=typo_prob,
                    add_typo_prob=add_typo_prob), tokens_splits,
            chunksize=min(8192, 1 + len(tokens_splits) // threads_number))))

    result = data.copy()
    result[Columns.Token] = [x[0] for x in typoed_tokens_splits]
    result[Columns.Split] = [x[1] for x in typoed_tokens_splits]
    result[Columns.CorrectToken] = data[Columns.Token]
    result[Columns.CorrectSplit] = data[Columns.Split]
    return result
