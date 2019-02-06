from os import path
import random
import string
from typing import Tuple

import pandas
from tqdm import tqdm

from lookout.style.typos.preprocessing.dev_utils import rand_bool
from lookout.style.typos.utils import COLUMNS

letters = list(string.ascii_lowercase)


def rand_insert(token: str):
    """
    Add random letter inside a token
    """
    letter = random.choice(letters)
    if len(token) == 0:
        return letter

    pos = random.choice(range(len(token) + 1))
    if pos == len(token):
        return token + letter
    return token[:pos] + letter + token[pos:]


def rand_delete(token: str):
    """
    Delete random symbol from a token
    """
    if len(token) == 0:
        return token
    pos = random.choice(range(len(token)))
    return token[:pos] + token[pos + 1:]


def rand_substitution(token: str):
    """
    Substitute random symbol with a letter inside a token
    """
    if len(token) == 0:
        return token
    pos = random.choice(range(len(token)))
    letter = random.choice(letters)
    return token[:pos] + letter + token[pos + 1:]


def rand_swap(token: str):
    """
    Swap two random consequent symbols
    """
    if len(token) < 2:
        return token
    pos = random.choice(range(len(token) - 1))
    return token[:pos] + token[pos + 1] + token[pos] + token[pos + 2:]


def rand_typo(token: str):
    """
    Make random typo in a token
    """
    typo_func = random.choice([rand_insert, rand_delete, rand_substitution, rand_swap])
    return typo_func(token)


def corrupt_tokens_in_df(data: pandas.DataFrame, typo_probability: float,
                         add_typo_probability: float) -> pandas.DataFrame:
    """
    Augment some of identifiers from dataframe with TYPO_PROBABILITY,
    consequent typos in the same word happen with ADD_TYPO_PROBABILITY each.
    Operations happens inplace.
    """
    tokens = list(data[COLUMNS["TOKEN"]])
    if COLUMNS["SPLIT"] in data.columns:
        token_split = list(data[COLUMNS["SPLIT"]])

    corrupted = []
    for row_number in tqdm(range(len(data))):
        if tokens[row_number] is not None:
            item = tokens[row_number]
            if len(item) > 1 and rand_bool(typo_probability):
                item = ''
                while len(item) < 2:
                    item = tokens[row_number]
                    item = rand_typo(str(item))
                    while rand_bool(add_typo_probability):
                        item = rand_typo(item)
                corrupted.append(True)
            else:
                corrupted.append(False)

            if COLUMNS["SPLIT"] in data.columns:
                split = str(token_split[row_number]).split()
                index = split.index(tokens[row_number])
                split[index] = item
                token_split[row_number] = " ".join(split)
            tokens[row_number] = item

    data[COLUMNS["TOKEN"]] = tokens
    if COLUMNS["SPLIT"] in data.columns:
        data[COLUMNS["CORRECT_SPLIT"]] = data[COLUMNS["SPLIT"]]
        data[COLUMNS["SPLIT"]] = token_split
    return data


def corrupt_tokens_in_file(data_file: str, typo_probability: float, add_typo_probability: float,
                           out_file: str, repeats: int = 1) -> None:
    with open(data_file, "r") as f:
        with open(out_file, "w") as out:
            for _ in range(repeats):
                for line in f:
                    tokens = []
                    for token in line.split():
                        item = token
                        if rand_bool(typo_probability):
                            item = ''
                            while len(item) == 0:
                                item = token
                                item = rand_typo(str(item))
                                while rand_bool(add_typo_probability):
                                    item = rand_typo(item)
                        tokens.append(item)
                    print(" ".join(tokens), file=out)


def train_test_split(data: pandas.DataFrame, test_portion: float
                     ) -> Tuple[pandas.DataFrame, pandas.DataFrame]:
    """
    Randomly split data on train and test
    """
    test = set(random.sample(range(len(data)), int(data.shape[0] * test_portion)))
    test_indices = [i in test for i in range(len(data))]
    train_indices = [i not in test for i in range(len(data))]
    return data[train_indices], data[test_indices]


def create_typos(args):
    data = pandas.read_pickle(args.input_file)
    corrupt_tokens_in_df(data, args.typo_probability, args.add_typo_probability
                         ).to_csv(args.out_file)
    if args.test_portion is not None:
        train, test = train_test_split(data, args.test_portion)
        path_split = path.split(args.out_file)
        train.to_csv(path.join(path_split[0], "train_" + path_split[1]))
        test.to_csv(path.join(path_split[0], "test_" + path_split[1]))
