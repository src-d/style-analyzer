import random
import string

import pandas
from tqdm import tqdm

from typos_functions import rand_bool


letters = list(string.ascii_lowercase)


def rand_insert(string):
    """
    Add random letter inside a string
    """
    letter = random.choice(letters)
    if len(string) == 0:
        return letter

    pos = random.choice(range(len(string) + 1))
    if pos == len(string):
        return string + letter
    return string[:pos] + letter + string[pos:]


def rand_delete(string):
    """
    Delete random symbol from a string
    """
    if len(string) == 0:
        return string
    pos = random.choice(range(len(string)))
    return string[:pos] + string[pos + 1:]


def rand_substitution(string):
    """
    Substitute random symbol with a letter inside a string
    """
    if len(string) == 0:
        return string
    pos = random.choice(range(len(string)))
    letter = random.choice(letters)
    return string[:pos] + letter + string[pos + 1:]


def rand_typo(string):
    """
    Make random typo in a string
    """
    typo_func = random.choice([rand_insert, rand_delete, rand_substitution])
    return typo_func(string)


def corrupt(data_file, typo_probability, add_typo_probability, out_file):
    """
    Augment some of identifiers from dataframe with TYPO_PROBABILITY,
    another typos in the same word happen with ADD_TYPO_PROBABILITY each
    """
    data = pandas.read_pickle(data_file)
    tokens = list(data.identifier)
    corrupted = []
    for row_number in tqdm(range(len(data))):
        if rand_bool(typo_probability):
            item = tokens[row_number]
            item = rand_typo(str(item))
            while rand_bool(add_typo_probability):
                item = rand_typo(item)

            tokens[row_number] = item
            corrupted.append(True)
        else:
            corrupted.append(False)
    data["typo"] = tokens
    data["corrupted"] = corrupted
    data.to_pickle(out_file)


def train_test_split(data_file, test_portion):
    """
    Split data on train and test chunks without mixing rows
    """
    data = pandas.read_pickle(data_file)
    edge = data.index[int(data.shape[0] * (1 - test_portion))]
    data.loc[:edge, :].to_pickle("train_" + data_file)
    data.loc[edge:, :].to_pickle("test_" + data_file)


def create_typos(args):
    corrupt(args.input_file, args.typo_probability,
            args.add_typo_probability, args.out_file)
    if args.test_portion is not None:
        train_test_split(args.out_file, args.test_portion)
