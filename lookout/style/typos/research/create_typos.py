import random
import string

import pandas
from tqdm import tqdm

from lookout.style.typos.research.dev_utils import rand_bool

letters = list(string.ascii_lowercase)


def rand_insert(string: str):
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


def rand_delete(string: str):
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


def rand_swap(string):
    """
    Swap two random consequent symbols
    """
    if len(string) < 2:
        return string
    pos = random.choice(range(len(string) - 1))
    return string[:pos] + string[pos + 1] + string[pos] + string[pos + 2:]


def rand_typo(string):
    """
    Make random typo in a string
    """
    typo_func = random.choice([rand_insert, rand_delete, rand_substitution, rand_swap])
    return typo_func(string)


def corrupt(data_file, typo_probability, add_typo_probability, out_file):
    """
    Augment some of identifiers from dataframe with TYPO_PROBABILITY,
    consequent typos in the same word happen with ADD_TYPO_PROBABILITY each
    """
    data = pandas.read_pickle(data_file)
    tokens = list(data.identifier)
    if "token_split" in data.columns:
        token_split = list(data.token_split)

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

            if "token_split" in data.columns:
                split = str(token_split[row_number]).split()
                index = split.index(tokens[row_number])
                split[index] = item
                token_split[row_number] = " ".join(split)
            tokens[row_number] = item

    data["typo"] = tokens
    data["corrupted"] = corrupted
    if "token_split" in data.columns:
        data["id_split"] = data["token_split"]
        data["token_split"] = token_split

    data.to_csv(out_file)


def corrupt_splits(data_file, typo_probability, add_typo_probability, out_file, repeats: int = 1):
    with open(data_file) as f:
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


def train_test_split(data_file, test_portion):
    """
    Randomly split data on train and test
    """
    data = pandas.read_csv(data_file, index_col=0)
    test = set(random.sample(range(len(data)), int(data.shape[0] * test_portion)))
    test_indices = [i in test for i in range(len(data))]
    train_indices = [i not in test for i in range(len(data))]

    data[train_indices].to_csv("train_" + data_file)
    data[test_indices].to_csv("test_" + data_file)


def create_typos(args):
    corrupt(args.input_file, args.typo_probability,
            args.add_typo_probability, args.out_file)
    if args.test_portion is not None:
        train_test_split(args.out_file, args.test_portion)
