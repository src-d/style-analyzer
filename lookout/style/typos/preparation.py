import multiprocessing
import os
import pathlib
import sys
import tempfile
from typing import Any, Mapping, Optional, Tuple
import urllib.request

import pandas
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from lookout.style.common import merge_dicts
from lookout.style.typos.corrector import TyposCorrector
from lookout.style.typos.corruption import corrupt_tokens_in_df
from lookout.style.typos.preprocessing import filter_splits, print_frequencies
from lookout.style.typos.utils import Columns, flatten_df_by_column


DATA_DIR = pathlib.Path(__file__).parent / "data"
DEFAULT_CONFIG = {
    "preparation": {
        "data_dir": str(DATA_DIR),
        "input_path": str(DATA_DIR / "raw_data.csv"),
        "dataset_url": "https://docs.google.com/uc?export=download&"
                       "id=1muNVWPe68XK8SFvqIv3V728NmkT46aTx",
        "frequency_column": "num_occ",
        "vocabulary_size": 10000,
        "frequencies_size": None,
        "raw_data_filename": "raw_data.csv",
        "vocabulary_filename": "vocabulary.csv",
        "frequencies_filename": "frequencies.csv",
        "prepared_filename": "prepared.csv",
    },
    "fasttext": {
        "size": 100000000,  # Number of identifiers to pick to train fasttext on
        "corrupt": True,  # Whether to corrupt some of the identifiers with artificial typos
        "typo_probability": 0.2,  # Which portion of picked identifiers contain a typoed token
        "add_typo_probability": 0.005,  # Which portion of corrupted tokens contain >1 mistake
        "path": str(DATA_DIR / "fasttext.bin"),  # Where to store trained fasttext model
        "dim": 8,  # Number of dimensions of embeddings
        "bucket": 200000,  # Number of hash buckets in the model
    },
    "datasets": {
        "train_size": 50000,
        "test_size": 10000,
        "typo_probability": 0.5,
        "add_typo_probability": 0.01,
        "train_path": str(DATA_DIR / "train.csv"),
        "test_path": str(DATA_DIR / "test.csv"),
    },
    "processes_number": multiprocessing.cpu_count(),
    "corrector_path": str(DATA_DIR / "corrector.asdf"),
}


class _DownloadProgressBar(tqdm):
    def update_to(self, b: int = 1, bsize: int = 1, tsize: Optional[int] = None) -> None:
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def _download_url(url: str, output_path: str) -> None:
    with _DownloadProgressBar(unit="MB", unit_scale=True,
                              miniters=1, desc=url.split("/")[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def prepare_data(config: Optional[Mapping[str, Any]] = None) -> pandas.DataFrame:
    """
    Generate all the necessary data from the raw dataset of split identifiers.

    Brief algorithm description:
    1. Derive vocabulary for typos correction which is a set of tokens, which is considered
       correctly spelled. All typos corrections will belong to the vocabulary.
       It is a set of most frequent tokens (based on given statistics).
    2. Save vocabulary and statistics for a given amount of most frequent tokens for future use.
    3. Filter raw data, leaving identifiers, containing only tokens from the vocabulary.
       The result is a dataset of tokens which will be considered correct. It will be used
       for creating artificial misspelling cases for training and testing the corrector model.
    4. Save prepared dataset, if needed.
    :param config: Dictionary with parameters for data preparation. Used fields are:
                   data_dir: Directory to put all derived data to.
                   drive_dataset_id: ID of google drive document, where a raw dataset is stored.
                   input_path: Path to a .csv dump of input dataframe. Should contain \
                               column Columns.Split. If None or file doesn't exist,
                               the dataset will be loaded from Google drive.
                   frequency_column: Name of the column with identifiers frequencies. If not \
                                     specified, every split is considered to have frequency 1.
                   vocabulary_size: Number of most frequent tokens to take as a vocabulary.
                   frequencies_size: Number of most frequent tokens to save frequencies info for. \
                                     This information will be used by corrector as features for \
                                     these tokens when they will be checked. If not specified, \
                                     frequencies for all present tokens will be saved.
                   raw_data_filename: Name of the .csv file in data_dir to put raw dataset \
                                      in case of loading from drive.
                   vocabulary_filename: Name of the .csv file in data_dir to save vocabulary to.
                   frequencies_filename: Name of the .csv file in data_dir to save frequencies to.
                   prepared_filename: Name of the .csv file in data_dir to save prepared \
                                      dataset to.
    :return: Dataset baked for training the typos correction.
    """
    if config is None:
        config = {}
    config = merge_dicts(DEFAULT_CONFIG["preparation"], config)

    os.makedirs(config["data_dir"], exist_ok=True)
    raw_data_path = config["input_path"]
    if raw_data_path is None or not os.path.exists(raw_data_path):
        raw_data_path = os.path.join(config["data_dir"],
                                     config["raw_data_filename"])
        _download_url(config["dataset_url"], raw_data_path)

    data = pandas.read_csv(raw_data_path, index_col=0, keep_default_na=False)
    if config["frequency_column"] not in data.columns:
        data[Columns.Frequency] = 1
    else:
        data = data.rename(columns={config["frequency_column"]: Columns.Frequency})

    # Expand dataframe by splits (repeat rows for every token in splits)
    data[Columns.Split] = data[Columns.Split].astype(str)
    flat_data = flatten_df_by_column(data, Columns.Split, Columns.Token,
                                     apply_function=lambda x: x.split())

    # Collect statistics for tokens
    stats = flat_data[[Columns.Frequency, Columns.Token]].groupby([Columns.Token]).sum()
    stats = stats.sort_values(by=Columns.Frequency, ascending=False)

    # Derive new vocabulary for future use
    frequencies_tokens = set(stats.index[:(config["frequencies_size"] or len(stats))])
    vocabulary_tokens = set(stats.index[:config["vocabulary_size"]])
    print_frequencies(vocabulary_tokens, stats, os.path.join(
        config["data_dir"], config["vocabulary_filename"]))
    print_frequencies(frequencies_tokens, stats, os.path.join(
        config["data_dir"], config["frequencies_filename"]))

    # Leave only splits that contain tokens from vocabulary
    prepared_data = filter_splits(flat_data, vocabulary_tokens)[[Columns.Frequency, Columns.Split,
                                                                 Columns.Token]]
    prepared_data.reset_index(drop=True, inplace=True)
    if config["prepared_filename"] is not None:
        prepared_data.to_csv(os.path.join(config["data_dir"], config["prepared_filename"]))
    return prepared_data


def train_fasttext(data: pandas.DataFrame, config: Optional[Mapping[str, Any]] = None) -> None:
    """
    Train fasttext model on the given dataset of code identifiers.

    :param data: Dataframe with columns Columns.Split and Columns.Frequency.
    :param config: Parameters for training the model, options:
                   size: Number of identifiers to pick from the given data to train fasttext on.
                   corrupt: Value indicating whether to make random artificial typos in \
                            the training data. Identifiers are corrupted with `typo_probability`.
                   typo_probability: Token corruption probability if `corrupt == True`.
                   add_typo_probability: Probability of second corruption in a corrupted token. \
                                         used if `corrupt == True`.
                   path: Path where to store the trained fasttext model.
                   dim: Number of dimensions for embeddings in the new model.
                   bucket: Number of hash buckets to keep in the fasttext model: \
                           the less there are, the more compact the model gets.
    """
    try:
        import fastText
    except ImportError:
        sys.exit("Please install fastText."
                 "Run `pip3 install git+https://github.com/facebookresearch/fastText"
                 "@51e6738d734286251b6ad02e4fdbbcfe5b679382`")
    if config is None:
        config = {}
    config = merge_dicts(DEFAULT_CONFIG["fasttext"], config)
    train_data = data[[len(str(x).split()) > 2 for x in data[Columns.Split]]].sample(
        config["size"], weights=Columns.Frequency, replace=True)
    if config["corrupt"]:
        train_data = corrupt_tokens_in_df(train_data, config["typo_probability"],
                                          config["add_typo_probability"])
    with tempfile.NamedTemporaryFile() as ids_file:
        with open(ids_file.name, "w") as f:
            for token_split in train_data[Columns.Split]:
                f.write(token_split + "\n")
        model = fastText.train_unsupervised(ids_file.name, minCount=1, epoch=10,
                                            dim=config["dim"],
                                            bucket=config["bucket"])
    model.save_model(config["path"])


def get_datasets(prepared_data: pandas.DataFrame,
                 config: Optional[Mapping[str, Any]] = None,
                 processes_number: int = DEFAULT_CONFIG["processes_number"],
                 ) -> Tuple[pandas.DataFrame, pandas.DataFrame]:
    """
    Create the train and the test datasets of typos.

    1. Take the specified number of lines from the input dataset.
    2. Make artificial typos in picked identifiers and split them into train and test.
    3. Return results.
    :param prepared_data: Dataframe of correct splitted identifiers. Must contain columns \
                          Columns.Split, Columns.Frequency and Columns.Token.
    :param config: Parameters for creating train and test datasets, options:
                   train_size: Train dataset size.
                   test_size: Test dataset size.
                   typo_probability: Probability of token corruption.
                   add_typo_probability: Probability of second corruption for a corrupted token.
                   train_path: Path to the .csv file where to save the train dataset.
                   test_path: Path to the .csv file where to save the test dataset.
    :param processes_number: Number of processes for multiprocessing.
    :return: Train and test datasets.
    """
    if config is None:
        config = {}
    config = merge_dicts(DEFAULT_CONFIG["datasets"], config)
    # With replace=True we get the real examples distribution, but there's a small
    # probability of having the same examples of misspellings in train and test datasets
    # (it IS small because a big number of random typos can be made in a single word)
    data = prepared_data[[len(x) > 1 for x in prepared_data[Columns.Token]]].sample(
        config["train_size"] + config["test_size"], weights=Columns.Frequency,
        replace=True)
    train, test = train_test_split(data[[Columns.Token, Columns.Split]],
                                   test_size=config["test_size"])
    train.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)
    train = corrupt_tokens_in_df(train, config["typo_probability"], config["add_typo_probability"],
                                 processes_number)
    test = corrupt_tokens_in_df(test, config["typo_probability"], config["add_typo_probability"],
                                processes_number)
    if config["test_path"] is not None:
        test.to_csv(config["test_path"])
    if config["train_path"] is not None:
        train.to_csv(config["train_path"])
    return train, test


def train_and_evaluate(train_data: pandas.DataFrame, test_data: pandas.DataFrame,
                       vocabulary_path: str, frequencies_path: str, fasttext_path: str,
                       processes_number: int = DEFAULT_CONFIG["processes_number"],
                       ) -> TyposCorrector:
    """
    Create and train TyposCorrector model on the given data.

    :param train_data: Dataframe which contains columns Columns.Token, Columns.Split and \
                       Columns.CorrectToken.
    :param test_data: Dataframe which contains columns Columns.Token, Columns.Split and \
                      Columns.CorrectToken.
    :param vocabulary_path: Path to a file with vocabulary.
    :param frequencies_path: Path to a file with tokens' frequencies.
    :param fasttext_path: Path to a FastText model dump.
    :param processes_number: Number of processes for multiprocessing.
    :return: Trained model.
    """
    model = TyposCorrector()
    model.initialize_ranker()
    model.initialize_generator(vocabulary_file=vocabulary_path,
                               frequencies_file=frequencies_path,
                               embeddings_file=fasttext_path)
    model.processes_number = processes_number
    model.train(train_data)
    model.evaluate(test_data)
    return model


def train_from_scratch(config: Optional[Mapping[str, Any]] = None) -> TyposCorrector:
    """
    Train TyposCorrector on raw data.

    1. Prepare data, for more info check :func:`prepare_data`.
    2. Construct train and test datasets, for more info check :func:`get_train_test`.
    3. Train and evaluate TyposCorrector model, for more info check :func:`train_and_evaluate`.
    4. Return result.
    :param config: Parameters for data preparation and corrector training.
    :return: Trained TyposCorrector model.
    """
    if config is None:
        config = {}
    config = merge_dicts(DEFAULT_CONFIG, config)
    prepared_data = prepare_data(config["preparation"])
    if config["fasttext"]["path"] is None or not os.path.exists(config["fasttext"]["path"]):
        train_fasttext(prepared_data, config["fasttext"])
    train_data, test_data = get_datasets(prepared_data, config["datasets"])
    model = train_and_evaluate(train_data, test_data,
                               os.path.join(config["preparation"]["data_dir"],
                                            config["preparation"]["vocabulary_filename"]),
                               os.path.join(config["preparation"]["data_dir"],
                                            config["preparation"]["frequencies_filename"]),
                               config["fasttext"]["path"], config["processes_number"])
    if config["corrector_path"] is not None:
        model.save(config["corrector_path"], series=0.0)
    return model
