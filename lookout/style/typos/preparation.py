from copy import deepcopy
import os
import pathlib
import tempfile
from typing import Any, Mapping, Optional, Tuple
import urllib.request

import fastText
import pandas
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from lookout.style.common import merge_dicts
from lookout.style.typos.corrector import TyposCorrector
from lookout.style.typos.corruption import corrupt_tokens_in_df
from lookout.style.typos.preprocessing import filter_splits, print_frequencies
from lookout.style.typos.utils import Columns, flatten_df_by_column


DATA_DIR = pathlib.Path(__file__).parent / "data"

defaults_for_fasttext = {
    "size": 100000000,
    "corrupt": True,
    "typo_probability": 0.2,
    "add_typo_probability": 0.005,
    "fasttext_path": str(DATA_DIR / "emb.bin"),
    "dim": 8,
    "bucket": 200000,
}

defaults_for_preparation = {
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


def prepare_data(params: Optional[Mapping[str, Any]] = None) -> pandas.DataFrame:
    """
    Generate all the necessary data from the raw dataset of split identifiers.

    Brief algorithm description:
    1. Derive vocabulary for typos correction which is a set of tokens, which is considered
       correctly spelled. All typos corrections will belong to the vocabulary.
       It is a set of most frequent tokens (based on given statistics).
    2. Save vocabulary and statistics for given amount of most frequent tokens for future use.
    3. Filter raw data, leaving identifiers, containing only tokens from the vocabulary.
       The result is a dataset of tokens which will be considered correct. It will be used
       for creating artificial misspelling cases for training and testing the corrector model.
    4. Save prepared dataset, if needed.
    :param params: Dictionary with parameters for data preparation. Used fields are:
                   data_dir: Directory to put all derived data to.
                   drive_dataset_id: ID of google drive document, where raw dataset is stored.
                   input_path: Path to a .csv dump of input dataframe. Should contain \
                               column Columns.Split. If None or file doesn't exist,
                               the dataset will be loaded from drive.
                   frequency_column: Name of column with identifiers frequencies. If not \
                                     specified, every split is considered to have frequency 1.
                   vocabulary_size: Number of most frequent tokens to take as a vocabulary.
                   frequencies_size: Number of most frequent tokens to save  frequencies info for.\
                                     This information will be used by corrector as features for \
                                     these tokens when they will be checked. If not specified, \
                                     frequencies for all present tokens will be saved.
                   raw_data_filename: Name of .csv file in data_dir to put raw dataset in case of \
                                      loading from drive.
                   vocabulary_path: Name of .csv file in data_dir to save vocabulary to.
                   frequencies_path: Name of .csv file in data_dir to save frequencies to.
    :return: Dataset baked for training the typos correction.
    """
    if params is None:
        params = deepcopy(defaults_for_preparation)
    else:
        params = merge_dicts(defaults_for_preparation, params)

    raw_data_path = params["input_path"]
    if raw_data_path is None or not os.path.exists(raw_data_path):
        raw_data_path = os.path.join(params["data_dir"], params["raw_data_filename"])
        _download_url(params["dataset_url"], raw_data_path)

    data = pandas.read_csv(raw_data_path, index_col=0)
    if params["frequency_column"] not in data.columns:
        data[Columns.Frequency] = 1
    else:
        data = data.rename(columns={params["frequency_column"]: Columns.Frequency})

    # Expand dataframe by splits (repeat rows for every token in splits)
    data[Columns.Split] = data[Columns.Split].astype(str)
    flat_data = flatten_df_by_column(data, Columns.Split, Columns.Token,
                                     apply_function=lambda x: x.split())

    # Collect statistics for tokens
    stats = flat_data[[Columns.Frequency, Columns.Token]].groupby([Columns.Token]).sum()
    stats = stats.sort_values(by=Columns.Frequency, ascending=False)

    # Derive new vocabulary for future use
    frequencies_tokens = set(stats.index[:(params["frequencies_size"] or len(stats))])
    vocabulary_tokens = set(stats.index[:params["vocabulary_size"]])
    print_frequencies(vocabulary_tokens, stats, os.path.join(params["data_dir"],
                                                             params["vocabulary_filename"]))
    print_frequencies(frequencies_tokens, stats, os.path.join(params["data_dir"],
                                                              params["frequencies_filename"]))

    # Leave only splits that contain tokens from vocabulary
    prepared_data = filter_splits(flat_data, vocabulary_tokens)[[Columns.Frequency, Columns.Split,
                                                                 Columns.Token]]
    return prepared_data


def get_datasets(prepared_data: pandas.DataFrame, train_size: int = 50000,
                 test_size: int = 10000, typo_probability: float = 0.5,
                 add_typo_probability: float = 0.01) -> Tuple[pandas.DataFrame, pandas.DataFrame]:
    """
    Create the train and the test datasets of typos.

    1. Take the specified number of lines from the input dataset.
    2. Make artificial typos in picked identifiers and split them into train and test.
    3. Return results.
    :param prepared_data: Dataframe of correct splitted identifiers. Must contain columns \
                          Columns.Split, Columns.Frequency and Columns.Token.
    :param train_size: Train dataset size.
    :param test_size: Test dataset size.
    :param typo_probability: Probability with which a token gets to be corrupted.
    :param add_typo_probability: Probability with which one more corruption happens to a \
                                 corrupted token.
    :return: Train and test datasets.
    """
    # With replace=True we get the real examples distribution, but there's a small
    # probability of having the same examples of misspellings in train and test datasets
    # (it IS small because a big number of random typos can be made in a single word)
    data = prepared_data.sample(train_size + test_size, weights=Columns.Frequency, replace=True)
    data = corrupt_tokens_in_df(data, typo_probability, add_typo_probability)
    return train_test_split(data, test_size=test_size)


def train_fasttext(data: pandas.DataFrame, params: Optional[Mapping[str, Any]] = None) -> None:
    """
    Train fasttext model on the given dataset of code identifiers.

    :param data: Dataframe with columns Columns.Split and Columns.Frequency.
    :param params: Parameters for training the model, options:
                   size: Number of identifiers to pick from the given data to train fasttext on.
                   corrupt: Value indicating whether to make random artificial typos in \
                            the training data. Identifiers are corrupted with `typo_probability`.
                   typo_probability: Probability with which a token is corrupted, used \
                                     if `corrupt=True`.
                   add_typo_probability: Probability with which another corruption happens in a \
                                  corrupted token, used if `corrupt=True`.
                   fasttext_path: Path where to store the trained fasttext model.
                   dim: Number of dimensions for embeddings in the new model.
                   bucket: Number of hash buckets to keep in the fasttext model: \
                           the less there are, the more compact the model gets.
    """
    if params is None:
        params = {}
    params = merge_dicts(defaults_for_fasttext, params)
    train_data = data.sample(params["size"], weights=Columns.Frequency, replace=True)
    if params["corrupt"]:
        train_data = corrupt_tokens_in_df(train_data, params["typo_probability"],
                                          params["add_typo_probability"])
    with tempfile.NamedTemporaryFile() as ids_file:
        with open(ids_file.name, "w") as f:
            for token_split in train_data[Columns.Split]:
                f.write(token_split + "\n")
        model = fastText.train_unsupervised(ids_file.name, minCount=1, epoch=10,
                                            dim=params["dim"], bucket=params["bucket"])
    model.save_model(params["fasttext_path"])


def train_and_evaluate(train_data: pandas.DataFrame, test_data: pandas.DataFrame,
                       vocabulary_path: str, frequencies_path: str, fasttext_path: str,
                       threads_number: int = 8) -> TyposCorrector:
    """
    Create and train TyposCorrector model on the given data.

    :param train_data: Dataframe which contains columns Columns.Token, Columns.Split and \
                       Columns.CorrectToken.
    :param test_data: Dataframe which contains columns Columns.Token, Columns.Split and \
                      Columns.CorrectToken.
    :param vocabulary_path: Path to a file with vocabulary.
    :param frequencies_path: Path to a file with tokens' frequencies.
    :param fasttext_path: Path to a FastText model dump.
    :param threads_number: Number of threads for multiprocessing.
    :return: Trained model.
    """
    model = TyposCorrector()
    model.initialize_ranker()
    model.initialize_generator(vocabulary_file=vocabulary_path,
                               frequencies_file=frequencies_path,
                               embeddings_file=fasttext_path)
    model.threads_number = threads_number
    model.train(train_data)
    model.evaluate(test_data)
    return model
