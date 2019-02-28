from copy import deepcopy
import os
import pathlib
from typing import Any, Mapping, Optional
import urllib.request

import pandas
from tqdm import tqdm

from lookout.style.common import merge_dicts
from lookout.style.typos.preprocessing import filter_splits, print_frequencies
from lookout.style.typos.utils import Columns, flatten_df_by_column


defaults_for_preparation = {
    "data_dir": str(pathlib.Path(__file__).parent / "data"),
    "input_path": str(pathlib.Path(__file__).parent / "data" / "raw_data.csv"),
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
