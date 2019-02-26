from typing import Any, Mapping, Optional

from google_drive_downloader import GoogleDriveDownloader as gdd
import pandas

from lookout.style.common import merge_dicts
from lookout.style.typos.preprocessing import filter_splits, print_frequencies
from lookout.style.typos.utils import Columns, flatten_df_by_column


DRIVE_DATASET_ID = "1muNVWPe68XK8SFvqIv3V728NmkT46aTx"
DRIVE_FASTTEXT_ID = "1hCOIwKn-QZLVv1S385HxyNMeERgKGIvo"


defaults_for_preparation = {
    "input_path": None,
    "frequency_column": "num_occ",
    "vocabulary_size": 10000,
    "frequencies_size": None,
    "vocabulary_path": "lookout/style/typos/data/vocabulary.csv",
    "frequencies_path": "lookout/style/typos/data/frequencies.csv",
}


def prepare_data(params: Optional[Mapping[str, Any]] = None) -> pandas.DataFrame:
    """
    Get all necessary data from raw dataset of splitted identifiers with some statistics.

    1. Derive vocabulary for typos correction - a set of tokens, which will be considered correctly
       spelled. All typos corrections will belong to the vocabulary.
       It is a set of most frequent tokens (based on given statistics).
    2. Save vocabulary and statistics for given amount of most frequent tokens for future use.
    3. Filter raw data, leaving identifiers, containing only tokens from the vocabulary.
       The result is a dataset of tokens which will be considered correct. It will be used
       for creating artificial misspelling cases for training and testing the corrector model.
    4. Save prepared dataset, if needed.

    :param params: Dictionary with parameters for data preparation. Used fields are:
                   input_path: Path to a .csv dump of input dataframe. Should contain \
                               column Columns.Split. If not specified, default \
                               dataset will be downloaded from google drive.
                   frequency_column: Name of column with identifiers frequencies. If not \
                                     specified, every split is considered to have frequency 1.
                   vocabulary_size: Number of most frequent tokens to take as a vocabulary.
                   frequencies_size: Number of most frequent tokens to save  frequencies info for.\
                                     This information will be used by corrector as features for \
                                     these tokens when they will be checked. If not specified, \
                                     frequencies for all present tokens will be saved.
                   vocabulary_path: .csv path to save vocabulary to.
                   frequencies_path: .csv path to save frequencies to.
    :return: Filtered dataset.
    """
    params = merge_dicts(defaults_for_preparation, params)
    if params["input_path"] is None:
        params["input_path"] = "lookout/style/typos/data/1M_repos2ids.csv"
        gdd.download_file_from_google_drive(file_id=DRIVE_DATASET_ID,
                                            dest_path=params["input_path"])
    data = pandas.read_csv(params["input_path"])
    if params["frequency_column"] not in data.columns:
        params["frequency_column"] = "freq"
        data[params["frequency_column"]] = [1] * len(data)

    # Expand dataframe by splits (repeat rows for every token in splits)
    flat_data = flatten_df_by_column(data, Columns.Split, Columns.Token,
                                     apply_function=lambda x: str(x).split())

    # Collect statistics for tokens
    stats = flat_data[[params["frequency_column"], Columns.Token]].groupby([Columns.Token]).sum()
    stats = stats.sort_values(by=[params["frequency_column"]], ascending=False)

    # Derive new vocabulary for future use
    params["frequencies_size"] = params["frequencies_size"] or len(stats)
    vocabulary_tokens = set(stats.index[:params["vocabulary_size"]])
    print_frequencies(vocabulary_tokens, params["frequency_column"], stats, params["vocabulary_path"])
    print_frequencies(set(stats.index[:params["frequencies_size"]]), params["frequency_column"], stats,
                      params["frequencies_path"])

    # Leave only splits that contain tokens from vocabulary
    prepared_data = filter_splits(flat_data, vocabulary_tokens)
    return prepared_data
