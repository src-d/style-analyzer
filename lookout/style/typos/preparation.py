import logging
import os
from pprint import pformat
import sys
import tempfile
from typing import Any, Mapping, Optional, Tuple
import urllib.request

import pandas
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from lookout.style.common import merge_dicts
from lookout.style.typos.config import DEFAULT_CORRECTOR_CONFIG
from lookout.style.typos.corrector import TyposCorrector
from lookout.style.typos.corruption import corrupt_tokens_in_df
from lookout.style.typos.utils import Columns, filter_splits, flatten_df_by_column, \
    print_frequencies


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
    log = logging.getLogger("prepare_data")
    if config is None:
        config = {}
    config = merge_dicts(DEFAULT_CORRECTOR_CONFIG["preparation"], config)

    os.makedirs(config["data_dir"], exist_ok=True)
    raw_data_path = config["input_path"]
    if raw_data_path is None or not os.path.exists(raw_data_path):
        raw_data_path = os.path.join(config["data_dir"],
                                     config["raw_data_filename"])
        log.warning("raw dataset was not found, downloading from %s to %s",
                    config["dataset_url"], raw_data_path)
        _download_url(config["dataset_url"], raw_data_path)

    data = pandas.read_csv(raw_data_path, index_col=0, keep_default_na=False)
    log.debug("raw dataset shape: %s", data.shape)
    if config["frequency_column"] not in data.columns:
        log.info("frequency column is not found. Set all frequencies to 1")
        data[Columns.Frequency] = 1
    else:
        log.info("frequency column `%s` is found", config["frequency_column"])
        data = data.rename(columns={config["frequency_column"]: Columns.Frequency})

    # Expand dataframe by splits (repeat rows for every token in splits)
    data[Columns.Split] = data[Columns.Split].astype(str)
    log.debug("expand data by splits")
    flat_data = flatten_df_by_column(data, Columns.Split, Columns.Token,
                                     apply_function=lambda x: x.split())
    log.debug("expanded data shape %s", flat_data.shape)

    log.info("collect statistics for tokens")
    stats = flat_data[[Columns.Frequency, Columns.Token]].groupby([Columns.Token]).sum()
    stats = stats.sort_values(by=Columns.Frequency, ascending=False)[Columns.Frequency]

    log.info("derive the new vocabulary")
    frequencies = stats.iloc[:(config["frequencies_size"] or len(stats))].to_dict()
    log.info("tokens with frequencies data size: %d", len(frequencies))
    vocabulary = stats.iloc[:config["vocabulary_size"]].to_dict()
    log.info("vocabulary size: %d", len(vocabulary))
    vocabulary_filepath = os.path.join(config["data_dir"], config["vocabulary_filename"])
    print_frequencies(vocabulary, vocabulary_filepath)
    log.info("vocabulary saved to %s", vocabulary_filepath)
    frequencies_filepath = os.path.join(config["data_dir"], config["frequencies_filename"])
    print_frequencies(frequencies, frequencies_filepath)
    log.info("tokens with frequencies data are saved to %s", frequencies_filepath)

    # Leave only splits that contain tokens from vocabulary
    prepared_data = filter_splits(flat_data, set(vocabulary.keys()))[
        [Columns.Frequency, Columns.Split, Columns.Token]]
    prepared_data.reset_index(drop=True, inplace=True)
    log.info("final dataset shape: %s", prepared_data.shape)
    if config["prepared_filename"] is not None:
        prepared_data_filepath = os.path.join(config["data_dir"], config["prepared_filename"])
        prepared_data.to_csv(prepared_data_filepath)
        log.info("final dataset is saved to %s", prepared_data_filepath)
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
    log = logging.getLogger("train_fasttext")
    if config is None:
        config = {}
    config = merge_dicts(DEFAULT_CORRECTOR_CONFIG["fasttext"], config)
    train_data = data[[len(str(x).split()) > 2 for x in data[Columns.Split]]].sample(
        config["size"], weights=Columns.Frequency, replace=True)
    if config["corrupt"]:
        train_data = corrupt_tokens_in_df(train_data, config["typo_probability"],
                                          config["add_typo_probability"])
    with tempfile.NamedTemporaryFile() as ids_file:
        with open(ids_file.name, "w") as f:
            for token_split in train_data[Columns.Split]:
                f.write(token_split + "\n")
        log.info("Training fasttext model...")
        model = fastText.train_unsupervised(ids_file.name, minCount=1, epoch=10,
                                            dim=config["dim"],
                                            bucket=config["bucket"])
    model.save_model(config["path"])
    log.info("fasttext model is saved to %s", config["path"])


def get_datasets(prepared_data: pandas.DataFrame,
                 config: Optional[Mapping[str, Any]] = None,
                 processes_number: int = DEFAULT_CORRECTOR_CONFIG["processes_number"],
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
    log = logging.getLogger("get_datasets")
    if config is None:
        config = {}
    config = merge_dicts(DEFAULT_CORRECTOR_CONFIG["datasets"], config)
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
    log.info("train dataset shape: %s", train.shape)
    log.info("test dataset shape: %s", test.shape)
    train = corrupt_tokens_in_df(train, config["typo_probability"], config["add_typo_probability"],
                                 processes_number)
    test = corrupt_tokens_in_df(test, config["typo_probability"], config["add_typo_probability"],
                                processes_number)
    if config["test_path"] is not None:
        test.to_csv(config["test_path"])
        log.info("test dataset is saved to %s", config["test_path"])
    if config["train_path"] is not None:
        train.to_csv(config["train_path"])
        log.info("train dataset is saved to %s", config["train_path"])
    return train, test


def train_and_evaluate(train_data: pandas.DataFrame, test_data: pandas.DataFrame,
                       vocabulary_path: str, frequencies_path: str, fasttext_path: str,
                       generation_config: Optional[Mapping[str, Any]] = None,
                       ranking_config: Optional[Mapping[str, Any]] = None,
                       processes_number: int = DEFAULT_CORRECTOR_CONFIG["processes_number"],
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
    :param generation_config: Candidates generation configuration.
    :param ranking_config: Ranking configuration.
    :param processes_number: Number of processes for multiprocessing.
    :return: Trained model.
    """
    model = TyposCorrector(ranking_config)
    model.initialize_generator(vocabulary_file=vocabulary_path,
                               frequencies_file=frequencies_path,
                               embeddings_file=fasttext_path, config=generation_config)
    model.processes_number = processes_number
    model.train(train_data)
    _, report = model.evaluate(test_data)
    print(report)
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
    log = logging.getLogger("train_from_scratch")
    if config is None:
        config = {}
    config = merge_dicts(DEFAULT_CORRECTOR_CONFIG, config)
    log.info("effective config:\n%s", pformat(config, width=120, compact=True))
    prepared_data = prepare_data(config["preparation"])
    if config["fasttext"]["path"] is None or not os.path.exists(config["fasttext"]["path"]):
        log.info("fasttext model is not found and will be trained")
        train_fasttext(prepared_data, config["fasttext"])
    train_data, test_data = get_datasets(prepared_data, config["datasets"])
    model = train_and_evaluate(train_data, test_data,
                               os.path.join(config["preparation"]["data_dir"],
                                            config["preparation"]["vocabulary_filename"]),
                               os.path.join(config["preparation"]["data_dir"],
                                            config["preparation"]["frequencies_filename"]),
                               config["fasttext"]["path"], config["generation"],
                               config["ranking"], config["processes_number"])
    if config["corrector_path"] is not None:
        model.save(config["corrector_path"], series=0.0)
        log.info("corrector model is saved to %s", config["corrector_path"])
    return model
