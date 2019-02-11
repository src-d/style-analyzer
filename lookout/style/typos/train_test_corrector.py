from typing import Tuple, Union

import pandas

from lookout.style.typos.corrector import TyposCorrector
from lookout.style.typos.utils import COLUMNS, flatten_df_by_column
from lookout.style.typos.preprocessing import (filter_splitted_identifiers, pick_subset_of_df,
                                               corrupt_tokens_in_df, print_frequencies, train_test_split)
from lookout.style.typos.preprocessing.metrics import print_scores


DEFAULT_VOCABULARY_SIZE = 10000
DEFAULT_VOCABULARY_PATH = "lookout/style/typos/data/vocabulary.csv"
DEFAULT_FREQUENCY_PATH = "lookout/style/typos/data/frequencies.csv"


def prepare_data(input_path: str, frequency_column: str = None,
                 vocabulary_size: int = DEFAULT_VOCABULARY_SIZE, frequencies_size: int = None,
                 vocabulary_path: str = DEFAULT_VOCABULARY_PATH,
                 frequencies_path: str = DEFAULT_FREQUENCY_PATH,
                 save_prepared_path: str = None) -> pandas.DataFrame:
    """
    Get all necessary data from raw dataset of splitted identifiers with some statistics:
    1. Derive vocabulary for typos correction - a set of tokens, which will be considered correctly
       spelled. All typos corrections will belong to the vocabulary.
       It is a set of most frequent tokens (based on given statistics).
    2. Save vocabulary and statistics for given amount of most frequent tokens for future use.
    3. Filter raw data, leaving identifiers, containing only tokens from the vocabulary.
       The result is a dataset of tokens which will be considered correct. It will be used
       for creating artificial misspelling cases for training and testing the corrector model.
    4. Save prepared dataset, if needed.
    :param input_path: Path to a .csv dump of input dataframe. Should contain column
                       COLUMNS["SPLIT"].
    :param frequency_column: Name of column with identifiers frequencies. If not specified,
                             every split is considered to have frequency 1.
    :param vocabulary_size: Number of most frequent tokens to take as a vocabulary.
    :param frequencies_size: Number of most frequent tokens to save frequencies info for.
                             This information will be used by corrector as features
                             for these tokens when they will be checked. If not specified,
                             frequencies for all present tokens will be saved.
    :param vocabulary_path: Path to save vocabulary to.
    :param frequencies_path: Path to save frequencies to.
    :param save_prepared_path: Path to save filtered dataset to.
    :return: Filtered dataset.
    """
    data = pandas.read_csv(input_path)

    if frequency_column not in data.columns:
        frequency_column = "freq"
        data[frequency_column] = [1] * len(data)

    # Expand dataframe by splits (repeat rows for every token in splits)
    flat_data = flatten_df_by_column(data, COLUMNS["SPLIT"], COLUMNS["TOKEN"],
                                     apply_function=lambda x: str(x).split())

    # Collect statistics for tokens
    stats = flat_data[[frequency_column, COLUMNS["TOKEN"]]].groupby([COLUMNS["TOKEN"]]).sum()
    stats = stats.sort_values(by=[frequency_column], ascending=False)

    # Derive new vocabulary for future use
    vocabulary_size = vocabulary_size or DEFAULT_VOCABULARY_SIZE
    vocabulary_tokens = set(stats.loc[stats.index[:vocabulary_size]])

    frequencies_size = frequencies_size or len(stats)

    # Saving frequencies info for future use in the corrector model
    print_frequencies(vocabulary_tokens, stats, vocabulary_path)
    print_frequencies(set(stats.loc[stats.index[:frequencies_size]]), stats,
                      frequencies_path)

    # Leave only splits that contain tokens from vocabulary
    prepared_data = filter_splitted_identifiers(flat_data, vocabulary_tokens)
    if save_prepared_path:
        prepared_data.to_csv(save_prepared_path)

    return prepared_data


def get_train_test(prepared_data: Union[pandas.DataFrame, str], train_size: int, test_size: int,
                   frequency_column: str = None, save_train_path: str = None,
                   save_test_path: str = None) -> Tuple[pandas.DataFrame, pandas.DataFrame]:
    """
    Create train and test datasets of typos:
    1. Pick specified amount of data from input dataset.
    2. Artificially create typos in picked identifiers and put them to train and test datasets.
    3. Save results, if needed.
    :param prepared_data: Dataframe or its .csv dump of correct splitted identifiers (output
                          of :func:`~prepare_data`. Must contain columns
                          COLUMNS["SPLIT"] and COLUMNS["TOKEN"].
    :param train_size: Train dataset size.
    :param test_size: Test dataset size.
    :param frequency_column: Column to use as weights for picking rows.
    :param save_train_path: Path to save train dataset.
    :param save_test_path: Path to save test dataset.
    :return: Train and test datasets.
    """
    if isinstance(prepared_data, str):
        prepared_data = pandas.read_csv(prepared_data)

    total = train_size + test_size
    picked_data = pick_subset_of_df(prepared_data, size=total, weight_column=frequency_column)
    corrupted_data = corrupt_tokens_in_df(picked_data, 0.5, 0.01)

    train_data, test_data = train_test_split(corrupted_data, test_size / total)

    if save_train_path:
        train_data.to_csv(save_train_path)
    if save_test_path:
        test_data.to_csv(save_test_path)

    return train_data, test_data


def train(train_data: Union[pandas.DataFrame, str], vocabulary_path: str, frequencies_path: str,
          embeddings_path: str, threads_number: int = 8, save_model_path: str = None
          ) -> TyposCorrector:
    """
    Create and train TyposCorrector model on given data.
    :param train_data: DataFrame, or its .csv dump, containing columns
                       COLUMNS["TOKEN"] and COLUMNS["CORRECT_TOKEN"],
                       column COLUMNS["SPLIT"] is optional, but used when present.
    :param vocabulary_path: Path to a file with vocabulary.
    :param frequencies_path: Path to a file with tokens' frequencies.
    :param embeddings_path: Path to a FastText model dump.
    :param threads_number: Number of threads for multiprocessing.
    :param save_model_path: Path to save model to.
    :return: Trained model.
    """
    model = TyposCorrector()
    model.initialize_ranker()
    model.initialize_generator(vocabulary_file=vocabulary_path,
                               frequencies_file=frequencies_path,
                               embeddings_file=embeddings_path)
    model.threads_number = threads_number
    
    if isinstance(train_data, str):
        train_data = pandas.read_csv(train_data, index_col=0)
    model.train(train_data)

    if save_model_path:
        model.save(save_model_path)

    return model


def test(model: Union[TyposCorrector, str], test_data: Union[pandas.DataFrame, str]) -> None:
    """
    Test TyposCorrector model on given dataset and print results to the standard output.
    :param model: TyposCorrector model or path to its dump.
    :param test_data: Dataframe or its .csv dump, containing columns
           COLUMNS["TOKEN"] and COLUMNS["CORRECT_TOKEN"].
    """
    if isinstance(test_data, str):
        test_data = pandas.read_csv(test_data, index_col=0)
    if isinstance(model, str):
        model_path = model
        model = TyposCorrector()
        model.load(model_path)
    suggestions_test = model.suggest(test_data)
    print_scores(test_data, suggestions_test)


def train_from_scratch(input_path: str = "lookout/style/typos/data/100k_repos2ids.csv",
                       frequency_column: str = "num_occ",
                       vocabulary_size: int = DEFAULT_VOCABULARY_SIZE, frequencies_size: int = 200000,
                       vocabulary_path: str = DEFAULT_VOCABULARY_PATH,
                       frequencies_path: str = DEFAULT_FREQUENCY_PATH,
                       save_prepared_path: str = "lookout/style/typos/data/prepared_data.csv",
                       train_size: int = 50000, test_size: int = 10000, threads_number: int = 8,
                       embeddings_path: str = "lookout/style/typos/data/id_vecs_10.bin",
                       save_train_path: str = "lookout/style/typos/data/train.csv",
                       save_test_path: str = "lookout/style/typos/data/test.csv",
                       save_model_path: str = "lookout/style/typos/data/new_corrector.asdf"
                       ) -> TyposCorrector:
    """
    Train TyposCorrector on raw data.
    1. Prepare data, for more info check :func:`~prepare_data`.
    2. Construct train and test datasets, for more info check :func:`~get_train_test`.
    3. Train TyposCorrector model. Save if needed.
    4. Test corrector and print results, if needed.
    :param input_path: Path to a .csv dump of input dataframe. Should contain column
                       COLUMNS["SPLIT"].
    :param frequency_column: Name of column with identifiers frequencies. If not specified,
                             every split is considered to have frequency 1.
    :param vocabulary_size: Number of most frequent tokens to take as a vocabulary.
    :param frequencies_size: Number of most frequent tokens to save frequencies info for.
                             This information will be used by corrector as features
                             for these tokens when they will be checked. If not specified,
                             frequencies for all present tokens will be saved.
    :param vocabulary_path: Path to save vocabulary to.
    :param frequencies_path: Path to save frequencies to.
    :param save_prepared_path: Path to save filtered dataset to.
    :param train_size: Train dataset size.
    :param test_size: Test dataset size.
    :param save_train_path: Path to save train dataset.
    :param save_test_path: Path to save test dataset.
    :param embeddings_path: Path to a FastText model dump.
    :param threads_number: Number of threads for multiprocessing.
    :param save_model_path: Path to save model to.
    :return: Trained TyposCorrector model.
    """
    # Prepare raw dataset for using by TyposCorrector and derive vocabulary if needed.
    prepared_data = prepare_data(input_path, frequency_column, vocabulary_size, frequencies_size,
                                 vocabulary_path, frequencies_path, save_prepared_path)

    # Constructing train and test datasets.
    train_data, test_data = get_train_test(prepared_data, train_size, test_size, frequency_column,
                                           save_train_path, save_test_path)

    # Train TyposCorrector model on obtained data.
    # Save model if save_model_path specified.
    model = train(train_data, vocabulary_path, frequencies_path, embeddings_path, threads_number,
                  save_model_path)

    # Print result on test part.
    if test_size > 0:
        test(model, test_data)

    return model
