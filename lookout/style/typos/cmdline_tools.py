"""
Command line utilities to check the quality of a model on a given dataset, visualize errors, etc.
"""
from argparse import ArgumentParser
from typing import Any

from lookout.core.cmdline import ArgumentDefaultsHelpFormatterNoNone
from lookout.core.slogging import setup as setup_slogging

from lookout.style.typos.train_test_corrector import (DEFAULT_VOCABULARY_SIZE,
                                                      DEFAULT_VOCABULARY_PATH,
                                                      DEFAULT_FREQUENCY_PATH)


def add_input_path_arg(my_parser: ArgumentParser, addition: str = ""):
    """
    Add an input path argument to an argparse parser.

    :param my_parser: Parser to add the argument to.
    :param addition: Addition to the default help message.
    """
    my_parser.add_argument(
        "-i", "--input-path", required=False, type=str,
        help="Path to a .csv dump of input dataframe. Should contain column 'token_split" +
             addition)


def add_frequency_column_arg(my_parser: ArgumentParser):
    """
    Add a frequency column argument to an argparse parser.

    :param my_parser: Parser to add the argument to.
    """
    my_parser.add_argument(
        "-fc", "--frequency_column", required=False, type=str,
        help="Name of column with identifiers frequencies. If not specified,"
             "every split is considered to have frequency 1.")


def add_vocabulary_size_arg(my_parser: ArgumentParser):
    """
    Add a vocabulary size argument to an argparse parser.

    :param my_parser: Parser to add the argument to.
    """
    my_parser.add_argument(
        "-vs", "--vocabulary_size", required=False, type=int,
        default=DEFAULT_VOCABULARY_SIZE,
        help="Number of most frequent tokens to take as a vocabulary.")


def add_frequencies_size_arg(my_parser: ArgumentParser):
    """
    Add a frequencies size argument to an argparse parser.

    :param my_parser: Parser to add the argument to.
    """
    my_parser.add_argument(
        "-fs", "--frequencies_size", required=False, type=int,
        help="Number of most frequent tokens to save frequencies info for."
             "If not specified, frequencies for all present tokens will be saved.")


def add_vocabulary_path_arg(my_parser: ArgumentParser):
    """
    Add a vocabulary path argument to an argparse parser.

    :param my_parser: Parser to add the argument to.
    """
    my_parser.add_argument(
        "-vp", "--vocabulary_path", required=False, type=str,
        default=DEFAULT_VOCABULARY_PATH,
        help="Path to save vocabulary to.")


def add_frequencies_path_arg(my_parser: ArgumentParser):
    """
    Add a frequencies path argument to an argparse parser.

    :param my_parser: Parser to add the argument to.
    """
    my_parser.add_argument(
        "-fp", "--frequencies_path", required=False, type=str,
        default=DEFAULT_FREQUENCY_PATH,
        help="Path to save frequencies to.")


def add_save_prepared_path_arg(my_parser: ArgumentParser):
    """
    Add a save prepared path argument to an argparse parser.

    :param my_parser: Parser to add the argument to.
    """
    my_parser.add_argument(
        "-spp", "--save_prepared_path", required=False, type=str,
        help="Path to save filtered dataset to.")


def add_train_size_arg(my_parser: ArgumentParser):
    """
    Add a train size argument to an argparse parser.

    :param my_parser: Parser to add the argument to.
    """
    my_parser.add_argument(
        "-trs", "--train_size", required=False, type=int,
        default=50000,
        help="Train dataset size.")


def add_test_size_arg(my_parser: ArgumentParser):
    """
    Add a test size argument to an argparse parser.

    :param my_parser: Parser to add the argument to.
    """
    my_parser.add_argument(
        "-ts", "--test_size", required=False, type=int,
        default=10000,
        help="Test dataset size.")


def add_save_train_path_arg(my_parser: ArgumentParser):
    """
    Add a save train path argument to an argparse parser.

    :param my_parser: Parser to add the argument to.
    """
    my_parser.add_argument(
        "-strp", "--save_train_path", required=False, type=str,
        help="Path to save train dataset to.")


def add_save_test_path_arg(my_parser: ArgumentParser):
    """
    Add a save test path argument to an argparse parser.

    :param my_parser: Parser to add the argument to.
    """
    my_parser.add_argument(
        "-stp", "--save_test_path", required=False, type=str,
        help="Path to save test dataset to.")


def add_threads_number_arg(my_parser: ArgumentParser):
    """
    Add a threads number argument to an argparse parser.

    :param my_parser: Parser to add the argument to.
    """
    my_parser.add_argument(
        "-tn", "--threads_number", required=False, type=int, default=8,
        help="Number of threads for multiprocessing.")


def add_save_model_path_arg(my_parser: ArgumentParser):
    """
    Add a save model path argument to an argparse parser.

    :param my_parser: Parser to add the argument to.
    """
    my_parser.add_argument(
        "-smp", "--save_model_path", required=False, type=str,
        default="lookout/style/typos/data/new_corrector.asdf",
        help="Path to save model to.")


def add_fasttext_path_arg(my_parser: ArgumentParser, addition: str = ""):
    """
    Add a fasttext path argument to an argparse parser.

    :param my_parser: Parser to add the argument to.
    :param addition: Addition to the default help message.
    """
    my_parser.add_argument(
        "-ft", "--fasttext_path", required=False, type=str,
        help="Path to a FastText model dump." + addition)


def create_parser() -> ArgumentParser:
    """
    Create a parser for the lookout.style.typos utility.

    :return: an ArgumentParser with an handler defined in the handler attribute.
    """
    # Deferred imports to speed up loading __init__
    from lookout.style.typos.train_test_corrector import (get_train_test, prepare_data, test,
                                                          train, train_from_scratch)

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatterNoNone)

    # General options
    parser.add("--log-level", default="DEBUG", help="Log verbosity level.")

    subparsers = parser.add_subparsers(help="Commands")

    def add_parser(name, help):
        return subparsers.add_parser(
            name, help=help, formatter_class=ArgumentDefaultsHelpFormatterNoNone)

    # Prepare raw data for corrector
    prepare_parser = add_parser("prepare-data", "Prepare raw dataset for corrector training.")
    prepare_parser.set_defaults(handler=prepare_data)
    add_input_path_arg(prepare_parser)
    add_frequency_column_arg(prepare_parser)
    add_vocabulary_size_arg(prepare_parser)
    add_frequencies_size_arg(prepare_parser)
    add_vocabulary_path_arg(prepare_parser)
    add_frequencies_path_arg(prepare_parser)
    add_save_prepared_path_arg(prepare_parser)

    # Get train and test datasets
    train_test_parser = add_parser("get-train-test",
                                   "Get train and test datasets with artificially"
                                   "created typos from prepared data.")
    train_test_parser.set_defaults(handler=get_train_test)
    train_test_parser.add_argument(
        "prepared_data", metavar="data", type=str,
        help=".csv dump of dataframe correct splitted identifiers"
             ". Must contain columns 'token_split' and 'token'."
    )
    add_train_size_arg(train_test_parser)
    add_test_size_arg(train_test_parser)
    add_frequency_column_arg(train_test_parser)
    add_save_train_path_arg(train_test_parser)
    add_save_test_path_arg(train_test_parser)

    # Train TyposCorrector model.
    train_parser = add_parser("train",
                              "Create and train TyposCorrector model on given data.")
    train_parser.set_defaults(handler=train)
    train_parser.add_argument("train_data", metavar="data", type=str,
                              help=".csv dump of a train dataframe, containing"
                                   "columns'token' and 'correct_token'.")
    add_vocabulary_path_arg(train_parser)
    add_frequencies_path_arg(train_parser)
    add_fasttext_path_arg(train_parser)
    add_threads_number_arg(train_parser)
    add_save_model_path_arg(train_parser)

    # Evaluate on different styles dataset
    test_parser = add_parser("test",
                             "Test TyposCorrector model on given dataset"
                             "and print results to the standard output.")
    test_parser.set_defaults(handler=test)
    test_parser.add_argument(
        "model", metavar="model", type=str,
        help="Path to TyposCorrector model dump (.asdf)")
    test_parser.add_argument(
        "test_data", metavar="data", type=str,
        help="csv dump of a test dataframe, containing"
             "columns 'token' and 'correct_token'.")

    # Train TyposCorrector from scratch
    train_from_scratch_parser = add_parser("train-from-scratch",
                                           "Train TyposCorrector on raw data.")
    train_from_scratch_parser.set_defaults(handler=train_from_scratch)
    add_input_path_arg(train_from_scratch_parser,
                       " If not specified, default dataset"
                       "will be downloaded from google drive.")
    add_fasttext_path_arg(train_from_scratch_parser,
                          " If not specified, default model will"
                          "be downloaded from google drive.")
    add_vocabulary_size_arg(train_from_scratch_parser)
    add_frequencies_size_arg(train_from_scratch_parser)
    add_vocabulary_path_arg(train_from_scratch_parser)
    add_frequencies_path_arg(train_from_scratch_parser)
    add_save_prepared_path_arg(train_from_scratch_parser)
    add_train_size_arg(train_from_scratch_parser)
    add_test_size_arg(train_from_scratch_parser)
    add_save_train_path_arg(train_from_scratch_parser)
    add_save_test_path_arg(train_from_scratch_parser)
    add_threads_number_arg(train_from_scratch_parser)
    add_save_model_path_arg(train_from_scratch_parser)

    return parser


def main() -> Any:
    """Entry point of the utility."""
    parser = create_parser()
    args = parser.parse_args()
    setup_slogging(args.log_level, False)
    delattr(args, "log_level")
    try:
        handler = args.handler
        delattr(args, "handler")
    except AttributeError:
        def print_usage(*args, **kwargs):
            parser.print_usage()

        handler = print_usage
    return handler(**vars(args))
