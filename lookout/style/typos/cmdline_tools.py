"""
Command line utilities to check the quality of a model on a given dataset, visualize errors, etc.
"""
from argparse import ArgumentParser
import json
from typing import Any, Mapping

from lookout.core.cmdline import ArgumentDefaultsHelpFormatterNoNone
from modelforge import slogging
import pandas

from lookout.style.typos.preparation import get_datasets, prepare_data, train_and_evaluate, \
    train_fasttext, train_from_scratch


def add_preparation_config_arg(my_parser: ArgumentParser) -> None:
    """
    Add config argument for data preparation.

    :param my_parser: Parser to add the argument to.
    """
    my_parser.add_argument(
        "--preparation-config", required=False, type=json.loads, default="{}",
        help="Config for data preparation in json format.")


def add_fasttext_config_arg(my_parser: ArgumentParser) -> None:
    """
    Add config argument for fasttext training.

    :param my_parser: Parser to add the argument to.
    """
    my_parser.add_argument(
        "--fasttext-config", required=False, type=json.loads, default="{}",
        help="Config for fasttext training in json format.")


def add_datasets_config_arg(my_parser: ArgumentParser) -> None:
    """
    Add config argument for train and test datasets generation.

    :param my_parser: Parser to add the argument to.
    """
    my_parser.add_argument(
        "--datasets-config", required=False, type=json.loads, default="{}",
        help="Config for datasets generation in json format.")


def add_data_path_arg(my_parser: ArgumentParser) -> None:
    """
    Add data path argument.

    :param my_parser: Parser to add the argument to.
    """
    my_parser.add_argument(
        "--data-path", required=True, type=str,
        help=".csv dump of a Dataframe with columns Columns.Split and Columns.Frequency.")


def add_save_model_path_arg(my_parser: ArgumentParser) -> None:
    """
    Add save model path argument.

    :param my_parser: Parser to add the argument to.
    """
    my_parser.add_argument(
        "--save-model-path", required=True, type=str,
        help="Path to save the trained model to (.asdf).")


def cli_train_fasttext(data_path: str, fasttext_config: Mapping[str, Any]) -> None:
    """Entry point for `train_fasttext`."""
    train_fasttext(pandas.read_csv(data_path, index_col=0, keep_default_na=False),
                   fasttext_config)


def cli_get_datasets(data_path: str, datasets_config: Mapping[str, Any]) -> None:
    """Entry point for `get_datasets`."""
    get_datasets(pandas.read_csv(data_path, index_col=0, keep_default_na=False), datasets_config)


def cli_train_corrector(data_path: str, test_data_path: str, vocabulary_path: str,
                        frequencies_path: str, fasttext_path: str, save_model_path: str) -> None:
    train = pandas.read_csv(data_path, index_col=0, keep_default_na=False)
    test = pandas.read_csv(test_data_path, index_col=0, keep_default_na=False)
    model = train_and_evaluate(train, test, vocabulary_path, frequencies_path, fasttext_path)
    model.save(save_model_path, series=0.0)


def create_parser() -> ArgumentParser:
    """
    Create a parser for the lookout.style.typos utility.

    :return: an ArgumentParser with an handler defined in the handler attribute.
    """
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatterNoNone)

    # General options
    slogging.add_logging_args(parser)
    subparsers = parser.add_subparsers(help="Commands")

    def add_parser(name, help):
        return subparsers.add_parser(
            name, help=help, formatter_class=ArgumentDefaultsHelpFormatterNoNone)

    # Prepare raw data for corrector
    prepare_parser = add_parser("prepare-data", "Prepare raw dataset for corrector training.")
    prepare_parser.set_defaults(handler=prepare_data)
    add_preparation_config_arg(prepare_parser)

    # Train new fasttext model on gien data
    fasttext_parser = add_parser("train-fasttext", "Train fasttext model on the given dataset"
                                                   "of code identifiers.")
    fasttext_parser.set_defaults(handler=cli_train_fasttext)
    add_data_path_arg(fasttext_parser)
    add_fasttext_config_arg(fasttext_parser)

    # Create train and test datasets with artificial typos
    datasets_parser = add_parser("get-datasets",
                                 "Create the train and the test datasets of typos.")
    datasets_parser.set_defaults(handler=cli_get_datasets)
    add_data_path_arg(datasets_parser)
    add_datasets_config_arg(datasets_parser)

    # Create, train and evaluate new corrector model
    train_parser = add_parser("train", "Create and train TyposCorrector model on the given data.")
    train_parser.set_defaults(handler=cli_train_corrector)
    add_data_path_arg(train_parser)
    train_parser.add_argument(
        "-t", "--test-data-path", required=True, type=str,
        help=".csv dump of a Dataframe with columns Columns.Split and Columns.Frequency."
    )
    train_parser.add_argument(
        "-v", "--vocabulary-path", required=True, type=str,
        help="Path to a .csv file with vocabulary."
    )
    train_parser.add_argument(
        "-f", "--frequencies-path", required=True, type=str,
        help="Path to a .csv file with tokens' frequencies."
    )
    train_parser.add_argument(
        "-e", "--fasttext-path", required=True, type=str,
        help="Path to a FastText model's dump (.bin)."
    )
    add_save_model_path_arg(train_parser)

    # Create and train new corrector model from scratch
    train_from_scratch_parser = add_parser(
        "train-from-scratch", "Create and train TyposCorrector  model on the given data.")
    train_from_scratch_parser.set_defaults(handler=train_from_scratch)
    add_preparation_config_arg(train_from_scratch_parser)
    add_fasttext_config_arg(train_from_scratch_parser)
    add_datasets_config_arg(train_from_scratch_parser)
    train_from_scratch_parser.add_argument(
        "-e", "--fasttext-path", type=str, required=False,
        help="Path to the pretrained fasttext model. If not specified correctly, "
             "new fasttext model will not be trained."
    )
    add_save_model_path_arg(train_from_scratch_parser)

    return parser


def main() -> Any:
    """Entry point of the utility."""
    parser = create_parser()
    args = parser.parse_args()
    slogging.setup(args.log_level, args.log_structured, args.log_config)
    delattr(args, "log_level")
    delattr(args, "log_structured")
    delattr(args, "log_config")
    try:
        handler = args.handler
        delattr(args, "handler")
    except AttributeError:
        def print_usage(*args, **kwargs):
            parser.print_usage()

        handler = print_usage
    return handler(**vars(args))
