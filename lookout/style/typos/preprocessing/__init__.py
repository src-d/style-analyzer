import argparse
import sys

from sourced.ml.cmd.args import ArgumentDefaultsHelpFormatterNoNone

from lookout.style.typos.preprocessing.create_typos import create_typos, corrupt_tokens_in_df, train_test_split
from lookout.style.typos.preprocessing.filter_identifiers import filter_identifiers, filter_splitted_identifiers
from lookout.style.typos.preprocessing.get_frequencies import get_frequencies, print_frequencies
from lookout.style.typos.preprocessing.pick_subset import pick_subset, pick_subset_of_df


def get_parser() -> argparse.ArgumentParser:
    """
    Creates the cmdline argument parser.
    """

    parser = argparse.ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatterNoNone,
                                     description='Typos correction tools')
    subparsers = parser.add_subparsers(help="Commands", dest="command")

    def add_parser(name, help_message):
        return subparsers.add_parser(name, help=help_message,
                                     formatter_class=ArgumentDefaultsHelpFormatterNoNone)

    # ------------------------------------------------------------------------
    create_typos_parser = add_parser("create_typos", "Create random edit typos with given \
                                     probabilities in a dataset of tokens. May also split \
                                     resulting dataset in given proportions")

    create_typos_parser.set_defaults(handler=create_typos)

    create_typos_parser.add_argument("input_file", metavar="input file", type=str,
                                     help="File of pickled input pandas dataframe. \
                                     Dataframe must contain column 'identifier'")

    create_typos_parser.add_argument("out_file", metavar="output file", type=str,
                                     help="File to dump pickled output dataframe. \
                                     Dataframe contains same columns as input one \
                                     plus columns \'typo\' and \'corrupted\'")

    create_typos_parser.add_argument("-tp", "--typo_probability", type=float, default=0.5,
                                     help="Probability of misspelling a word")

    create_typos_parser.add_argument("-atp", "--add_typo_probability", type=float, default=0.1,
                                     help="Probability of making additional typo in a same word")

    create_typos_parser.add_argument("-s", "--test_portion", type=float, help="Test portion \
                                     for spliting data on train and test chunks without mixing \
                                     rows")

    # ------------------------------------------------------------------------
    filter_parser = add_parser("filter_identifiers", "Filter dataframe of splitted identifiers. \
                               Leave only rows with identifiers, which contain as tokens after \
                               split only words from provided vocabulary collection")

    filter_parser.set_defaults(handler=filter_identifiers)

    filter_parser.add_argument("id_file", metavar="identifiers file", type=str,
                               help="File with pickled identifiers dataframe. \
                               Dataframe must contain column 'token_split'")

    filter_parser.add_argument("vocabulary_file", metavar="vocabulary file", type=str,
                               help="File with pickled vocabulary array")

    filter_parser.add_argument("out_file", metavar="output file", type=str,
                               help="File to put pickled filtered dataframe")

    # ------------------------------------------------------------------------
    frequencies_parser = add_parser("get_frequencies", "Get frequencies for tokens \
                                    of specified vocabulary from the stats file")

    frequencies_parser.set_defaults(handler=get_frequencies)

    frequencies_parser.add_argument("vocabulary_file", metavar="vocabulary file", type=str,
                                    help="File with pickled vocabulary array")

    frequencies_parser.add_argument("stats_file", metavar="identifiers stats file", type=str,
                                    help="File with pickled identifiers stats dataframe. \
                                    Dataframe must be indexed by 'identifier' and \
                                    contain column 'num_occ'.")

    frequencies_parser.add_argument("out_file", metavar="output file", type=str,
                                    help="File csv to put frequencies data \
                                    with rows in a format (token, frequency).")

    # ------------------------------------------------------------------------
    pick_parser = add_parser("pick_subset", "Pick random subset of rows in an input dataframe \
                             and save them in an output dataframe")

    pick_parser.set_defaults(handler=pick_subset)

    pick_parser.add_argument("input_file", metavar="input file", type=str,
                             help="File with pickled input dataframe.")

    pick_parser.add_argument("picked_portion", metavar="picked portion", type=float,
                             help="Portion of rows to pick.")

    pick_parser.add_argument("out_file", metavar="output file", type=str,
                             help="File to dump picked dataframe.")

    pick_parser.add_argument("-w", "--weight_column", type=str,
                             help="Column to use as weights for rows to pick."
                                  "If not specified, uniform weights are assigned.")

    return parser


def main():
    """
    Creates all the argparse-rs and invokes the function from set_defaults().
    :return: The result of the function from set_defaults().
    """

    parser = get_parser()
    args = parser.parse_args()

    try:
        handler = args.handler
    except AttributeError:
        def print_usage(_):
            parser.print_usage()

        handler = print_usage
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
