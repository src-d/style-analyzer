import argparse
import sys

from lookout.core.cmdline import ArgumentDefaultsHelpFormatterNoNone

from lookout.style.typos.research.baseline import baseline
from lookout.style.typos.research.create_typos import create_typos
from lookout.style.typos.research.filter_identifiers import filter_identifiers
from lookout.style.typos.research.get_frequencies import get_frequencies
from lookout.style.typos.research.pick_subset import pick_subset


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

    baseline_parser = add_parser("baseline", """
    Typos correction model, based on SymSpell lookout algorithm

    https://github.com/wolfgarbe/SymSpell

    and simple Random Forest classifier, based on token frequencies
    and edit distance between typo and candidate.

    May use pretrained model.

    Dumps model if the dump model file specified.

    Training: Requires token frequencies info and training dataframe,
              may also accept dataframe with precalculated candidates suggestions.

    Testing:  Requires training first or pretrained model, testing dataframe,
              may also accept dataframe with precalculated candidates suggestions.
              Prints results to screen by default or to the specified file.
    """)

    baseline_parser.set_defaults(handler=baseline)

    baseline_parser.add_argument("-p", "--pretrained_file", type=str,
                                 help="File contaning dump of pretrained model")

    baseline_parser.add_argument("-f", "--frequencies_file", type=str,
                                 help="File containing token frequencies \
                                 with rows in a format (token, frequency)")

    baseline_parser.add_argument("-tr", "--train_file", type=str,
                                 help="File of pickled training dataframe. Dataframe \
                                 indexed by 'id' and containing columns 'identifier', 'typo'")

    baseline_parser.add_argument("-ctr", "--cand_train_file", type=str,
                                 help="File of pickled training candidates dataframe. \
                                 Candidates dataframe has to contain columns \
                                 ('id', 'typo', 'candidate', 'typo_freq', \
                                 'cand_freq', 'distance')")

    baseline_parser.add_argument("-d", "--dump_file", type=str,
                                 help="File to dump model")

    baseline_parser.add_argument("-te", "--test_file", type=str,
                                 help="File of pickled testing dataframe. \
                                 Dataframe indexed by 'id' and containing column 'typo'")

    baseline_parser.add_argument("-cte", "--cand_test_file", type=str,
                                 help="File of pickled testing candidates dataframe.\
                                 Candidates dataframe has to contain columns \
                                 ('id', 'typo', 'candidate', 'typo_freq', \
                                 'cand_freq', 'distance')")

    baseline_parser.add_argument("-o", "--out_file", type=str,
                                 help="File for printing testing results")

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
