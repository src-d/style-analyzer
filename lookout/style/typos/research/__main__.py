import argparse
import sys

from sourced.ml.cmd.args import ArgumentDefaultsHelpFormatterNoNone

from lookout.style.typos.research.baseline import baseline
from lookout.style.typos.research.nn_prediction import DEFAULT_BATCH_SIZE, DEFAULT_DECAY, \
    DEFAULT_LR, DEFAULT_NUM_EPOCHS, DEFAULT_NUM_NEURONS, create_and_train_nn_prediction_from_file
from lookout.style.typos.utils import CORRECT_TOKEN_COLUMN, TYPO_COLUMN


def create_parser():
    parser = argparse.ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatterNoNone,
                                     description='Typos correction tools')
    subparsers = parser.add_subparsers(help="Commands", dest="command")

    def add_parser(name, help_message):
        return subparsers.add_parser(name, help=help_message,
                                     formatter_class=ArgumentDefaultsHelpFormatterNoNone)

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

    nn_parser = add_parser("nn_prediction", "Train nn model for correction embedding prediction.")
    nn_parser.add_argument("--fasttext", type=str, required=True,
                           help="Path to the binary dump of a FastText model.")
    nn_parser.add_argument("--data", type=str, required=True,
                           help="Path to a CSV dump of pandas.DataFrame containing"
                                "columns [%s, %s]" % (CORRECT_TOKEN_COLUMN, TYPO_COLUMN))
    nn_parser.add_argument("--dump", type=str,
                           help="Path to the file where to dump the trained NN model.")
    nn_parser.add_argument("--num-neurons", type=int, help="Number of neurons in each hidden layer.",
                           default=DEFAULT_NUM_NEURONS)
    nn_parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                           help="Batch size for training.")
    nn_parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="Learning rate.")
    nn_parser.add_argument("--decay", type=float, default=DEFAULT_DECAY,
                           help="Learning rate exponential decay per epoch.")
    nn_parser.add_argument("--num-epochs", type=int, default=DEFAULT_NUM_EPOCHS,
                           help="Number of training passes over the dataset.")
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()
    return create_and_train_nn_prediction_from_file(**vars(args))


if __name__ == "__main__":
    sys.exit(main())
