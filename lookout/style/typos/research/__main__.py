import argparse
import logging
import sys

from lookout.core import slogging
from lookout.core.cmdline import ArgumentDefaultsHelpFormatterNoNone

from lookout.style.typos.research.nn_prediction import DEFAULT_BATCH_SIZE, DEFAULT_DECAY, \
    DEFAULT_LR, DEFAULT_NUM_EPOCHS, DEFAULT_NUM_NEURONS, create_and_train_nn_prediction_from_file
from lookout.style.typos.utils import Columns


def create_parser():
    parser = argparse.ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatterNoNone,
                                     description="Train nn model for correction embedding"
                                                 "prediction.")
    parser.add_argument("--fasttext", type=str, required=True,
                        help="Path to the binary dump of a FastText model.")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to a CSV dump of pandas.DataFrame containing"
                             "columns [%s, %s]" % (Columns.CorrectToken, Columns.Token))
    parser.add_argument("--dump", type=str,
                        help="Path to the file where to dump the trained NN model.")
    parser.add_argument("--num-neurons", type=int, help="Number of neurons in each hidden layer.",
                        default=DEFAULT_NUM_NEURONS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="Learning rate.")
    parser.add_argument("--decay", type=float, default=DEFAULT_DECAY,
                        help="Learning rate exponential decay per epoch.")
    parser.add_argument("--num-epochs", type=int, default=DEFAULT_NUM_EPOCHS,
                        help="Number of training passes over the dataset.")
    parser.add_argument("--log-level", default="INFO", choices=logging._nameToLevel,
                        help="Logging verbosity.")
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()
    slogging.setup(args.log_level, False)
    return create_and_train_nn_prediction_from_file(**vars(args))


if __name__ == "__main__":
    sys.exit(main())
