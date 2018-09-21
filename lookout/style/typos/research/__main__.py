import argparse
import sys

from lookout.core.cmdline import ArgumentDefaultsHelpFormatterNoNone
from lookout.style.typos.nn_prediction import create_and_train_nn_prediction_from_file
from lookout.style.typos.utils import CORRECT_TOKEN_COLUMN, TYPO_COLUMN


def create_parser():
    parser = argparse.ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatterNoNone,
                                     description="Train nn model for correction embedding"
                                                 "prediction.")
    parser.add_argument("fasttext_file", type=str,
                        help="Path to binary dump of fasttext model.")
    parser.add_argument("data_file", type=str,
                        help="Path to csv dump of pandas.DataFrame containing"
                             "columns [%s, %s]" % (CORRECT_TOKEN_COLUMN, TYPO_COLUMN))
    parser.add_argument("dump", type=str, help="Path to file to dump trained nn model.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate.")
    parser.add_argument("--decay", type=float, default=0.9, help="Decay.")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs.")
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()
    return create_and_train_nn_prediction_from_file(**vars(args))


if __name__ == "__main__":
    sys.exit(main())
