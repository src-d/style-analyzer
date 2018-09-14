"""Utilities to check the quality of a model on a given dataset and to visualize its errors."""
from argparse import ArgumentParser
import sys
from typing import Any

from lookout.core.cmdline import ArgumentDefaultsHelpFormatterNoNone
from lookout.style.format.quality_report import quality_report
from lookout.style.format.visualization import visualize


def create_parser() -> ArgumentParser:
    """
    Create a parser for the lookout.style.format utility.

    :return: an ArgumentParser with an handler defined in the handler attribute.
    """
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatterNoNone)
    subparsers = parser.add_subparsers(help="Commands")

    def add_parser(name, help):
        return subparsers.add_parser(
            name, help=help, formatter_class=ArgumentDefaultsHelpFormatterNoNone)

    # Evaluation
    eval_parser = add_parser("eval", "Evaluate trained model on given dataset.")
    eval_parser.set_defaults(handler=quality_report)
    eval_parser.add_argument("-i", "--input-pattern", required=True, type=str,
                             help="Path to folder with source code - "
                                  "should be in a format compatible with glob (ends with**/* "
                                  "and surrounded by quotes. Ex: `path/**/*`).")
    eval_parser.add_argument("--bblfsh", default="0.0.0.0:9432",
                             help="Babelfish server's address.")
    eval_parser.add_argument("-l", "--language", default="javascript",
                             help="Programming language to use.")
    eval_parser.add_argument("-n", "--n-files", default=0, type=int,
                             help="How many files with most mispredictions to show. "
                                  "If n <= 0 show all.")
    eval_parser.add_argument("-m", "--model", required=True, help="Path to saved FormatModel.")

    # Visualization
    vis_parser = add_parser("vis", "Visualize mispredictions of the model on the given file.")
    vis_parser.set_defaults(handler=visualize)
    vis_parser.add_argument("-i", "--input-filename", required=True,
                            help="Path to file to analyze.")
    vis_parser.add_argument("--bblfsh", default="0.0.0.0:9432",
                            help="Babelfish server's address.")
    vis_parser.add_argument("-l", "--language", default="javascript",
                            help="Programming language to use.")
    vis_parser.add_argument("-m", "--model", required=True, help="Path to saved FormatModel.")
    return parser


def main() -> Any:
    """Entry point of the utility."""
    parser = create_parser()
    args = parser.parse_args()
    try:
        handler = args.handler
        delattr(args, "handler")
    except AttributeError:
        def print_usage(_):
            parser.print_usage()

        handler = print_usage
    return handler(**vars(args))


if __name__ == "__main__":
    sys.exit(main())
