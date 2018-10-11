"""Utilities to check the quality of a model on a given dataset and to visualize its errors."""
from argparse import ArgumentParser
import sys
from typing import Any

from lookout.core.cmdline import ArgumentDefaultsHelpFormatterNoNone
from lookout.core.slogging import setup as setup_slogging
from lookout.style.format.quality_report import quality_report
from lookout.style.format.robustness import style_robustness_report
from lookout.style.format.rule_stat import print_rules_report
from lookout.style.format.visualization import visualize


def create_parser() -> ArgumentParser:
    """
    Create a parser for the lookout.style.format utility.

    :return: an ArgumentParser with an handler defined in the handler attribute.
    """
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatterNoNone)

    # General options
    parser.add("--log-level", default="DEBUG", help="Log verbosity level.")

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
    eval_parser.add_argument("-m", "--model-path", required=True,
                             help="Path to saved FormatModel.")

    # Visualization
    vis_parser = add_parser("vis", "Visualize mispredictions of the model on the given file.")
    vis_parser.set_defaults(handler=visualize)
    vis_parser.add_argument("-i", "--input-filename", required=True,
                            help="Path to file to analyze.")
    vis_parser.add_argument("--bblfsh", default="0.0.0.0:9432",
                            help="Babelfish server's address.")
    vis_parser.add_argument("-l", "--language", default="javascript",
                            help="Programming language to use.")
    vis_parser.add_argument("-m", "--model-path", required=True, help="Path to saved FormatModel.")

    # Rules statistics
    rule_parser = add_parser("rule", "Statistics about rules.")
    rule_parser.set_defaults(handler=print_rules_report)
    rule_parser.add_argument("-i", "--input-pattern", required=True, type=str,
                             help="Path to folder with source code - "
                                  "should be in a format compatible with glob (ends with**/* "
                                  "and surrounded by quotes. Ex: `path/**/*`).")
    rule_parser.add_argument("--bblfsh", default="0.0.0.0:9432",
                             help="Babelfish server's address.")
    rule_parser.add_argument("-l", "--language", default="javascript",
                             help="Programming language to use.")
    rule_parser.add_argument("-m", "--model-path", required=True,
                             help="Path to saved FormatModel.")

    # Style robustness quality report
    robust_parser = add_parser("robust", "Quality report made by analyzing how well the model is"
                                         " able to fix random style mistakes among a repository.")
    robust_parser.set_defaults(handler=style_robustness_report)
    robust_parser.add_argument("--true-repo", required=True, type=str,
                               help="Path to the directory containing the files of the true "
                                    "repository.")
    robust_parser.add_argument("--noisy-repo", required=True, type=str,
                               help="Path to the directory containing the files of the true repo "
                                    "modified by adding artificial style mistakes.")
    robust_parser.add_argument("-m", "--model-path", required=True, help="Path to the model.")
    robust_parser.add_argument("--bblfsh", default="0.0.0.0:9432",
                               help="Babelfish server's address.")
    robust_parser.add_argument("-l", "--language", default="javascript",
                               help="Programming language to use.")
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
        def print_usage(_):
            parser.print_usage()

        handler = print_usage
    return handler(**vars(args))


if __name__ == "__main__":
    sys.exit(main())
