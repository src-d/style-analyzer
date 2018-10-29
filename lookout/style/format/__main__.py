"""Utilities to check the quality of a model on a given dataset and to visualize its errors."""
from argparse import ArgumentParser
import sys
from typing import Any

from lookout.core.cmdline import ArgumentDefaultsHelpFormatterNoNone
from lookout.core.slogging import setup as setup_slogging
from lookout.style.format.benchmarks.evaluate_smoke import evaluate_smoke_entry
from lookout.style.format.benchmarks.generate_smoke import generate_smoke_entry
from lookout.style.format.quality_report import quality_report
from lookout.style.format.robustness import plot_pr_curve, style_robustness_report
from lookout.style.format.rule_stat import print_rules_report
from lookout.style.format.visualization import visualize


def add_input_pattern_arg(my_parser: ArgumentParser):
    """
    Add an input pattern argument to an argpase parser.

    :param my_parser: Parser to add the argument to.
    """
    my_parser.add_argument(
        "-i", "--input-pattern", required=True, type=str,
        help="Path to folder with source code - "
             "should be in a format compatible with glob (ends with**/* "
             "and surrounded by quotes. Ex: `path/**/*`).")


def add_model_args(my_parser: ArgumentParser):
    """
    Add arguments to specify model path and language.

    :param my_parser: Parser to add the arguments to.
    """
    my_parser.add_argument(
        "-m", "--model-path", required=True,
        help="Path to the saved FormatModel.")
    my_parser.add_argument(
        "-l", "--language", default="javascript",
        help="Programming language to use.")


def add_bblfsh_arg(my_parser: ArgumentParser):
    """
    Add an argument to specify the babelfish server connection address.

    :param my_parser: Parser to add the argument to.
    """
    my_parser.add_argument(
        "--bblfsh", default="0.0.0.0:9432",
        help="Babelfish server's address.")


def add_true_noisy_repos_args(my_parser: ArgumentParser):
    """
    Add arguments to specify the path of true and noisy repositories.

    :param my_parser: Parser to add the arguments to.
    """
    my_parser.add_argument(
        "--true-repo", required=True, type=str,
        help="Path to the directory containing the files of the true repository.")
    my_parser.add_argument(
        "--noisy-repo", required=True, type=str,
        help="Path to the directory containing the files of the true repo "
             "modified by adding artificial style mistakes.")


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
    add_input_pattern_arg(eval_parser)
    add_bblfsh_arg(eval_parser)
    add_model_args(eval_parser)
    eval_parser.add_argument("-n", "--n-files", default=0, type=int,
                             help="How many files with most mispredictions to show. "
                                  "If n <= 0 show all.")

    # Visualization
    vis_parser = add_parser("vis", "Visualize mispredictions of the model on the given file.")
    vis_parser.set_defaults(handler=visualize)
    vis_parser.add_argument("-i", "--input-filename", required=True,
                            help="Path to file to analyze.")
    add_bblfsh_arg(vis_parser)
    add_model_args(vis_parser)

    # Rules statistics
    rule_parser = add_parser("rule", "Statistics about rules.")
    rule_parser.set_defaults(handler=print_rules_report)
    add_input_pattern_arg(rule_parser)
    add_bblfsh_arg(rule_parser)
    add_model_args(rule_parser)

    # Style robustness quality report, includes precision, recall and F1-score
    robust_parser = add_parser("robust-eval", "Quality report made by analyzing how well the "
                                              "is able to fix random style mistakes among a model "
                                              "repository: includes precision, recall and "
                                              "F1-score.")
    robust_parser.set_defaults(handler=style_robustness_report)
    add_true_noisy_repos_args(robust_parser)
    add_bblfsh_arg(robust_parser)
    add_model_args(robust_parser)

    # Plot Precision and Recall curves
    pr_curve_parser = add_parser("pr-curve", "Plot Precision/Recall curves with different rules "
                                             "selected based on their confidence.")
    pr_curve_parser.set_defaults(handler=plot_pr_curve)
    add_true_noisy_repos_args(pr_curve_parser)
    add_bblfsh_arg(pr_curve_parser)
    add_model_args(pr_curve_parser)
    pr_curve_parser.add_argument("--support-threshold", type=int, default=0,
                                 help="Support threshold to filter relevant rules.")
    pr_curve_parser.add_argument("-o", "--output", required=True, type=str,
                                 help="Path to the output figure. Could be a png or svg file.")

    # Generate dataset of different styles in code for smoke testing.
    gen_smoke_parser = add_parser("gen-smoke-dataset",
                                  "Generate dataset with different styles. "
                                  "Helps to check the basic system functionality. "
                                  "Only JavaScript code is supported now.")
    gen_smoke_parser.set_defaults(handler=generate_smoke_entry)
    gen_smoke_parser.add_argument(
        "inputpath", type=str,
        help="Path to the tar.xz archive containing initial repositories.")
    gen_smoke_parser.add_argument(
        "outputpath", type=str,
        help="Path to the directory where the generated dataset should be stored.")
    gen_smoke_parser.add_argument(
        "--force", default=False, action="store_true",
        help="Override output directory if exists."
    )

    # Evaluate on different styles dataset
    eval_gen_styles_parser = add_parser("eval-smoke-dataset",
                                        "Evaluate on the dataset with different styles.")
    eval_gen_styles_parser.set_defaults(handler=evaluate_smoke_entry)
    eval_gen_styles_parser.add_argument(
        "inputpath", type=str,
        help="Path to the directory where the generated dataset is stored. "
             "To generate a dataset run gen-smoke-dataset command.")
    eval_gen_styles_parser.add_argument(
        "reportpath", type=str,
        help="Path for report performance output.")

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


if __name__ == "__main__":
    sys.exit(main())
