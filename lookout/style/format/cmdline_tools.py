"""
Command line utilities to check the quality of a model on a given dataset, visualize errors, etc.
"""
from argparse import ArgumentParser
import json
from typing import Any

from lookout.core.cmdline import ArgumentDefaultsHelpFormatterNoNone
from modelforge import slogging

from lookout.style.format.descriptions import dump_rule
from lookout.style.format.model import FormatModel


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
        "--bblfsh", default="localhost:9432",
        help="Babelfish server's address.")


def add_rules_thresholds(my_parser: ArgumentParser):
    """
    Add threshold arguments to filter rules.

    :param my_parser: Parser to add the arguments to.
    """
    # This default values also used in `make report-noisy`.
    # You should think twice before changing it.
    my_parser.add_argument("--confidence-threshold", type=float, default=0,
                           help="Confidence threshold to filter relevant rules.")
    my_parser.add_argument("--support-threshold", type=int, default=80,
                           help="Support threshold to filter relevant rules.")


def add_config_arg(my_parser: ArgumentParser) -> None:
    """
    Add analyzer config argument.

    :param my_parser: Parser to add the arguments to.
    """
    my_parser.add_argument(
        "--config", type=json.loads, default="{}",
        help="Config for analyzer in json format.")


def dump_rule_entry(model, hash):
    """Command-line entry for "tool rule"."""
    model = FormatModel().load(model)
    dump_rule(model, hash)


def create_parser() -> ArgumentParser:
    """
    Create a parser for the lookout.style.format utility.

    :return: an ArgumentParser with an handler defined in the handler attribute.
    """
    # Deferred imports to speed up loading __init__
    from lookout.style.format.benchmarks.compare_quality_reports import \
        compare_quality_reports_entry
    from lookout.style.format.benchmarks.evaluate_smoke import evaluate_smoke_entry
    from lookout.style.format.benchmarks.generate_smoke import generate_smoke_entry
    from lookout.style.format.benchmarks.quality_report import generate_quality_report
    from lookout.style.format.benchmarks.general_report import print_reports
    from lookout.style.format.benchmarks.quality_report_noisy import quality_report_noisy
    from lookout.style.format.benchmarks.expected_vnodes_number import \
        calc_expected_vnodes_number_entry

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatterNoNone)

    # General options
    slogging.add_logging_args(parser)
    subparsers = parser.add_subparsers(help="Commands")

    def add_parser(name, help):
        return subparsers.add_parser(
            name, help=help, formatter_class=ArgumentDefaultsHelpFormatterNoNone)

    # Evaluation
    eval_parser = add_parser("eval", "Evaluate trained model on given dataset.")
    eval_parser.set_defaults(handler=print_reports)
    add_input_pattern_arg(eval_parser)
    add_bblfsh_arg(eval_parser)
    add_model_args(eval_parser)
    eval_parser.add_argument("-n", "--n-files", default=0, type=int,
                             help="How many files with most mispredictions to show. "
                                  "If n <= 0 show all.")

    # Generate quality report for the given data
    quality_report_parser = add_parser("quality-report",
                                       "Generate quality report on a given data.")
    quality_report_parser.set_defaults(handler=generate_quality_report)
    add_config_arg(quality_report_parser)
    quality_report_parser.add_argument(
        "-i", "--input", required=True,
        help="csv file with repositories to make report. Should contain url, to and from columns.")
    quality_report_parser.add_argument(
        "-o", "--output", required=True,
        help="Directory where to save results.")
    quality_report_parser.add_argument(
        "-f", "--force", default=False, action="store_true",
        help="Force to overwrite results stored in output directory if True. \
                 Stored results will be used if False.")
    quality_report_parser.add_argument(
        "-b", "--bblfsh", help="Bblfsh address to use.")
    quality_report_parser.add_argument(
        "--database", default=None, help="sqlite3 database path to store the models."
                                         "Temporary file is used if not set.")
    quality_report_parser.add_argument(
        "--fs", default=None, help="Model repository file system root. "
                                   "Temporary directory is used if not set.")

    # Generate the quality report based on the artificial noisy dataset
    quality_report_noisy_parser = add_parser("quality-report-noisy", "Quality report on the "
                                                                     "artificial noisy dataset")
    quality_report_noisy_parser.set_defaults(handler=quality_report_noisy)
    add_config_arg(quality_report_noisy_parser)
    add_bblfsh_arg(quality_report_noisy_parser)
    add_rules_thresholds(quality_report_noisy_parser)
    quality_report_noisy_parser.add_argument(
        "-l", "--language", default="javascript",
        help="Programming language to use.")
    quality_report_noisy_parser.add_argument(
        "--repos", type=str,
        help="list of urls or paths to the repositories to analyze. Should be strings separated "
             "by newlines.")
    quality_report_noisy_parser.add_argument(
        "--precision-threshold", type=float, default=0.95,
        help="Precision threshold tolerated for the model.")
    quality_report_noisy_parser.add_argument(
        "-o", "--dir-output", required=True, type=str,
        help="Path to the output directory where to store the quality report and the "
             "precision-recall curve.")

    # Compare two quality reports summaries
    compare_quality_parser = add_parser(
        "compare-quality",
        "Creates a file with the differences in quality metrics between two reports.")
    compare_quality_parser.set_defaults(handler=compare_quality_reports_entry)
    compare_quality_parser.add_argument(
        "--base", type=str, required=True,
        help="Baseline report. Usually the latest report from ./report/ directory.")
    compare_quality_parser.add_argument(
        "--new", type=str, required=True,
        help="New report. Usually It is a report generated for master or any local \
                       change you did and want to validate.")
    compare_quality_parser.add_argument(
        "-o", "--output", type=str, required=True,
        help="Path to the file to save result or - to print to stdout.")

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
        help="Override output directory if exists.")

    # Evaluate on different styles dataset
    eval_smoke_parser = add_parser("eval-smoke-dataset",
                                   "Evaluate on the dataset with different styles.")
    eval_smoke_parser.set_defaults(handler=evaluate_smoke_entry)
    add_config_arg(eval_smoke_parser)
    eval_smoke_parser.add_argument(
        "inputpath", type=str,
        help="Path to the directory where the generated dataset is stored. "
             "To generate a dataset run gen-smoke-dataset command.")
    eval_smoke_parser.add_argument(
        "reportdir", type=str,
        help="Path for report performance output directory.")
    eval_smoke_parser.add_argument(
        "--bblfsh",
        help="Babelfish server's address.")
    eval_smoke_parser.add_argument(
        "--database", type=str, default=None,
        help="Path to the sqlite3 database with trained models metadata. "
             "Enables reusing previously trained models.")

    rule_parser = add_parser("rule", "Print rule description by its hash.")
    rule_parser.set_defaults(handler=dump_rule_entry)
    rule_parser.add_argument("model", help="Path to the model file.")
    rule_parser.add_argument("hash", help="Hash of the rule (8 chars).")

    # FIXME(zurk): remove when https://github.com/src-d/style-analyzer/issues/557 is resolved
    calc_expected_vnodes = add_parser("calc-expected-vnodes-number",
                                      "Write the CSV file with expected numbers of virtual nodes "
                                      "extracted from repositories. Required for quality report "
                                      "generation. It is a workaround for "
                                      "https://github.com/src-d/style-analyzer/issues/557. "
                                      "Docker service is required to be running.")
    calc_expected_vnodes.set_defaults(handler=calc_expected_vnodes_number_entry)
    calc_expected_vnodes.add_argument(
        "-i", "--input", required=True,
        help="CSV file with repositories for quality report."
             "Should contain url, to and from columns.")
    calc_expected_vnodes.add_argument(
        "-o", "--output", required=True, help="Path to a output csv file.")
    calc_expected_vnodes.add_argument(
        "-r", "--runs", default=3, help="Number of repeats to ensure the result correctness.")

    return parser


def main() -> Any:
    """Entry point of the utility."""
    parser = create_parser()
    args = parser.parse_args()
    try:
        handler = args.handler
        delattr(args, "handler")
    except AttributeError:
        def print_usage(*args, **kwargs):
            parser.print_usage()

        handler = print_usage
    return handler(**vars(args))
