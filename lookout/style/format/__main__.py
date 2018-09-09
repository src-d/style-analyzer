import argparse
import sys

from lookout.core.cmdline import ArgumentDefaultsHelpFormatterNoNone
from lookout.style.format.quality_report import quality_report
from lookout.style.format.visualization import visualize


def create_parser():
    parser = argparse.ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatterNoNone)
    subparsers = parser.add_subparsers(help="Commands", dest="command")

    def add_parser(name, help):
        return subparsers.add_parser(
            name, help=help, formatter_class=ArgumentDefaultsHelpFormatterNoNone)

    # Evaluation
    eval_parser = add_parser("eval", "Evaluate trained model on given dataset.")
    eval_parser.set_defaults(handler=quality_report)
    eval_parser.add_argument("-i", "--input", required=True, type=str,
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
    vis_parser.add_argument("-i", "--input", required=True, help="Path to folder with source "
                                                                 "code.")
    vis_parser.add_argument("--bblfsh", default="0.0.0.0:9432",
                            help="Babelfish server's address.")
    vis_parser.add_argument("-l", "--language", default="javascript",
                            help="Programming language to use.")
    vis_parser.add_argument("-m", "--model", required=True, help="Path to saved FormatModel.")
    return parser


def main():
    parser = create_parser()
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
