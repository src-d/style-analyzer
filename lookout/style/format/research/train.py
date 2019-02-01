"""Facilities to train a FormatModel without a lookout server."""
from argparse import ArgumentParser

from lookout.core.analyzer import ReferencePointer
from lookout.core.slogging import setup

from lookout.style.format.benchmarks.quality_report_noisy import train


def main():
    setup("DEBUG", False)
    parser = ArgumentParser()
    parser.add_argument("training_dir",
                        help="Path to the directory containing the files to train from.")
    parser.add_argument("output_path", help="Path to the model to write.")
    parser.add_argument("--bblfsh", default="0.0.0.0:9432", help="Address of babelfish server.")
    parser.add_argument("--language", default="javascript", help="Language to filter on.")
    parser.add_argument("--config",
                        help="Path to a YAML file containing config to apply during training.")
    args = parser.parse_args()

    kwargs = vars(args)
    kwargs["ref"] = ReferencePointer(kwargs["training_dir"], "HEAD", "<unknown>")
    train(**kwargs)


if __name__ == "__main__":
    main()
