"""Facilities to train a FormatModel without a lookout server."""
from argparse import ArgumentParser
import glob
from os.path import join

from bblfsh.client import BblfshClient
from lookout.core.analyzer import ReferencePointer
from lookout.core.slogging import setup
from yaml import safe_load

from lookout.style.format.analyzer import FormatAnalyzer
from lookout.style.format.quality_report import prepare_files
from lookout.style.format.utils import FakeDataStub


def train(training_dir: str, output_path: str, language: str, bblfsh: str, config: str
          ) -> None:
    """
    Train a FormatModel for debugging purposes.

    :param training_dir: Path to the directory containing the files to train from.
    :param output_path: Path to the model to write.
    :param language: Language to filter on.
    :param bblfsh: Address of the babelfish server.
    :param config: Path to a YAML config to use during the training.
    """
    bblfsh_client = BblfshClient(bblfsh)
    if config is not None:
        with open(config) as fh:
            config = safe_load(fh)
    else:
        config = {}
    filenames = glob.glob(join(training_dir, "**", "*"), recursive=True)
    model = FormatAnalyzer.train(
        ReferencePointer("someurl", "someref", "somecommit"),
        config,
        FakeDataStub(prepare_files(filenames, bblfsh_client, language))
    )
    model.save(output_path)


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
    train(**vars(args))


if __name__ == "__main__":
    main()
