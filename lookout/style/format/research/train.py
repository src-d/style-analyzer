"""Facilities to train a FormatModel without a lookout server."""
from argparse import ArgumentParser
import glob
import logging
from os.path import join
from typing import Iterable, Optional

from bblfsh.client import BblfshClient
from lookout.core.analyzer import ReferencePointer
from lookout.core.api.service_data_pb2 import Change, File
from lookout.core.lib import filter_files
from lookout.core.slogging import setup
from yaml import safe_load

from lookout.style.format.analyzer import FormatAnalyzer


class FakeDataService:
    """Fake data service to replace lookout one."""

    def __init__(self, bblfsh_client: BblfshClient, files: Optional[Iterable[File]],
                 changes: Optional[Iterable[Change]]) -> None:
        """Construct a fake data service."""
        self.data_stub = FakeDataStub(files, changes)
        self.bblfsh_client = bblfsh_client

    def get_data(self):
        return self.data_stub

    def get_bblfsh(self):
        return self.bblfsh_client._stub


class FakeDataStub:
    """Fake data stub to replace lookout one."""

    def __init__(self, files: Optional[Iterable[File]], changes: Optional[Iterable[Change]]):
        """Construct a fake data stub."""
        self.files = files
        self.changes = changes

    def GetFiles(self, _):
        return self.files

    def GetChanges(self, _):
        return self.changes


def train(training_dir: str, output_path: str, language: str, bblfsh: str, config: str,
          ) -> None:
    """
    Train a FormatModel for debugging purposes.

    :param training_dir: Path to the directory containing the files to train from.
    :param output_path: Path to the model to write.
    :param language: Language to filter on.
    :param bblfsh: Address of the babelfish server.
    :param config: Path to a YAML config to use during the training.
    """
    log = logging.getLogger("train")
    bblfsh_client = BblfshClient(bblfsh)
    if config is not None:
        with open(config) as fh:
            config = safe_load(fh)
    else:
        config = FormatAnalyzer.defaults_for_train
    filenames = glob.glob(join(training_dir, "**", "*"), recursive=True)
    model = FormatAnalyzer.train(
        ReferencePointer("someurl", "someref", "somecommit"),
        config,
        FakeDataService(bblfsh_client, filter_files(filenames=filenames,
                                                    line_length_limit=config["global"]["line_length_limit"],
                                                    client=bblfsh_client, language=language, log=log), None)
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
