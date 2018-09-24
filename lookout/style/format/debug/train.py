"""Facilities to train a FormatModel without a lookout server."""
from os.path import join
from typing import Iterable

from bblfsh.client import BblfshClient
from yaml import safe_load

from lookout.core.analyzer import ReferencePointer
from lookout.core.api.service_data_pb2 import File
from lookout.style.format.analyzer import FormatAnalyzer
from lookout.style.format.debug.utils import prepare_files


class _FakeDataStub:
    def __init__(self, files: Iterable[File]) -> None:
        self.files = files

    def GetFiles(self, _) -> Iterable[File]:
        return self.files


def train(training_dir: str, output_path: str, language: str, bblfsh: str, config: str
          ) -> None:
    """
    Train a FormatModel for debugging purposes.

    :param training_dir: Path to the directory containing the files to train from.
    :param output_path: Path to the model to write.
    :param language: Language to filter on.
    :param bblfsh: Address of the babelfish server.
    :param config: Path to a YAML config to use during the training.
    :return: Trained FormatModel.
    """
    bblfsh_client = BblfshClient(bblfsh)
    if config is not None:
        with open(config) as fh:
            config = safe_load(fh)
    else:
        config = {}
    model = FormatAnalyzer.train(ReferencePointer("someurl", "someref", "somecommit"),
                                 config,
                                 _FakeDataStub(files=prepare_files(join(training_dir, "**", "*"),
                                                                   bblfsh_client,
                                                                   language)))
    model.save(output_path)
