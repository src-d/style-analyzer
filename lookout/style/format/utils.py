"""Commonly used utils."""
from typing import Any, Dict, Iterable, Sequence
import warnings

from bblfsh import BblfshClient
from bblfsh.client import NonUTF8ContentException
from lookout.core.api.service_analyzer_pb2 import Comment
from lookout.core.api.service_data_pb2 import File
from lookout.core.lib import filter_files_by_path
import numpy
from sklearn.exceptions import UndefinedMetricWarning
import sklearn.metrics
from tqdm import tqdm


warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


class FakeDataStub:
    """Fake data source."""

    def __init__(self, files: Iterable[File]) -> None:
        """
        Initialize FakeDataStub with sequence of files.

        :param files: sequence of files.
        """
        self.files = files

    def GetFiles(self, _) -> Iterable[File]:
        """
        Return sequence of files.

        :param _: noop.
        :return: sequence of files.
        """
        return self.files


def prepare_files(filenames: Iterable[str], client: BblfshClient,
                  language: str) -> Iterable[File]:
    """
    Prepare the given folder for analysis by extracting UASTs and creating the gRPC wrappers.

    :param filenames: List of paths to files to analyze.
    :param client: Babelfish client. Babelfish server should be started accordingly.
    :param language: Language to consider. Will discard the other languages.
    :return: Iterator of File-s with content, uast, path and language set.
    """
    files = []
    for file in tqdm(filter_files_by_path(list(filenames))):
        try:
            res = client.parse(file)
        except NonUTF8ContentException:
            # skip files that can't be parsed because of UTF-8 decoding errors.
            continue
        if res.status == 0 and res.language.lower() == language.lower():
            uast = res.uast
            path = file
            with open(file) as f:
                content = f.read().encode("utf-8")
            files.append(File(content=content, uast=uast, path=path,
                              language=res.language.lower()))
    return files


def prepare_data_stub(input_pattern: str, client: BblfshClient, language: str):
    """
    Prepare the given folder for analysis and mimic DataStub from core.

    :param input_pattern: Path to folder with source code -  should be in a format compatible with
                          glob (ends with **/*  and surrounded by quotes. Ex: `path/**/*`).
    :param client: Babelfish client. Babelfish server should be started accordingly.
    :param language: Language to consider. Will discard the other languages.
    :return: Iterator of File-s with content, uast, path and language set.
    """
    return FakeDataStub(files=prepare_files(input_pattern, client, language))


def generate_comment(filename: str, confidence: int, line: int, text: str) -> Comment:
    """
    Generate comment.

    :param filename: filename.
    :param confidence: confidence of comment. Should be in range [0, 100].
    :param line: line number for comment. Expecting 1-based indexing. If 0 - comment for the whole
                 file.
    :param text: comment text.
    :return: generated comment.
    """
    assert 0 <= confidence <= 100, "Confidence should be in range 0~100 but value is '%s'" % \
                                   confidence
    assert isinstance(line, int), "Line should be integer but it's type is '%s'" % type(line)
    assert 0 <= line, "Expected value >= 0 but got '%s'" % line
    comment = Comment()
    comment.file = filename
    comment.confidence = confidence
    comment.line = line
    comment.text = text
    return comment


def get_classification_report(y_pred: numpy.ndarray, y_true: numpy.ndarray,
                              target_names: Sequence[str]) -> Dict[str, Any]:
    """
    Colllect the main information that is needed for quality report generation.

    :param y_pred: predicted target values.
    :param y_true: true targets.
    :param target_names: labels to names mapping.
    :return: Dictionary with the report information inside.
    """
    have_prediction = y_pred >= 0
    # Predicted Positive Condition Rate calculation
    ppcr = numpy.sum(have_prediction) / have_prediction.shape[0]
    report = sklearn.metrics.classification_report(
        y_true[have_prediction], y_pred[have_prediction], output_dict=True,
        target_names=target_names, labels=list(range(len(target_names))))
    report_full = sklearn.metrics.classification_report(
        y_true, y_pred, output_dict=True,
        target_names=target_names, labels=list(range(len(target_names))))
    confusion_matrix = sklearn.metrics.confusion_matrix(y_true, y_pred)
    return {"ppcr": ppcr,
            "report": report,
            "report_full": report_full,
            "confusion_matrix": confusion_matrix,
            "target_names": target_names}
