"""Commonly used utils."""
from typing import Any, Dict, Iterable, Sequence
import warnings

from lookout.core.api.service_analyzer_pb2 import Comment
from lookout.core.api.service_data_pb2 import File
import numpy
from sklearn.exceptions import UndefinedMetricWarning
import sklearn.metrics


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


def get_uast_parents(uast: "bblfsh.Node") -> Dict[int, "bblfsh.Node"]:
    """
    Create a mapping from id of the node in the UAST to its parent Node.

    :param uast: UAST to get parents mapping for.
    :return: Parents mapping.
    """
    parents = {}
    queue = [uast]
    while queue:
        node = queue.pop()
        for child in node.children:
            parents[id(child)] = node
        queue.extend(node.children)
    return parents
