"""Facilities to report the quality of a given model on a given dataset."""
from collections import Counter

from bblfsh import BblfshClient
import numpy
from sklearn.metrics import classification_report, confusion_matrix

from lookout.style.format.debug.utils import prepare_files
from lookout.style.format.features import FeatureExtractor, CLASSES
from lookout.style.format.model import FormatModel


def quality_report(input_pattern: str, bblfsh: str, language: str, n_files: int, model: str
                   ) -> None:
    """Print several different reports for a given model on a given dataset."""
    client = BblfshClient(bblfsh)
    files = prepare_files(input_pattern, client, language)
    print("Number of files: %s" % (len(files)))

    fe = FeatureExtractor(language=language)
    X, y, nodes = fe.extract_features(files)

    analyzer = FormatModel().load(model)
    rules = analyzer._rules_by_lang[language]
    y_pred = rules.predict(X)

    target_names = [CLASSES[cls_ind] for cls_ind in numpy.unique(y)]
    print("Classification report:\n" + classification_report(y, y_pred, target_names=target_names))
    print("Confusion matrix:\n" + str(confusion_matrix(y, y_pred)))

    # sort files by mispredictions and print them
    file_mispred = []
    for gt, pred, vn in zip(y, y_pred, nodes):
        if gt != pred:
            file_mispred.append(vn.path)
    file_stat = Counter(file_mispred)

    to_show = file_stat.most_common()
    if n_files > 0:
        to_show = to_show[:n_files]

    print("Files with most errors:\n" + "\n".join(map(str, to_show)))
