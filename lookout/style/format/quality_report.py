"""Facilities to report the quality of a given model on a given dataset."""
from collections import Counter
import glob

from bblfsh import BblfshClient
import numpy
from sklearn.metrics import classification_report, confusion_matrix

from lookout.style.format.feature_extractor import FeatureExtractor
from lookout.style.format.feature_utils import CLASSES
from lookout.style.format.model import FormatModel
from lookout.style.format.utils import prepare_files, profile


@profile
def quality_report(input_pattern: str, bblfsh: str, language: str, n_files: int, model_path: str
                   ) -> None:
    """Print several different reports for a given model on a given dataset."""
    model = FormatModel().load(model_path)
    rules = model[language]
    print("Model parameters: %s" % rules.origin_config)
    print("Stats about rules: %s" % rules)

    client = BblfshClient(bblfsh)
    filenames = glob.glob(input_pattern, recursive=True)
    files = prepare_files(filenames, client, language)
    print("Number of files: %s" % (len(files)))

    fe = FeatureExtractor(language=language, **rules.origin_config["feature_extractor"])
    res = fe.extract_features(files)

    if res is None:
        print("Failed to parse files, aborting report...")
        return
    X, y, vnodes_y, vnodes = res

    y_pred, _ = rules.predict(X, vnodes_y, vnodes, language)

    target_names = [CLASSES[cls_ind] for cls_ind in numpy.unique(y)]
    print("Classification report:\n" + classification_report(y, y_pred, target_names=target_names))
    print("Confusion matrix:\n" + str(confusion_matrix(y, y_pred)))

    # sort files by mispredictions and print them
    file_mispred = []
    for gt, pred, vn in zip(y, y_pred, vnodes_y):
        if gt != pred:
            file_mispred.append(vn.path)
    file_stat = Counter(file_mispred)

    to_show = file_stat.most_common()
    if n_files > 0:
        to_show = to_show[:n_files]

    print("Files with most errors:\n" + "\n".join(map(str, to_show)))
