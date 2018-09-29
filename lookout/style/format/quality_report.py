"""Facilities to report the quality of a given model on a given dataset."""
from collections import Counter
import glob
from typing import Iterable

from bblfsh import BblfshClient
from bblfsh.client import NonUTF8ContentException
import numpy
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from lookout.core.api.service_data_pb2 import File
from lookout.style.format.feature_extractor import FeatureExtractor
from lookout.style.format.feature_utils import CLASSES
from lookout.style.format.files_filtering import filter_filepaths
from lookout.style.format.model import FormatModel


def prepare_files(folder: str, client: BblfshClient, language: str) -> Iterable[File]:
    """
    Prepare the given folder for analysis by extracting UASTs and creating the gRPC wrappers.

    :param folder: Path to the folder to analyze.
    :param client: Babelfish client. Babelfish server should be started accordingly.
    :param language: Language to consider. Will discard the other languages
    """
    files = []

    # collect filenames with full path
    filenames = glob.glob(folder, recursive=True)

    for file in tqdm(filter_filepaths(filenames)):
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


def quality_report(input_pattern: str, bblfsh: str, language: str, n_files: int, model_path: str
                   ) -> None:
    """Print several different reports for a given model on a given dataset."""
    model = FormatModel().load(model_path)
    rules = model[language]
    print("Model parameters: %s" % rules.origin_config)
    print("Stats about rules: %s" % rules)

    client = BblfshClient(bblfsh)
    files = prepare_files(input_pattern, client, language)
    print("Number of files: %s" % (len(files)))

    fe = FeatureExtractor(language=language, **rules.origin_config["feature_extractor"])
    res = fe.extract_features(files)

    if res is None:
        print("Failed to parse files, aborting report...")
        return
    X, y, vnodes_y, _ = res
    X, _ = fe.select_features(X, y)

    y_pred = rules.predict(X)

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
