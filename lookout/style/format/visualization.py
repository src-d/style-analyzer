"""Utilities to visualize the errors made on a file."""
from collections import namedtuple
import os

from bblfsh import BblfshClient

from lookout.core.api.service_data_pb2 import File
from lookout.style.format.feature_extractor import FeatureExtractor
from lookout.style.format.feature_utils import CLASSES
from lookout.style.format.model import FormatModel

RED = "\033[41m"
GREEN = "\033[42m"
BLUE = "\033[94m"
ENDC = "\033[m"


Misprediction = namedtuple("Misprediction", ["y", "pred", "node", "rule"])


def prepare_file(filename: str, client: BblfshClient, language: str) -> File:
    """
    Prepare the given file for analysis by extracting UAST and creating the gRPC wrapper.

    :param filename: Path to the filename to analyze.
    :param client: Babelfish client. Babelfish server should be started accordingly.
    :param language: Language to consider. Will discard the other languages
    """
    assert os.path.isfile(filename), "\"%s\" should be a file" % filename
    res = client.parse(filename, language)
    assert res.status == 0, "Parse returned status %s for file %s" % (res.status, filename)
    error_log = "Language for % should be %s instead of %s"
    assert res.language.lower() == language.lower(), error_log % (filename, language, res.language)

    with open(filename) as f:
        content = f.read().encode("utf-8")

    return File(content=content, uast=res.uast, path=filename)


def visualize(input_filename: str, bblfsh: str, language: str, model_path: str) -> None:
    """Visualize the errors made on a single file."""
    model = FormatModel().load(model_path)
    rules = model[language]
    print("Model parameters: %s" % rules.origin_config)
    print("Stats about rules: %s" % rules)

    client = BblfshClient(bblfsh)
    file = prepare_file(input_filename, client, language)

    fe = FeatureExtractor(language=language, **rules.origin_config["feature_extractor"])
    res = fe.extract_features([file])

    if res is None:
        print("Failed to parse files, aborting visualization...")
        return
    X, y, vnodes_y = res
    X, _ = fe.select_features(X, y)

    y_pred, winner = rules.predict(X, True)

    mispred = []
    for gt, pred, node, rule in zip(y, y_pred, vnodes_y, winner):
        if gt != pred:
            mispred.append(Misprediction(gt, pred, node, rule))
    print("Errors: %s out of %s mispredicted" % (len(mispred), len(vnodes_y)))

    mispred = sorted(mispred, key=lambda r: r.node.start.offset)

    new_content = ENDC
    old_content = file.content.decode("utf-8")
    for i in range(len(mispred)):
        wrong = mispred[i]
        start = wrong.node.start.offset
        end = wrong.node.end.offset
        if end == start:
            end = start + len(wrong.node.value)

        if i == 0 and start != 0:
            new_content += old_content[:start]
        new_content += GREEN + CLASSES[wrong.y] + RED + CLASSES[wrong.pred] + BLUE + \
            "<rule%s>" % wrong.rule + ENDC

        if i == len(mispred) - 1:
            if end != len(old_content):
                new_content += old_content[end:]
        else:
            new_content += old_content[end:mispred[i + 1].node.start.offset]
    print("Visualization:\n" + new_content)
