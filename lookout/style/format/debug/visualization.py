"""Utilities to visualize the errors made on a file."""
from collections import namedtuple

from bblfsh import BblfshClient

from lookout.style.format.features import CLASSES, FeatureExtractor
from lookout.style.format.model import FormatModel
from lookout.style.format.debug.utils import prepare_file


RED = "\033[41m"
GREEN = "\033[42m"
ENDC = '\033[m'


Misprediction = namedtuple("Misprediction", ["y", "pred", "node"])


def visualize(input_filename: str, bblfsh: str, language: str, model: str) -> None:
    """Visualize the errors made on a single file."""
    client = BblfshClient(bblfsh)
    file = prepare_file(input_filename, client, language)

    fe = FeatureExtractor(language=language)
    X, y, nodes = fe.extract_features([file])

    analyzer = FormatModel().load(model)
    rules = analyzer._rules_by_lang[language]
    y_pred = rules.predict(X)

    mispred = []
    for gt, pred, node in zip(y, y_pred, nodes):
        if gt != pred:
            mispred.append(Misprediction(gt, pred, node))
    print("Errors: %s out of %s mispredicted" % (len(mispred), len(nodes)))

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

        new_content += GREEN + CLASSES[wrong.y] + RED + CLASSES[wrong.pred] + ENDC

        if i == len(mispred) - 1 and end != len(old_content):
            new_content += old_content[end:]
        else:
            new_content += old_content[end:mispred[i + 1].node.start.offset]

    print("Visualization:\n" + new_content)
