"""Utilities to visualize the errors made on a file."""
from collections import defaultdict
from difflib import SequenceMatcher
import os

from bblfsh import BblfshClient
from lookout.core.api.service_data_pb2 import File

from lookout.style.format.classes import CLASSES
from lookout.style.format.feature_extractor import FeatureExtractor
from lookout.style.format.model import FormatModel
from lookout.style.format.robustness import Misprediction

RED = "\033[41m"
GREEN = "\033[42m"
BLUE = "\033[94m"
ENDC = "\033[m"


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
    print("Model parameters: %s" % rules.origin)
    print("Stats about rules: %s" % rules)

    client = BblfshClient(bblfsh)
    file = prepare_file(input_filename, client, language)

    fe = FeatureExtractor(language=language, **rules.origin_config["feature_extractor"])
    X, y, vnodes_y, vnodes = fe.extract_features([file])

    y_pred, _, _ = rules.predict(X, vnodes_y, vnodes, fe)

    # collect lines with mispredictions - could be removed
    mispred_lines = set()
    lines = set()
    for gt, pred, node in zip(y, y_pred, vnodes_y):
        lines.add((node.path, node.start.line))
        if gt != pred:
            mispred_lines.add((node.path, node.start.line))
    print("Number of lines with mispredictions: %s out of %s mispredicted" %
          (len(mispred_lines), len(lines)))

    # collect mispredictions and all other predictions for each line with mistake
    mispred = defaultdict(list)
    for gt, pred, node in zip(y, y_pred, vnodes_y):
        if (node.path, node.start.line) in mispred_lines:
            mispred[(node.path, node.start.line)].append(Misprediction(gt, pred, node))

    # sort each line
    for value in mispred.values():
        value.sort(key=lambda k: k.node.start.offset)

    # final mispredictions
    final_mispred = []
    for line in sorted(mispred):
        gt = [m.y for m in mispred[line]]
        pred = [m.pred for m in mispred[line]]
        s = SequenceMatcher(None, gt, pred)
        blocks = s.get_matching_blocks()

        if blocks[0].a != 0:
            # mispredictions before the first matching block
            final_mispred.extend(mispred[line][:blocks[0].a])
        for i in range(len(blocks) - 1):
            final_mispred.extend(mispred[line][blocks[i].a:blocks[i + 1].a])
        if blocks[-1].a != len(mispred[line]):
            # mispredictions after the last matching block
            final_mispred.extend(mispred[line][blocks[-1].a:])

    mispred = sorted([misp for misp in final_mispred if misp.y != misp.pred],
                     key=lambda r: r.node.start.offset)

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

        if i == len(mispred) - 1:
            if end != len(old_content):
                new_content += old_content[end:]
        else:
            new_content += old_content[end:mispred[i + 1].node.start.offset]
    print("Visualization:\n" + new_content)
