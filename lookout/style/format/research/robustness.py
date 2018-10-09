from argparse import ArgumentParser
from collections import defaultdict
from difflib import SequenceMatcher
import glob
import os
from typing import Callable, Dict, List, NamedTuple, Set, Tuple

from bblfsh import BblfshClient
import numpy

from lookout.core.slogging import setup
from lookout.style.format.feature_extractor import FeatureExtractor
from lookout.style.format.feature_utils import CLASS_INDEX
from lookout.style.format.files_filtering import filter_filepaths
from lookout.style.format.model import FormatModel
from lookout.style.format.quality_report import prepare_files


Misprediction = NamedTuple("Misprediction", [("y", numpy.ndarray), ("pred", numpy.ndarray),
                                             ("node", List[Callable]), ("rule", numpy.ndarray)])


def get_content_from_repo(folder: str) -> Dict[str, str]:
    content = {}
    filenames = glob.glob(folder, recursive=True)
    for file in filter_filepaths(filenames):
        with open(file) as g:
            content[file] = g.read()
    return content

    
def get_difflib_changes(true_content: Dict[str, str], noisy_content: Dict[str, str]
                        ) -> Tuple[List[str], List[str], Dict[str, Set[int]], int]:
    true_files, noisy_files = set(), set()
    nb_changes = 0
    lines_changed = defaultdict(set)
    for (tf, tc), (nf, nc) in zip(true_content.items(), noisy_content.items()):
        matcher = SequenceMatcher(a=tc.splitlines(), b=nc.splitlines())
        for action, _, _, j1, j2 in matcher.get_opcodes():
            if action is not "equal":
                lines_changed[nf].update(range(j1, j2+1))
                true_files.add(tf)
                noisy_files.add(nf)
                nb_changes += 1
    return sorted(list(true_files)), sorted(list(noisy_files)), lines_changed, nb_changes


def get_mispreds(y: numpy.ndarray, y_pred: numpy.ndarray, nodes: List[Callable],
                 winner: numpy.ndarray) -> List[NamedTuple]:
    mispreds = []
    for gt, pred, vn, rule in zip(y, y_pred, nodes, winner):
        if gt != pred:
            mispreds.append(Misprediction(gt, pred, vn, rule))
    return mispreds


def get_diff_mispreds(mispreds_noise: List[NamedTuple], lines_changed: Dict[str, Set[int]]
                      ) -> Dict[str, NamedTuple]:
    diff_mispreds = {}
    for m in mispreds_noise:
        mispred_lines = set(range(m.node.start.line, m.node.end.line+1))
        if set.intersection(mispred_lines, lines_changed[m.node.path]):
            try:
                if m.node.start.offset < diff_mispreds[m.node.path].node.start.offset:
                    diff_mispreds[m.node.path] = m
            except KeyError:
                diff_mispreds[m.node.path] = m
    return diff_mispreds


def files2mispreds(files: List[str], rules: Callable, client: str, language: str
                   ) -> Tuple[List[NamedTuple], List[Callable]]:
    files = prepare_files(files, client, language)
    fe = FeatureExtractor(language=language, **rules.origin_config["feature_extractor"])
    X, y, vnodes_y, _ = fe.extract_features(files)
    X, _ = fe.select_features(X, y)
    y_pred, winner = rules.predict(X, True)
    mispreds = get_mispreds(y, y_pred, vnodes_y, winner)
    return mispreds, vnodes_y


def get_style_fixes(diff_mispreds: Dict[str, NamedTuple], vnodes: List[Callable],
                    true_files: List[str], noisy_files: List[str]) -> List[NamedTuple]:
    style_fixes = []
    for i in range(len(true_files)):
        try:
            mispred = diff_mispreds[noisy_files[i]]
        except KeyError:
            continue
        for vn in vnodes:
            if vn.path == true_files[i] and vn.start.offset == mispred.node.start.offset:
                if mispred.pred == vn.y :
                    style_fixes.append(mispred)
                break
    return style_fixes


def style_robustness_report(true_repo: str, noisy_repo: str, bblfsh: str, language: str,
                            model: str) -> None:
    client = BblfshClient(bblfsh)

    true_content = get_content_from_repo(true_repo)
    noisy_content = get_content_from_repo(noisy_repo)
    true_files, noisy_files, lines_changed, nb_changes = get_difflib_changes(true_content, noisy_content)

    print()
    print("Number of files modified by adding style noise : %d / %d" % (len(true_files), len(true_content)))
    del true_content, noisy_content
    
    analyzer = FormatModel().load(model)
    rules = analyzer[language]

    _, vnodes_y_true = files2mispreds(true_files, rules, client, language)
    mispreds_noise, _ = files2mispreds(noisy_files, rules, client, language)

    diff_mispreds = get_diff_mispreds(mispreds_noise, lines_changed)
    print("Number of artificial mistakes potentially fixed by the model (diff of mispredictions): %d / %d"
        % (len(diff_mispreds), nb_changes))
    
    style_fixes = get_style_fixes(diff_mispreds, vnodes_y_true, true_files, noisy_files)
    
    print("style-analyzer fixes in the noisy repos : %d / %d -> %.1f %%"
        % (len(style_fixes), nb_changes, 100 *len(style_fixes) / nb_changes))
    print()
    print("list of files where the style-analyzer succeeds in fixing the random noise :")
    for mispred in style_fixes:
        print(mispred.node.path)


def main():
    setup("DEBUG", False)
    parser = ArgumentParser()
    parser.add_argument("true_repo", type=str,
                        help="Path to the directory containing the files of the true repository.")
    parser.add_argument("noisy_repo", type=str,
                        help="Path to the directory containing the files of the true repo "
                             "modified by adding artificial style noise.")
    parser.add_argument("model", help="Path to the model.")
    parser.add_argument("--bblfsh", default="0.0.0.0:9432", help="Address of babelfish server.")
    parser.add_argument("--language", default="javascript", help="Language to filter on.")
    args = parser.parse_args()
    style_robustness_report(**vars(args))


if __name__ == "__main__":
    main()