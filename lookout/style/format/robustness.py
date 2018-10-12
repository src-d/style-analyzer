"""Evaluate how well a given model is able to fix style mistakes randomly added in a repository."""
from collections import defaultdict
from difflib import SequenceMatcher
import glob
from typing import Iterable, List, Mapping, NamedTuple, Set, Tuple

from bblfsh import BblfshClient
import numpy

from lookout.style.format.feature_extractor import FeatureExtractor
from lookout.style.format.feature_utils import VirtualNode
from lookout.style.format.files_filtering import filter_filepaths
from lookout.style.format.model import FormatModel
from lookout.style.format.quality_report import prepare_files
from lookout.style.format.rules import Rules


Misprediction = NamedTuple("Misprediction", [("y", numpy.ndarray), ("pred", numpy.ndarray),
                                             ("node", List[VirtualNode]), ("rule", numpy.ndarray)])


def get_content_from_repo(folder: str) -> Mapping[str, str]:
    """
    Extract the content of the files given their path.

    :param folder: Path to the files to get the content from.
    :return: Dictionary where the key is the path to a file and its value the content of the file.
    """
    content = {}
    filenames = glob.glob(folder, recursive=True)
    for file in filter_filepaths(filenames):
        with open(file) as g:
            content[file] = g.read()
    return content


def get_difflib_changes(true_content: Mapping[str, str], noisy_content: Mapping[str, str]
                        ) -> Tuple[Iterable[str], Iterable[str], Mapping[str, Set[int]], int]:
    """
    Return the files and lines that have been modified.

    Given 2 contents of one repository (the original and its noisy version), returns the list files
    that have been modified, the lines that have changed.

    :param true_content: Dictionary containing the content of the original repository.
    :param noisy_content: Dictionary containing the content of the noisy version of the repository.
    :return: The list of files where a style mistake has been added, and the mirror list of the
             original files, and the dictionary of the sets of lines modified by file.
             The number of lines that have been modified, must be equal to the number
             of modified files.
    """
    true_files, noisy_files = set(), set()
    lines_changed = defaultdict(set)
    for (tf, tc), (nf, nc) in zip(true_content.items(), noisy_content.items()):
        matcher = SequenceMatcher(a=tc.splitlines(), b=nc.splitlines())
        for action, _, _, j1, j2 in matcher.get_opcodes():
            if action is not "equal":
                lines_changed[nf].update(range(j1, j2+1))
                true_files.add(tf)
                noisy_files.add(nf)
    return sorted(true_files), sorted(noisy_files), lines_changed


def files2vnodes(files: Iterable[str], rules: Rules, client: str, language: str
                 ) -> Iterable[VirtualNode]:
    """
    Return the `VirtualNodes` extracted from a list of files.

    :param files: List of files to get `Mispredictions` and `VirtualNodes` from.
    :param rules: rules of the style-analyzer model.
    :param client: Babelfish client. Babelfish server should be started accordingly.
    :param language: Language to consider, others will be discarded.
    :return: List of `VirtualNodes` extracted from a given list of files.
    """
    files = prepare_files(files, client, language)
    fe = FeatureExtractor(language=language, **rules.origin_config["feature_extractor"])
    X, y, vnodes_y, _ = fe.extract_features(files)
    return vnodes_y


def files2mispreds(files: Iterable[str], rules: Rules, client: str, language: str
                   ) -> Iterable[Misprediction]:
    """
    Return the model's `Mispredictions` on a list of files.

    :param files: List of files to get `Mispredictions` from.
    :param rules: rules of the style-analyzer model.
    :param client: Babelfish client. Babelfish server should be started accordingly.
    :param language: Language to consider, others will be discarded.
    :return: List of `Mispredictions` extracted from a given list of files.
    """
    files = prepare_files(files, client, language)
    fe = FeatureExtractor(language=language, **rules.origin_config["feature_extractor"])
    X, y, vnodes_y, _ = fe.extract_features(files)
    X, _ = fe.select_features(X, y)
    y_pred, winner = rules.predict(X, True)
    mispreds = get_mispreds(y, y_pred, vnodes_y, winner)
    return mispreds


def get_mispreds(y: numpy.ndarray, y_pred: numpy.ndarray, nodes: Iterable[VirtualNode],
                 winner: numpy.ndarray) -> Iterable[Misprediction]:
    """
    Return the list of `Mispredictions` where the labels differ.

    :param y: Numpy 1-dimensional array of labels.
    :param y_pred: Numpy 1-dimensional array of predicted labels by the model.
    :param nodes: List of `VirtualNodes`.
    :param winner: Numpy 1-dimensional array of the winning rule indices for each sample.
    :return: List of `Mispredictions` where the labels `y` and `y_pred` differ.
    """
    mispreds = []
    for gt, pred, vn, rule in zip(y, y_pred, nodes, winner):
        if gt != pred:
            mispreds.append(Misprediction(gt, pred, vn, rule))
    return mispreds


def get_diff_mispreds(mispreds: Iterable[Misprediction], lines_changed: Mapping[str, Set[int]]
                      ) -> Mapping[str, Misprediction]:
    """
    Filter `Mispredictions` to select those involving at least one line that has been modified.

    :param mispreds: List of `Mispredictions` to filter.
    :param lines_changed: Dict of lines that have been changed when adding random noise.
    :return: Dictionary of the `Mispredictions` involving at least one line
             that has been modified when adding random noise.
    """
    diff_mispreds = {}
    for m in mispreds:
        mispred_lines = set(range(m.node.start.line, m.node.end.line + 1))
        if set.intersection(mispred_lines, lines_changed[m.node.path]):
            try:
                if m.node.start.offset < diff_mispreds[m.node.path].node.start.offset:
                    diff_mispreds[m.node.path] = m
            except KeyError:
                diff_mispreds[m.node.path] = m
    return diff_mispreds


def get_style_fixes(mispreds: Mapping[str, Misprediction], vnodes: Iterable[VirtualNode],
                    true_files: Iterable[str], noisy_files: Iterable[str]
                    ) -> Iterable[Misprediction]:
    """
    Return `Mispredictions` that fix the style mistakes added.

    Given a list of `Mispredictions` potentially fixing a style mistake added since involving
    at least one line that has been modified, return the list of `Mispredicitons` really fixing
    that mistake because their prediction on the noisy files would match the ground truth
    labels of the original files.

    :param mispreds: Dictionary of `Mispredictions` potentially fixing a style mistake.
    :param vnodes: List of `VirtualNodes` extracted from the list of `true_files`.
    :param true_files: list of files of the original repos where a style mistake has been added.
    :param noisy_files: list of files from the noisy repos where a modification has been made
    :return: List of `Mispredictions` where the prediction on a noisy file matches the ground truth
             label of the original file i.e. `Mispredictions` actually fixing the mistakes added.
    """
    style_fixes = []
    for true_file, noisy_file in zip(true_files, noisy_files):
        try:
            mispred = mispreds[noisy_file]
        except KeyError:
            continue
        for vn in vnodes:
            if vn.path == true_file and vn.start.offset == mispred.node.start.offset:
                if mispred.pred == vn.y:
                    style_fixes.append(mispred)
                break
    return style_fixes


def style_robustness_report(true_repo: str, noisy_repo: str, bblfsh: str, language: str,
                            model_path: str) -> None:
    """
    Print the quality report of a model tested on a given repository.

    The tests consists in adding random style mistakes in the given repo and looking how well
    the model is able to fix them according to the style of the original repository.

    :param true_repo: Path to the original repository we want to test the model on.
    :param noisy_repo: Path to the noisy version of the repository where 1 style mistake is
           randomly added in every file.
    :param bblfsh: Babelfish client. Babelfish server should be started accordingly.
    :param language: Language to consider, others will be discarded.
    :param model_path: Path to the model to test. It should be previously trained on the original
           repository located in ':param true_repo:'.
    """
    client = BblfshClient(bblfsh)

    true_content = get_content_from_repo(true_repo)
    noisy_content = get_content_from_repo(noisy_repo)
    print("len true content :", len(true_content))
    true_files, noisy_files, lines_changed = get_difflib_changes(true_content, noisy_content)

    print()
    print("Number of files modified by adding style noise : %d / %d"
          % (len(true_files), len(true_content)))
    del true_content, noisy_content

    analyzer = FormatModel().load(model_path)
    rules = analyzer[language]

    vnodes_y_true = files2vnodes(true_files, rules, client, language)
    mispreds_noise = files2mispreds(noisy_files, rules, client, language)

    diff_mispreds = get_diff_mispreds(mispreds_noise, lines_changed)
    changes_count = len(lines_changed)
    print("Number of artificial mistakes potentially fixed by the model"
          "(diff of mispredictions): %d / %d" % (len(diff_mispreds), changes_count))

    style_fixes = get_style_fixes(diff_mispreds, vnodes_y_true, true_files, noisy_files)

    print("style-analyzer fixes in the noisy repos : %d / %d -> %.1f %%"
          % (len(style_fixes), changes_count, 100 * len(style_fixes) / changes_count))

    true_positive = len(style_fixes)
    false_positive = len(diff_mispreds) - len(style_fixes)
    false_negative = changes_count - len(diff_mispreds)
    try:
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
    except ZeroDivisionError:
        precision = 0
        recall = 0

    print("precision :", round(precision, 3))
    print("recall :", round(recall, 3))

    print()
    print("list of files where the style-analyzer succeeds in fixing the random noise :")
    for mispred in style_fixes:
        print(mispred.node.path)
