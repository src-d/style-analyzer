"""Evaluate how well a given model is able to fix style mistakes randomly added in a repository."""
from collections import defaultdict
from difflib import SequenceMatcher
import glob
import logging
import sys
from typing import Iterable, List, Mapping, NamedTuple, Set, Tuple

from bblfsh import BblfshClient
import numpy

from lookout.style.format.feature_extractor import FeatureExtractor
from lookout.style.format.feature_utils import VirtualNode
from lookout.style.format.files_filtering import filter_filepaths
from lookout.style.format.model import FormatModel
from lookout.style.format.rules import Rules
from lookout.style.format.utils import prepare_files

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
    :return: The list of files where a style mistake has been added, and the mirror list of the \
             original files, and the dictionary of the sets of lines modified by file. \
             The number of lines that have been modified, must be equal to the number \
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


def files2vnodes(files: Iterable[str], feature_extractor: FeatureExtractor, client: str
                 ) -> Iterable[VirtualNode]:
    """
    Return the `VirtualNode`-s extracted from a list of files.

    :param files: List of files to get `Misprediction`-s and `VirtualNode`-s from.
    :param feature_extractor: FeatureExtractor to use.
    :param client: Babelfish client. Babelfish server should be started accordingly.
    :return: List of `VirtualNode`-s extracted from a given list of files.
    """
    files = prepare_files(files, client, feature_extractor.language)
    _, _, vnodes_y, _, _, _ = feature_extractor.extract_features(files)
    return vnodes_y


def files2mispreds(files: Iterable[str], feature_extractor: FeatureExtractor, rules: Rules,
                   client: BblfshClient) -> Iterable[Misprediction]:
    """
    Return the model's `Misprediction`-s on a list of files.

    :param files: List of files to get `Misprediction`-s from.
    :param feature_extractor: FeatureExtractor to use.
    :param rules: Rules to use for prediction.
    :param client: Babelfish client. Babelfish server should be started accordingly.
    :return: List of `Misprediction`-s extracted from a given list of files.
    """
    files = prepare_files(files, client, feature_extractor.language)
    X, y, vnodes_y, vnodes, vnodes_parents, parents = feature_extractor.extract_features(files)
    y_pred, winners, safe_preds = rules.predict(X=X, y=y, vnodes_y=vnodes_y, vnodes=vnodes,
                                                files={f.path: f for f in files},
                                                feature_extractor=feature_extractor,
                                                client=client, vnodes_parents=vnodes_parents,
                                                parents=parents)
    y = y[safe_preds]
    vnodes_y = [vn for i, vn in enumerate(list(vnodes_y)) if i in safe_preds]
    mispreds = get_mispreds(y, y_pred, vnodes_y, winners)
    return mispreds


def get_mispreds(y: numpy.ndarray, y_pred: numpy.ndarray, nodes: Iterable[VirtualNode],
                 winners: numpy.ndarray) -> Iterable[Misprediction]:
    """
    Return the list of `Misprediction`-s where the labels differ.

    :param y: Numpy 1-dimensional array of labels.
    :param y_pred: Numpy 1-dimensional array of predicted labels by the model.
    :param nodes: List of `VirtualNode`-s.
    :param winners: Numpy 1-dimensional array of the winning rule indices for each sample.
    :return: List of `Misprediction`-s where the labels `y` and `y_pred` differ.
    """
    mispreds = []
    for gt, pred, vn, rule in zip(y, y_pred, nodes, winners):
        if gt != pred:
            mispreds.append(Misprediction(gt, pred, vn, rule))
    return mispreds


def get_diff_mispreds(mispreds: Iterable[Misprediction], lines_changed: Mapping[str, Set[int]]
                      ) -> Mapping[str, Misprediction]:
    """
    Filter `Misprediction`-s to select those involving at least one line that has been modified.

    :param mispreds: List of `Misprediction`-s to filter.
    :param lines_changed: Dict of lines that have been changed when adding random noise.
    :return: Dictionary of the `Misprediction`-s involving at least one line \
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
                    true_files: Iterable[str], noisy_files: Iterable[str],
                    feature_extractor: FeatureExtractor) -> Iterable[Misprediction]:
    """
    Return `Misprediction`-s that fix the style mistakes added.

    Given a list of `Misprediction`-s potentially fixing a style mistake added since involving
    at least one line that has been modified, return the list of `Mispredicitons` really fixing
    that mistake because their prediction on the noisy files would match the ground truth
    labels of the original files.

    :param mispreds: Dictionary of `Misprediction`-s potentially fixing a style mistake.
    :param vnodes: List of `VirtualNode`-s extracted from the list of `true_files`.
    :param true_files: list of files of the original repos where a style mistake has been added.
    :param noisy_files: list of files from the noisy repos where a modification has been made
    :param feature_extractor: FeatureExtractor used to extract features.
    :return: List of `Misprediction`-s where the prediction on a noisy file matches the ground \
             truth label of the original file i.e. `Misprediction`-s actually fixing the mistakes \
             added.
    """
    style_fixes = []
    for true_file, noisy_file in zip(true_files, noisy_files):
        try:
            mispred = mispreds[noisy_file]
        except KeyError:
            continue
        for vn in vnodes:
            if vn.path == true_file and vn.start.offset == mispred.node.start.offset:
                print(feature_extractor.labels_to_class_sequences[mispred.pred], vn.y)
                if feature_extractor.labels_to_class_sequences[mispred.pred] == vn.y:
                    style_fixes.append(mispred)
                break
    return style_fixes


def compute_metrics(changes_count: int, predictions_count: int, true_positive: int,
                    ) -> Tuple[float, float, float]:
    """
    Compute precision, recall and F1-score metrics.

    :param changes_count: Overall number of cases.
    :param predictions_count: Total number of predictions made by the model.
    :param true_positive: Number of positive cases predicted as positive.
    :return: Precision, recall and F1-score metrics.
    """
    false_positive = predictions_count - true_positive
    false_negative = changes_count - predictions_count
    try:
        precision = true_positive / (true_positive + false_positive)
    except ZeroDivisionError:
        precision = 1.
    try:
        recall = true_positive / (true_positive + false_negative)
    except ZeroDivisionError:
        recall = 0.
    try:
        f1_score = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        f1_score = 0.
    return precision, recall, f1_score


def style_robustness_report(true_repo: str, noisy_repo: str, bblfsh: str, language: str,
                            model_path: str) -> None:
    """
    Print the quality report of a model tested on a given repository.

    The tests consists in adding random style mistakes in the given repo and looking how well
    the model is able to fix them according to the style of the original repository.

    :param true_repo: Path to the original repository we want to test the model on.
    :param noisy_repo: Path to the noisy version of the repository where 1 style mistake is \
           randomly added in every file.
    :param bblfsh: Babelfish client. Babelfish server should be started accordingly.
    :param language: Language to consider, others will be discarded.
    :param model_path: Path to the model to test. It should be previously trained on the original \
           repository located in ':param true_repo:'.
    """
    log = logging.getLogger("style_robustness_report")

    true_content = get_content_from_repo(true_repo)
    noisy_content = get_content_from_repo(noisy_repo)
    true_files, noisy_files, lines_changed = get_difflib_changes(true_content, noisy_content)
    log.info("Number of files modified by adding style noise: %d / %d", len(true_files),
             len(true_content))
    del true_content, noisy_content

    client = BblfshClient(bblfsh)
    analyzer = FormatModel().load(model_path)
    rules = analyzer[language]
    feature_extractor = FeatureExtractor(language=language,
                                         **rules.origin_config["feature_extractor"])
    vnodes_y_true = files2vnodes(true_files, feature_extractor, client)
    mispreds_noise = files2mispreds(noisy_files, feature_extractor, rules, client)
    diff_mispreds = get_diff_mispreds(mispreds_noise, lines_changed)
    changes_count = len(lines_changed)
    log.info("Number of artificial mistakes potentially fixed by the model "
             "(diff of mispredictions): %d / %d", len(diff_mispreds), changes_count)
    style_fixes = get_style_fixes(diff_mispreds, vnodes_y_true, true_files, noisy_files,
                                  feature_extractor)
    log.info("style-analyzer fixes in the noisy repos: %d / %d -> %.1f %%",
             len(style_fixes), changes_count, 100 * len(style_fixes) / changes_count)

    precision, recall, f1_score = compute_metrics(changes_count=changes_count,
                                                  predictions_count=len(diff_mispreds),
                                                  true_positive=len(style_fixes))
    print("precision:", round(precision, 3))
    print("recall:", round(recall, 3))
    print("F1 score:", round(f1_score, 3))

    print()
    print("list of files where the style-analyzer succeeds in fixing the random noise:")
    for mispred in style_fixes:
        print(mispred.node.path)


def filter_relevant_rules(rules: Iterable[Rules], support_threshold: int, log: logging.Logger
                          ) -> Iterable[Tuple[int, float]]:
    """
    Filter relevant rules that have a support higher than `support threshold`.

    :param rules: List of `Rules` from the model.
    :param support_threshold: Support threshold to filter relevant rules.
    :param log: Logger.
    :return: List of `Rules` index and confidence we filter according to `support_threshold`.
    """
    log.info("Filtering rules with support higher than %d", support_threshold)
    rules_id = [(i, r.stats.conf, r.stats.support) for i, r in enumerate(rules)
                if r.stats.support > support_threshold]
    rules_selection = sorted(rules_id, key=lambda k: k[1], reverse=True)
    log.info("Number of rules decreased from %d to %d", len(rules), len(rules_selection))
    return rules_selection


def plot_curve(x: numpy.ndarray, y: numpy.ndarray, output: str) -> None:
    """
    Plot y versus x as lines and markers using matplotlib.

    :param x: 1-D numpy array containing the x coordinates.
    :param y: 1-D numpy array containing the y coordinates.
    :param output: Path to the output figure, could be either a png or svg file.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit("Matplotlib is required to plot the Precision/Recall curve")
    plt.figure(figsize=(15, 10))
    ax = plt.subplot(111)
    ax.plot(x, y, marker="x", linestyle="--")
    ax.set_ylabel("Precision", fontsize=17, labelpad=15)
    ax.set_xlabel("Recall", fontsize=17, labelpad=15)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.savefig(output)


def plot_pr_curve(true_repo: str, noisy_repo: str, bblfsh: str, language: str,
                  model_path: str, support_threshold: int, output: str) -> None:
    """
    Plot a precision/recall curve with rules having higher support than `support_threshold`.

    :param true_repo: Path to the original repository we want to test the model on.
    :param noisy_repo: Path to the noisy version of the repository where 1 style mistake is \
           randomly added in every file.
    :param bblfsh: Babelfish client. Babelfish server should be started accordingly.
    :param language: Language to consider, others will be discarded.
    :param model_path: Path to the model to test. It should be previously trained on the original \
           repository located in ':param true_repo:'.
    :param support_threshold: Support threshold to filter relevant rules.
    :param output: Path to the output figure. Could yield to a png or svg file.
    """
    log = logging.getLogger("plot_pr_curve")

    true_content = get_content_from_repo(true_repo)
    noisy_content = get_content_from_repo(noisy_repo)
    true_files, noisy_files, lines_changed = get_difflib_changes(true_content, noisy_content)
    log.info("Number of files modified by adding style noise: %d / %d", len(true_files),
             len(true_content))
    del true_content, noisy_content

    client = BblfshClient(bblfsh)
    analyzer = FormatModel().load(model_path)
    rules = analyzer[language]
    feature_extractor = FeatureExtractor(language=language,
                                         **rules.origin_config["feature_extractor"])
    vnodes_y_true = files2vnodes(true_files, feature_extractor, client)
    mispreds_noise = files2mispreds(noisy_files, feature_extractor, rules, client)
    diff_mispreds = get_diff_mispreds(mispreds_noise, lines_changed)
    changes_count = len(lines_changed)

    precisions, recalls = [], []
    rules_selection = filter_relevant_rules(rules.rules, support_threshold, log)
    for i in range(len(rules_selection)):
        filtered_mispreds = {k: m for k, m in diff_mispreds.items()
                             if any(r[0] == m.rule for r in rules_selection[:i + 1])}
        style_fixes = get_style_fixes(filtered_mispreds, vnodes_y_true,
                                      true_files, noisy_files, feature_extractor)
        precision, recall, f1_score = compute_metrics(changes_count=changes_count,
                                                      predictions_count=len(filtered_mispreds),
                                                      true_positive=len(style_fixes))
        precisions.append(round(precision, 3))
        recalls.append(round(recall, 3))
        log.debug("precision: %.3f", precision)
        log.debug("recall: %.3f", recall)
        log.debug("F1 score: %.3f", f1_score)

    print("recall x:", recalls)
    print("precision y:", precisions)
    plot_curve(numpy.asarray(recalls), numpy.asarray(precisions), output)
