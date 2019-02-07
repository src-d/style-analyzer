"""Evaluate how well a given model is able to fix style mistakes randomly added in a repository."""
from collections import defaultdict
from difflib import SequenceMatcher
import glob
import logging
from operator import itemgetter
import os
import subprocess
import sys
import tempfile
from typing import Iterable, List, Mapping, NamedTuple, Optional, Set, Tuple

from bblfsh import BblfshClient
import jinja2
from lookout.core.analyzer import ReferencePointer
import numpy
from yaml import safe_load

from lookout.style.format.analyzer import FormatAnalyzer
from lookout.style.format.benchmarks.general_report import FakeDataService
from lookout.style.format.feature_extractor import FeatureExtractor
from lookout.style.format.files_filtering import filter_filepaths
from lookout.style.format.model import FormatModel
from lookout.style.format.postprocess import filter_uast_breaking_preds
from lookout.style.format.rules import Rules
from lookout.style.format.utils import prepare_files
from lookout.style.format.virtual_node import VirtualNode

# format: url,clean_commit,noisy_commit
REPOSITORIES = """
https://github.com/warenlg/axios,75c8b3f146aaa8a71f7dca0263686fb1799f8f31,b5d60bb7aaa1b3ba0f286a5dad3028968831fd1d
https://github.com/warenlg/jquery,dfa92ccead70d7dd5735a36c6d0dd1af680271cd,6f0f8a9bb739c5fe6d979736ef5d1e4a0be83446
""".strip()

Misprediction = NamedTuple("Misprediction", [("y", numpy.ndarray), ("pred", numpy.ndarray),
                                             ("node", List[VirtualNode]), ("rule", numpy.ndarray)])


def train(training_dir: str, ref: ReferencePointer, output_path: str, language: str, bblfsh: str,
          config: Optional[str]) -> FormatModel:
    """
    Train a FormatModel for debugging purposes.

    :param training_dir: Path to the directory containing the files to train from.
    :param ref: Reference pointer to repository for training
    :param output_path: Path to the model to write.
    :param language: Language to filter on.
    :param bblfsh: Address of the babelfish server.
    :param config: Path to a YAML config to use during the training.

    :return: Trained FormatNodel.
    """
    bblfsh_client = BblfshClient(bblfsh)
    if config is not None:
        with open(config) as fh:
            config = safe_load(fh)
    else:
        config = {}
    filenames = glob.glob(os.path.join(training_dir, "**", "*"), recursive=True)
    model = FormatAnalyzer.train(
        ref,
        config,
        FakeDataService(bblfsh_client, prepare_files(filenames, bblfsh_client, language), None))
    model.save(output_path)
    return model


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


def get_difflib_changes(true_content: Mapping[str, str], noisy_content: Mapping[str, str],
                        ) -> Tuple[Iterable[str], Iterable[str], Mapping[str, Set[int]], int]:
    """
    Return the files and the first offsets that have been changed when adding random noise.

    Given 2 contents of one repository (the original and its noisy version), returns the list of \
    files that have been modified, the first offsets that have been changed.

    :param true_content: Dictionary containing the content of the original repository.
    :param noisy_content: Dictionary containing the content of the noisy version of the repository.
    :return: The list of files where a style mistake has been added, and the mirror list of the \
             original files, and the dictionary of firsts offsets that have been changed when \
             adding random noise.
    """
    true_files, noisy_files = set(), set()
    start_changes = {}
    for (tf, tc), (nf, nc) in zip(true_content.items(), noisy_content.items()):
        if tc == nc:
            continue
        matcher = SequenceMatcher(a=tc, b=nc)
        first_offset_changed = matcher.get_matching_blocks()[0].size - 1
        if first_offset_changed < len(tc) and first_offset_changed < len(nc):
            start_changes[nf] = first_offset_changed
            true_files.add(tf)
            noisy_files.add(nf)
    return sorted(true_files), sorted(noisy_files), start_changes


def files2vnodes(files: Iterable[str], feature_extractor: FeatureExtractor, client: str,
                 ) -> Iterable[VirtualNode]:
    """
    Return the `VirtualNode`-s extracted from a list of files.

    :param files: List of files to get `Misprediction`-s and `VirtualNode`-s from.
    :param feature_extractor: FeatureExtractor to use.
    :param client: Babelfish client. Babelfish server should be started accordingly.
    :return: List of `VirtualNode`-s extracted from a given list of files.
    """
    files = prepare_files(files, client, feature_extractor.language)
    _, _, (vnodes_y, _, _, _) = feature_extractor.extract_features(files)
    return vnodes_y


def files2mispreds(files: Iterable[str], feature_extractor: FeatureExtractor, rules: Rules,
                   client: BblfshClient, log: logging.Logger) -> Iterable[Misprediction]:
    """
    Return the model's `Misprediction`-s on a list of files.

    :param files: List of files to get `Misprediction`-s from.
    :param feature_extractor: FeatureExtractor to use.
    :param rules: Rules to use for prediction.
    :param client: Babelfish client. Babelfish server should be started accordingly.
    :param log: Logger.
    :return: List of `Misprediction`-s extracted from a given list of files.
    """
    files = prepare_files(files, client, feature_extractor.language)
    X, y, (vnodes_y, vnodes, vnode_parents, node_parents) = feature_extractor \
        .extract_features(files)
    y_pred, rule_winners, _, grouped_quote_predictions = rules.predict(
        X=X, vnodes_y=vnodes_y, vnodes=vnodes, feature_extractor=feature_extractor)
    y_pred = rules.fill_missing_predictions(y_pred, y)
    y, y_pred, vnodes_y, rule_winners, safe_preds = filter_uast_breaking_preds(
        y=y, y_pred=y_pred, vnodes_y=vnodes_y, vnodes=vnodes,
        files={f.path: f for f in files}, feature_extractor=feature_extractor, stub=client._stub,
        vnode_parents=vnode_parents, node_parents=node_parents, rule_winners=rule_winners,
        grouped_quote_predictions=grouped_quote_predictions)
    mispreds = get_mispreds(y, y_pred, vnodes_y, rule_winners)
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


def get_diff_mispreds(mispreds: Iterable[Misprediction], start_changes: Mapping[str, int],
                      ) -> Mapping[str, Misprediction]:
    """
    Filter `Misprediction`-s to select those involving at least one line that has been modified.

    :param mispreds: List of `Misprediction`-s to filter.
    :param start_changes: Dict of first offsets that have been changed when adding random noise.
    :return: Dictionary of the `Misprediction`-s located at the offset where a random mistake \
             has been previously added.
    """
    diff_mispreds = {}
    for m in mispreds:
        if m.node.start.offset >= start_changes[m.node.path] and m.node.path not in diff_mispreds:
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
            if vn.path == true_file and vn.start.offset >= mispred.node.start.offset:
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
    :return: Prediction rate and precision metrics.
    """
    false_positive = predictions_count - true_positive
    predr = predictions_count / changes_count
    try:
        precision = true_positive / (true_positive + false_positive)
    except ZeroDivisionError:
        precision = 1.
    return predr, precision


def plot_curve(repositories: Iterable[str], predrs: Mapping[str, numpy.ndarray],
               precisions: Mapping[str, numpy.ndarray], precision_threshold: float,
               limit_conf_id: Mapping[str, int], path_to_figure: str) -> None:
    """
    Plot y versus x as lines and markers using matplotlib.

    :param repositories: List of the repository names we plot the precision-recall curve.
    :param predrs: Dict of 1-D numpy array containing the x coordinates.
    :param precisions: Dict of 1-D numpy array containing the y coordinates.
    :param precision_threshold: Precision threshold tolerated by the model. \
           Limit drawn as a red horizontal line on the figure.
    :param limit_conf_id: Dict of last accepted rule indices according to the maximum \
           confidence threshold observed.
    :param path_to_figure: Path to the output figure, in png format.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit("Matplotlib is required to plot the Precision/Recall curve")
    f, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True, tight_layout=True)
    f.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.xlabel("Normalized number of rules")
    plt.set_title("Prediction rate and precision's evolutions on the artificial noisy dataset",
                  fontsize=17)
    for repo in repositories:
        x0 = limit_conf_id[repo]
        ax1.plot(numpy.asarray(rules[repo]), numpy.asarray(predrs[repo]))
        ax1.set_ylabel("PredR")
        ax1.spines["right"].set_visible(False)
        ax1.spines["top"].set_visible(False)
        ax2.plot(numpy.asarray(rules[repo]), numpy.asarray(predictions[repo]))
        ax2.set_ylabel("precision")
        ax2.spines["right"].set_visible(False)
        ax2.spines["top"].set_visible(False)
    f.add_subplot(212, frameon=False)
    plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    handles, labels = plt.get_legend_handles_labels()
    plt.legend(handles, labels, loc="best", fontsize=17)
    plt.savefig(path_to_figure, pad_inches=0, bbox_inches="tight")


def quality_report_noisy(bblfsh: str, language: str, confidence_threshold: float,
                         support_threshold: int, precision_threshold: float,
                         dir_output: str, retrain: bool, repos: Optional[str] = None,
                         ) -> None:
    """
    Generate a quality report on the artificial noisy dataset including a precision-recall curve.

    :param bblfsh: Babelfish client. Babelfish server should be started accordingly.
    :param language: Language to consider, others will be discarded.
    :param confidence_threshold: Confidence threshold to filter relevant rules.
    :param support_threshold: Support threshold to filter relevant rules.
    :param precision_threshold: Precision threshold tolerated by the model. \
           Limit drawn as a red horizontal line on the figure.
    :param dir_output: Path to the output directory where to store the quality report in Markdown \
           and the precision-recall curve in png format.
    :param repos: Input list of urls or paths to the repositories to analyze. \
           Should be strings separated by newlines.
    :param retrain: Set True if you want to force model retraining.
    """
    log = logging.getLogger("quality_report_noisy")
    repo_names = []
    last_accepted_rule = {}
    predrs, precisions, accepted_rules = (defaultdict(list) for _ in range(3))
    n_mistakes, prec_max_predr, confidence_threshold_exp, max_predr, \
        n_rules, n_rules_filtered = ({} for _ in range(6))
    if repos is None:
        repos = REPOSITORIES
    try:
        client = BblfshClient(bblfsh)
        log.info("Repositories: %s", repos)
        with tempfile.TemporaryDirectory() as tmpdirname:
            for raw in repos.splitlines():
                repo_path, clean_commit, noisy_commit = raw.split(",")
                repo = repo_path.split("/")[-1]
                if repo_path.startswith("https://github.com"):
                    log.info("Fetching %s", repo_path)
                    git_dir = os.path.join(tmpdirname, repo)
                    git_dir_noisy = os.path.join(tmpdirname, repo + "_noisy")
                    cmd1 = "git clone --single-branch --branch master %s %s" % (repo_path, git_dir)
                    cmd2 = "git clone --single-branch --branch style-noise-1-per-file %s %s" \
                        % (repo_path, git_dir_noisy)
                    try:
                        for cmd in (cmd1, cmd2):
                            log.debug("Running: %s", cmd)
                            subprocess.check_call(cmd.split())
                    except subprocess.CalledProcessError as e:
                        raise ConnectionError("Unable to fetch repository %s" % repo_path) from e
                    input_pattern = os.path.join(git_dir, "**", "*.js")
                    input_pattern_noisy = os.path.join(git_dir_noisy, "**", "*.js")
                    model_path = os.path.join(git_dir_noisy, "style-analyzer-model", "model.asdf")
                else:
                    input_pattern = os.path.join(repo_path, "**", "*.js")
                    input_pattern_noisy = os.path.join(repo_path + "_noisy", "**", "*.js")
                    model_path = os.path.join(repo_path, "model.asdf")
                true_content = get_content_from_repo(input_pattern)
                noisy_content = get_content_from_repo(input_pattern_noisy)
                true_files, noisy_files, start_changes = get_difflib_changes(true_content,
                                                                             noisy_content)
                if not true_files:
                    raise ValueError("Noisy repo should count at least one artificial mistake")
                log.info("Number of files modified by adding style noise: %d / %d",
                         len(true_files), len(true_content))
                del true_content, noisy_content
                if retrain:
                    ref = ReferencePointer(repo_path, "HEAD", clean_commit)
                    format_model = train(git_dir, ref, model_path, language, bblfsh, None)
                else:
                    format_model = FormatModel().load(model_path)
                rules = format_model[language]
                feature_extractor = FeatureExtractor(language=language,
                                                     **rules.origin_config["feature_extractor"])
                vnodes_y_true = files2vnodes(true_files, feature_extractor, client)
                mispreds_noise = files2mispreds(noisy_files, feature_extractor, rules, client, log)
                diff_mispreds = get_diff_mispreds(mispreds_noise, start_changes)
                changes_count = len(start_changes)
                n_rules[repo] = len(rules.rules)
                rules_id = [(i, r.stats.conf) for i, r in enumerate(rules.rules)
                            if r.stats.conf > confidence_threshold
                            and r.stats.support > support_threshold]
                rules_id = sorted(rules_id, key=lambda k: k[1], reverse=True)
                for i in range(len(rules_id)):
                    filtered_mispreds = {k: m for k, m in diff_mispreds.items()
                                         if any(r[0] == m.rule for r in rules_id[:i + 1])}
                    style_fixes = get_style_fixes(filtered_mispreds, vnodes_y_true,
                                                  true_files, noisy_files, feature_extractor)
                    predr, precision = compute_metrics(changes_count=changes_count,
                                                       predictions_count=len(filtered_mispreds),
                                                       true_positive=len(style_fixes))
                    predrs[repo].append(round(predr, 3))
                    precisions[repo].append(round(precision, 3))
                log.info([m.node.path for m in style_fixes])
                print("prediction rate x:", predrs[repo])
                print("precision y:", precisions[repo])

                # compute some stats and quality metrics for the model's evaluation
                repo_names.append(repo)
                n_mistakes[repo] = len(true_files)
                prec_max_predr[repo] = precisions[repo][-1]
                max_predr[repo] = max(predrs[repo])
                n_rules_filtered[repo] = len(rules_id)
                # compute the confidence and prediction rate limit for a given precision threshold
                for i, (predr, prec) in enumerate(zip(predrs[repo], precisions[repo])):
                    if prec >= precision_threshold:
                        accepted_rules[repo].append((i, rules_id[i][1], predr))
                last_accepted_rule[repo] = min(accepted_rules[repo], key=itemgetter(1))
                confidence_threshold_exp[repo] = (last_accepted_rule[repo][0],
                                                  last_accepted_rule[repo][1])
    finally:
        client._channel.close()

    # compute the index of the last accepted rule according to the maximum confidence threshold
    limit_conf_id = {}
    max_confidence_threshold_exp = max(confidence_threshold_exp.values(), key=itemgetter(1))
    for repo, rules in accepted_rules.items():
        for rule in rules:
            if rule[1] < max_confidence_threshold_exp[1]:
                break
            limit_conf_id[repo] = rule[0]

    # compile the curves showing the evolutions of the prediction rate and precision
    path_to_figure = os.path.join(dir_output, "pr_curves.png")
    plot_curve(repo_names, predrs, precisions, precision_threshold, limit_conf_id, path_to_figure)

    # compile the markdown template for the report through jinja2
    loader = jinja2.FileSystemLoader((os.path.join(os.path.dirname(__file__), "..", "templates"),),
                                     followlinks=True)
    env = jinja2.Environment(
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )
    env.globals.update(range=range)
    template = loader.load(env, "noisy_quality_report.md.jinja2")
    report = template.render(repos=repo_names, n_mistakes=n_mistakes,
                             prec_max_predr=prec_max_predr,
                             confidence_threshold_exp=round(max_confidence_threshold_exp[1], 2),
                             max_predr=max_predr, confidence_threshold=confidence_threshold,
                             support_threshold=support_threshold,
                             n_rules=n_rules, n_rules_filtered=n_rules_filtered,
                             path_to_figure=path_to_figure)

    # write the quality report
    repo_pathrt = os.path.join(dir_output, "report_noise.md")
    with open(repo_pathrt, "w", encoding="utf-8") as f:
        f.write(report)
