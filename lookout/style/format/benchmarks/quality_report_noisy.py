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
from typing import Iterable, List, Mapping, NamedTuple, Optional, Set, Tuple, Union

from bblfsh import BblfshClient
import jinja2
from lookout.core.analyzer import ReferencePointer
from lookout.core.bytes_to_unicode_converter import BytesToUnicodeConverter
from lookout.core.lib import filter_files_by_path, parse_files
import numpy
from yaml import safe_load

from lookout.style.format.analyzer import FormatAnalyzer
from lookout.style.format.benchmarks.general_report import FakeDataService
from lookout.style.format.feature_extractor import FeatureExtractor
from lookout.style.format.model import FormatModel
from lookout.style.format.rules import Rules
from lookout.style.format.uast_stability_checker import UASTStabilityChecker
from lookout.style.format.virtual_node import VirtualNode

# format: url,clean_commit,noisy_commit
REPOSITORIES = """
https://github.com/warenlg/axios,75c8b3f146aaa8a71f7dca0263686fb1799f8f31,b5d60bb7aaa1b3ba0f286a5dad3028968831fd1d
https://github.com/warenlg/jquery,dfa92ccead70d7dd5735a36c6d0dd1af680271cd,6f0f8a9bb739c5fe6d979736ef5d1e4a0be83446
""".strip()

Misprediction = NamedTuple("Misprediction", [("y", numpy.ndarray), ("pred", numpy.ndarray),
                                             ("node", List[VirtualNode]), ("rule", numpy.ndarray)])


def train(training_dir: str, ref: ReferencePointer, output_path: str, language: str, bblfsh: str,
          config: Optional[Union[str, dict]], log: Optional[logging.Logger] = None) -> FormatModel:
    """
    Train a FormatModel for debugging purposes.

    :param training_dir: Path to the directory containing the files to train from.
    :param ref: Reference pointer to repository for training
    :param output_path: Path to the model to write.
    :param language: Language to filter on.
    :param bblfsh: Address of the babelfish server.
    :param config: Path to a YAML config to use during the training or \
                   json-like object with a config.
    :param log: logger used to report during training.
    :return: Trained FormatNodel.
    """
    bblfsh_client = BblfshClient(bblfsh)
    if config is not None:
        if isinstance(config, str):
            with open(config) as fh:
                config = safe_load(fh)
    else:
        config = {}
    config = FormatAnalyzer._load_config(config)
    filepaths = glob.glob(os.path.join(training_dir, "**", "*.js"), recursive=True)
    model = FormatAnalyzer.train(ref, config, FakeDataService(
        bblfsh_client=bblfsh_client,
        files=parse_files(
            filepaths=filepaths,
            line_length_limit=config["train"][language]["line_length_limit"],
            overall_size_limit=config["train"][language]["overall_size_limit"],
            client=bblfsh_client,
            language=language,
            log=log),
        changes=None))
    model.save(output_path)
    return model


def get_content_from_repo(folder: str) -> Mapping[str, str]:
    """
    Extract the content of the files given their path.

    :param folder: Path to the files to get the content from.
    :return: Dictionary where the key is the path to a file and its value the content of the file.
    """
    content = {}
    filepaths = glob.glob(folder, recursive=True)
    for file in filter_files_by_path(filepaths):
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


def files2vnodes(filepaths: Iterable[str], feature_extractor: FeatureExtractor, rules: Rules,
                 client: BblfshClient) -> Iterable[VirtualNode]:
    """
    Return the `VirtualNode`-s extracted from a list of files.

    :param filepaths: List of files to get `Misprediction`-s and `VirtualNode`-s from.
    :param feature_extractor: FeatureExtractor to use.
    :param rules: Rules to use for prediction.
    :param client: Babelfish client. Babelfish server should be started accordingly.
    :return: List of `VirtualNode`-s extracted from a given list of files.
    """
    files = parse_files(filepaths=filepaths,
                        line_length_limit=rules.origin_config["line_length_limit"],
                        overall_size_limit=rules.origin_config["overall_size_limit"],
                        client=client, language=feature_extractor.language)
    _, _, (vnodes_y, _, _, _) = feature_extractor.extract_features(
        map(BytesToUnicodeConverter.convert_file, files))
    return vnodes_y


def files2mispreds(filepaths: Iterable[str], feature_extractor: FeatureExtractor, rules: Rules,
                   client: BblfshClient, log: logging.Logger) -> Iterable[Misprediction]:
    """
    Return the model's `Misprediction`-s on a list of files.

    :param filepaths: List of files to get `Misprediction`-s from.
    :param feature_extractor: FeatureExtractor to use.
    :param rules: Rules to use for prediction.
    :param client: Babelfish client. Babelfish server should be started accordingly.
    :param log: Logger.
    :return: List of `Misprediction`-s extracted from a given list of files.
    """
    files = parse_files(filepaths=filepaths,
                        line_length_limit=rules.origin_config["line_length_limit"],
                        overall_size_limit=rules.origin_config["overall_size_limit"],
                        client=client, language=feature_extractor.language)
    files = list(map(BytesToUnicodeConverter.convert_file, files))
    X, y, (vnodes_y, vnodes, vnode_parents, node_parents) = feature_extractor \
        .extract_features(files)
    y_pred, rule_winners, _, grouped_quote_predictions = rules.predict(
        X=X, vnodes_y=vnodes_y, vnodes=vnodes, feature_extractor=feature_extractor)
    y_pred = rules.fill_missing_predictions(y_pred, y)
    checker = UASTStabilityChecker(feature_extractor=feature_extractor)
    y, y_pred, vnodes_y, rule_winners, safe_preds = checker.check(
        y=y, y_pred=y_pred, vnodes_y=vnodes_y, vnodes=vnodes, files=list(files), stub=client._stub,
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
    prediction_rate = predictions_count / changes_count
    try:
        precision = true_positive / (true_positive + false_positive)
    except ZeroDivisionError:
        precision = 1.
    return prediction_rate, precision


def plot_curve(repositories: Iterable[str], prediction_rates: Mapping[str, numpy.ndarray],
               precisions: Mapping[str, numpy.ndarray], precision_threshold: float,
               limit_conf_id: Mapping[str, int], path_to_figure: str) -> None:
    """
    Plot y versus x as lines and markers using matplotlib.

    :param repositories: List of the repository names we plot the precision-recall curve.
    :param prediction_rates: Dict of 1-D numpy array containing the x coordinates.
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
    f, axes = plt.subplots(2, 1, figsize=(12, 12), sharex=True, tight_layout=True)
    f.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)
    plt.xlabel("Normalized number of rules", fontsize=17, labelpad=20)
    for repo in repositories:
        x0 = limit_conf_id[repo]
        prediction_rates_array = numpy.asarray(prediction_rates[repo])
        precisions_array = numpy.asarray(precisions[repo])
        rules = numpy.asarray([i / prediction_rates_array.shape[0]
                               for i in range(prediction_rates_array.shape[0])])
        for ax, metric, ylabel in zip(axes, (prediction_rates_array, precisions_array),
                                      ("prediction rate", "precision")):
            ax.plot(rules[x0:], metric[x0:], color="lightgrey")
            ax.plot(rules[:x0 + 1], metric[:x0 + 1], label=repo)
            ax.set_ylabel(ylabel, fontsize=17, labelpad=20)
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.tick_params(labelsize=17, color="gray")
    f.add_subplot(212, frameon=False)
    plt.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles, labels, loc="upper right", fontsize=17)
    plt.savefig(path_to_figure, pad_inches=0, bbox_inches="tight")


def quality_report_noisy(bblfsh: str, language: str, confidence_threshold: float,
                         support_threshold: int, precision_threshold: float, dir_output: str,
                         config: Optional[dict] = None, repos: Optional[str] = None) -> None:
    """
    Generate a quality report on the artificial noisy dataset including evaluation curves.

    :param bblfsh: Babelfish client. Babelfish server should be started accordingly.
    :param language: Language to consider, others will be discarded.
    :param confidence_threshold: Confidence threshold to filter relevant rules.
    :param support_threshold: Support threshold to filter relevant rules.
    :param precision_threshold: Precision threshold tolerated by the model. \
           Limit drawn as a red horizontal line on the figure.
    :param dir_output: Path to the output directory where to store the quality report in Markdown \
           and the precision-recall curve in png format.
    :param config: FormatAnalyzer config to use. Default one is used if not set.
    :param repos: Input list of urls to the repositories to analyze. \
           Should be strings separated by newlines. If it is None, \
           we use the string defined at the beginning of the file.
    """
    log = logging.getLogger("quality_report_noisy")

    # initialization
    repo_names = []
    last_accepted_rule = {}
    prediction_rates, precisions, accepted_rules = (defaultdict(list) for _ in range(3))
    n_mistakes, prec_max_prediction_rate, confidence_threshold_exp, max_prediction_rate, \
        n_rules, n_rules_filtered = ({} for _ in range(6))
    if repos is None:
        repos = REPOSITORIES
    try:
        # fetch the the original and noisy repositories
        client = BblfshClient(bblfsh)
        log.info("Repositories: %s", repos)
        with tempfile.TemporaryDirectory() as tmpdirname:
            for raw in repos.splitlines():
                repo_path, clean_commit, noisy_commit = raw.split(",")
                repo = repo_path.split("/")[-1]
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

                # train the model on the original repository
                ref = ReferencePointer(repo_path, "HEAD", clean_commit)
                model_path = os.path.join(git_dir, "model.asdf")
                format_model = train(training_dir=git_dir, ref=ref, output_path=model_path,
                                     language=language, bblfsh=bblfsh, config=config, log=log)
                rules = format_model[language]

                # extract the raw data and the diff from the repositories
                input_pattern = os.path.join(git_dir, "**", "*.js")
                input_pattern_noisy = os.path.join(git_dir_noisy, "**", "*.js")
                true_content = get_content_from_repo(input_pattern)
                noisy_content = get_content_from_repo(input_pattern_noisy)
                true_files, noisy_files, start_changes = get_difflib_changes(true_content,
                                                                             noisy_content)
                if not true_files:
                    raise ValueError("Noisy repo should count at least one artificial mistake")
                log.info("Number of files modified by adding style noise: %d / %d",
                         len(true_files), len(true_content))
                del true_content, noisy_content

                # extract the features
                feature_extractor = FeatureExtractor(language=language,
                                                     **rules.origin_config["feature_extractor"])
                vnodes_y_true = files2vnodes(true_files, feature_extractor, rules, client)
                mispreds_noise = files2mispreds(noisy_files, feature_extractor, rules, client, log)

                # compute the prediction rate and precision score on the artificial noisy dataset
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
                    prediction_rate, precision = compute_metrics(
                        changes_count=changes_count,
                        predictions_count=len(filtered_mispreds),
                        true_positive=len(style_fixes))
                    prediction_rates[repo].append(round(prediction_rate, 3))
                    precisions[repo].append(round(precision, 3))
                print("prediction rate x:", prediction_rates[repo])
                print("precision y:", precisions[repo])

                # compute other statistics and quality metrics for the model's evaluation
                repo_names.append(repo)
                n_mistakes[repo] = len(true_files)
                prec_max_prediction_rate[repo] = precisions[repo][-1]
                max_prediction_rate[repo] = max(prediction_rates[repo])
                n_rules_filtered[repo] = len(rules_id)

                # compute the confidence and prediction rate limit for a given precision threshold
                for i, (prediction_rate, prec) in enumerate(zip(prediction_rates[repo],
                                                                precisions[repo])):
                    if prec >= precision_threshold:
                        accepted_rules[repo].append((i, rules_id[i][1], prediction_rate))
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

    # compile the curves showing the evolutions of the prediction rate and precision score
    path_to_figure = os.path.join(dir_output, "pr_curves.png")
    plot_curve(repo_names, prediction_rates, precisions, precision_threshold,
               limit_conf_id, path_to_figure)

    # compile the markdown template for the report through jinja2
    loader = jinja2.FileSystemLoader((os.path.join(os.path.dirname(__file__), "..", "templates"),),
                                     followlinks=True)
    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=True)
    env.globals.update(range=range)
    template = loader.load(env, "noisy_quality_report.md.jinja2")
    report = template.render(repos=repo_names, n_mistakes=n_mistakes,
                             prec_max_prediction_rate=prec_max_prediction_rate,
                             confidence_threshold_exp=round(max_confidence_threshold_exp[1], 2),
                             max_prediction_rate=max_prediction_rate,
                             confidence_threshold=confidence_threshold,
                             support_threshold=support_threshold,
                             n_rules=n_rules, n_rules_filtered=n_rules_filtered,
                             path_to_figure=path_to_figure)

    # write the quality report
    repo_pathrt = os.path.join(dir_output, "report_noise.md")
    with open(repo_pathrt, "w", encoding="utf-8") as f:
        f.write(report)
