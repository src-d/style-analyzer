"""Facilities to report the quality and statistics of a given rules on a given dataset."""
from collections import defaultdict
import glob
import io
import logging
from typing import Any, Iterable, Mapping, MutableMapping, Union

from bblfsh import BblfshClient
import numpy
from sklearn.exceptions import NotFittedError
from tqdm import tqdm

from lookout.style.format.benchmarks.general_report import FormatModel, generate_model_report, \
    ReportAnalyzer
from lookout.style.format.benchmarks.time_profile import profile
from lookout.style.format.feature_extractor import FeatureExtractor
from lookout.style.format.postprocess import filter_uast_breaking_preds
from lookout.style.format.utils import prepare_files


class RuleStat:
    """Stats about ground truth and predicted classes for a given rule."""

    def __init__(self, feature_extractor: FeatureExtractor) -> None:
        """Construct a RuleStat."""
        self.gt_classes = [0 for _ in feature_extractor.labels_to_class_sequences]
        self.pred_classes = [0 for _ in feature_extractor.labels_to_class_sequences]


def generate_rule_table(rule_stat: Mapping[Any, RuleStat], feature_extractor: FeatureExtractor) \
        -> str:
    """
    Generate table from statistics about rules.

    :param rule_stat: mapping {rule: RuleStat).
    :param feature_extractor: FeatureExtractor used to extract features.
    :return: table in str format.
    """
    class_names = feature_extractor.composite_class_representations
    with io.StringIO() as output:
        print("Legend: predictions/ground truth", file=output)
        report = [["#rule"] + class_names + ["n_mistakes", "support"]]
        for rule in sorted(rule_stat):
            line = ["Rule number %s: " % rule]
            line.extend(["%s/%s" % (pred, gt) for gt, pred in zip(rule_stat[rule].gt_classes,
                                                                  rule_stat[rule].pred_classes)])

            n_mistakes = int(sum(abs(pred - gt)
                                 for gt, pred in zip(rule_stat[rule].gt_classes,
                                                     rule_stat[rule].pred_classes)) / 2)
            line.append(str(n_mistakes))
            support = str(sum(rule_stat[rule].pred_classes))
            line.append(support)
            report.append(line)
        max_cols = [0] * len(report[0])
        for line in report:
            for i, col in enumerate(line):
                max_cols[i] = max(max_cols[i], len(col))
        for line in report:
            new_line = ""
            for i, col in enumerate(line):
                new_line += col.ljust(max_cols[i])
                if i != len(line) - 1:
                    new_line += "|"
            print(new_line, file=output)
        return output.getvalue()


def print_rule_table(rule_stat: Mapping[Any, RuleStat], feature_extractor: FeatureExtractor) \
        -> None:
    """
    Print table from statistics about rules.

    :param rule_stat: mapping {rule: RuleStat).
    :param feature_extractor: FeatureExtractor used to extract features.
    """
    print(generate_rule_table(rule_stat=rule_stat, feature_extractor=feature_extractor))


def generate_report(y: Union[numpy.ndarray, Iterable[Union[int, float]]],
                    y_pred: Union[numpy.ndarray, Iterable[Union[int, float]]],
                    winners: Union[numpy.ndarray, Iterable[Union[int, float]]],
                    feature_extractor: FeatureExtractor) -> str:
    """
    Generate report about predictions and triggered rules.

    Description: rule, predicted labels, true labels, support, number of errors.

    :param y: labels from feature extractor.
    :param y_pred: predicted labels.
    :param winners: winner rules
    :param feature_extractor: FeatureExtractor used to extract features.
    :return: report in str format.
    """
    rule_stat = defaultdict(lambda: RuleStat(feature_extractor))

    for gt, pred, rule in tqdm(zip(y, y_pred, winners), total=y.shape[0]):
        rule_stat[rule].gt_classes[gt] += 1
        rule_stat[rule].pred_classes[pred] += 1

    return "Overall statistics:\n" + generate_rule_table(rule_stat=rule_stat,
                                                         feature_extractor=feature_extractor)


@profile
def print_rules_report(input_pattern: str, bblfsh: str, language: str, model_path: str) -> None:
    """Print several different reports for a given model on a given dataset."""
    model = FormatModel().load(model_path)
    if language not in model:
        raise NotFittedError()
    rules = model[language]
    print("Model parameters: %s" % rules.origin_config)
    print("Stats about rules: %s" % rules)

    fe = FeatureExtractor(language=language, **rules.origin_config["feature_extractor"])
    print(generate_model_report(model, language, fe))
    client = BblfshClient(bblfsh)
    filenames = glob.glob(input_pattern, recursive=True)
    files = prepare_files(filenames, client, language)
    print("Number of files: %s" % (len(files)))

    res = fe.extract_features(files)

    if res is None:
        print("Failed to parse files, aborting report...")
        return

    X, y, (vnodes_y, vnodes, vnode_parents, node_parents) = res
    y_pred, rule_winners, _, grouped_quote_predictions = rules.predict(
        X=X, vnodes_y=vnodes_y, vnodes=vnodes, feature_extractor=fe)
    y, y_pred, vnodes_y, rule_winners, safe_preds = filter_uast_breaking_preds(
        y=y, y_pred=y_pred, vnodes_y=vnodes_y, vnodes=vnodes, files={f.path: f for f in files},
        feature_extractor=fe, stub=client._stub, vnode_parents=vnode_parents,
        node_parents=node_parents, rule_winners=rule_winners,
        grouped_quote_predictions=grouped_quote_predictions)

    print(generate_report(y=y, y_pred=y_pred, winners=rule_winners, feature_extractor=fe))


class RuleStatAnalyzer(ReportAnalyzer):
    """
    Rules report analyzer.

    * analyze - generate report per changed files or aggregated changes.
    * train - generate report for all files.

    How-to:

    1) Launch analyzer: `analyzer run lookout.style.format.rule_stat -c config.yml`

    2) Query analyzer

    2.1) Rules report query for all files:
    ```
    lookout-sdk push ipv4://analyzer:port --git-dir /git/dir/ --from REV1 --to  REV2  \
    --config-json='{"style.format.analyzer.RuleStatAnalyzer":  \
    {"model_path": "/saved/model.asdf"}}'
    ```

    2.2) Rules report for changes per file:
    ```
    lookout-sdk review ipv4://analyzer:port --git-dir /git/dir/ --from REV1 --to  REV2  \
    --config-json='{"style.format.analyzer.RuleStatAnalyzer":  \
    {"model_path": "/saved/model.asdf"}}'
    ```

    2.3) Rules report for aggregated changes:
    ```
    lookout-sdk review ipv4://analyzer:port --git-dir /git/dir/ --from REV1 --to  REV2  \
    --config-json='{"style.format.analyzer.RuleStatAnalyzer":  \
    {"model_path": "/saved/model.asdf", "aggregate": true}}'
    ```
    """

    log = logging.getLogger("RuleStatAnalyzer")
    name = "style.format.analyzer.RuleStatAnalyzer"
    version = "1"
    description = "Source code formatting rule stats analysis: " \
                  "support, triggered rules, errors, etc."

    def __init__(self, model: FormatModel, url: str, config: Mapping[str, Any]) -> None:
        """Initialize analyzer with pretrained model and configuration."""
        super().__init__(model, url, config)
        self.config = self._load_analysis_config(self.config)

    @classmethod
    def generate_model_report(cls, model: FormatModel, config: MutableMapping[str, Any],
                              feature_extractor: FeatureExtractor):
        """
        Generate report about model. Has to be implemented in child class.

        :param model: model that should be described in report.
        :param config: configuration.
        :param feature_extractor: feature extractor.
        :return: report.
        """
        res = []
        for lang in model:
            res.append(generate_model_report(model=model, language=lang,
                                             feature_extractor=feature_extractor))
        return "\n".join(res)

    @classmethod
    def generate_quality_report(cls, y: Union[numpy.ndarray, Iterable[Union[int, float]]],
                                y_pred: Union[numpy.ndarray, Iterable[Union[int, float]]],
                                rule_winners: Iterable[int], feature_extractor: FeatureExtractor,
                                **kwargs) -> str:
        """
        Generate report about predictions and triggered rules.

        Description: rule, predicted labels, true labels, number of errors, support.

        :param y: labels.
        :param y_pred: predicted labels.
        :param rule_winners: rule winner.
        :param feature_extractor: FeatureExtractor used to extract features.
        :param kwargs: helper to handel additional parameters.
        :return: report.
        """
        return generate_report(y=y, y_pred=y_pred, winners=rule_winners,
                               feature_extractor=feature_extractor)


analyzer_class = RuleStatAnalyzer
