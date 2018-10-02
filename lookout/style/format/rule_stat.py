"""Facilities to report the quality and statistics of a given rules on a given dataset."""
from collections import defaultdict

from bblfsh import BblfshClient
from tqdm import tqdm

from lookout.style.format.feature_extractor import FeatureExtractor
from lookout.style.format.feature_utils import CLASSES
from lookout.style.format.model import FormatModel
from lookout.style.format.quality_report import prepare_files


class RuleStat:
    def __init__(self):
        self.gt_classes = [0 for _ in range(len(CLASSES))]
        self.pred_classes = [0 for _ in range(len(CLASSES))]


def print_rule_table(rule_stat):
    print("Legend: predictions/ground truth")
    report = [["#rule"] + list(CLASSES) + ["n_mistakes", "support"]]
    for rule in sorted(rule_stat):
        line = ["Rule number %s: " % rule]
        line.extend(
            ["%s/%s" % (pred, gt) for gt, pred in zip(rule_stat[rule].gt_classes,
                                                      rule_stat[rule].pred_classes)]
        )

        n_mistakes = int(sum(abs(pred - gt) for gt, pred in zip(rule_stat[rule].gt_classes,
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
        print(new_line)


def rules_report(input_pattern: str, bblfsh: str, language: str, model_path: str) -> None:
    """Print several different reports for a given model on a given dataset."""
    model = FormatModel().load(model_path)
    rules = model[language]
    print("Model parameters: %s" % rules.origin_config)
    print("Stats about rules: %s" % rules)

    for i, rule in enumerate(rules.rules):
        print("Rule %s: %s" % (i, rule))

    client = BblfshClient(bblfsh)
    files = prepare_files(input_pattern, client, language)
    print("Number of files: %s" % (len(files)))

    fe = FeatureExtractor(language=language, **rules.origin_config["feature_extractor"])
    res = fe.extract_features(files)

    if res is None:
        print("Failed to parse files, aborting report...")
        return

    X, y, _, _ = res
    X, _ = fe.select_features(X, y)

    y_pred, winners = rules.predict(X, True)

    rule_stat = defaultdict(RuleStat)

    for gt, pred, rule in tqdm(zip(y, y_pred, winners), total=y.shape[0]):
        rule_stat[rule].gt_classes[gt] += 1
        rule_stat[rule].pred_classes[pred] += 1

    print("Overall statistics:")
    print_rule_table(rule_stat)
