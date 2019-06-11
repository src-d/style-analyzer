"""Modelforge model for the format analyzer."""
from copy import deepcopy
import io
from itertools import islice
from pprint import pprint
from typing import Dict, Iterable, List, Mapping, Tuple  # noqa: F401

from lookout.core.analyzer import AnalyzerModel
import numpy
from sourced.ml.core.models.license import DEFAULT_LICENSE

from lookout.style.format.rules import Rule, RuleAttribute, Rules, RuleStats


class FormatModel(AnalyzerModel):
    """
    A modelforge model to store Rules instances.

    It is required to store all the Rules for different programming languages in a single model,
    named after each language.
    Note that Rules must be fitted and Rules.base_model is not saved.

    Developer note:
    Each Rules must provide enough information to reproduce it bit-to-bit in form of a
    configuration dictionary. Model is simple and must remain simple.
    """

    LICENSE = DEFAULT_LICENSE

    def __init__(self, **kwargs):
        """Construct a FormatModel."""
        super().__init__(**kwargs)
        self._rules_by_lang = {}  # type: Dict[str, Rules]

    @property
    def languages(self) -> List[str]:
        """Return the languages for which this model has trained rules available."""
        return sorted(self._rules_by_lang)

    def dump(self) -> str:
        """Serialize this model and return the result as a string."""
        result = io.StringIO()
        result.write(super().dump())
        for lang, rules in sorted(self._rules_by_lang.items()):
            result.write("\n\n# %s\n%s\n" % (lang, rules))
            try:
                for ds in ("train", "test"):
                    print("## %s" % ds, file=result)
                    print("PPCR: %f" % self[lang].classification_report[ds]["ppcr"], file=result)
                    for r in ("report", "report_full"):
                        print("### %s" % r, file=result)
                        for avg in ("macro", "micro", "weighted"):
                            print(avg, file=result)
                            pprint(self[lang].classification_report[ds][r]["%s avg" % avg],
                                   stream=result)
            except KeyError:
                print("<classification report was not computed>", file=result)
        return result.getvalue().strip()

    def _generate_tree(self) -> dict:
        tree = super()._generate_tree()
        languages = self.languages
        tree.update(
            languages=languages,
            origin_configs=[self[lang].origin_config for lang in languages],
            ruless=[self._disassemble_rules(self[lang].rules) for lang in languages],
            classification_reports=[self._disassemble_classification_report(
                self[lang].classification_report) for lang in languages],
        )
        return tree

    def _load_tree(self, tree: dict) -> None:
        super()._load_tree(tree)
        for lang, origin_config, rules, report in zip(
                tree["languages"], tree["origin_configs"], tree["ruless"],
                tree["classification_reports"]):
            self[lang] = Rules(self._assemble_rules(rules), deepcopy(origin_config))
            self[lang]._classification_report = self._assemble_classification_report(report)

    def __len__(self) -> int:
        return len(self._rules_by_lang)

    def __getitem__(self, lang: str) -> Rules:
        """
        Get the Rules estimator by its language.

        :param lang: Estimator language.
        :return: Rules estimator instance.
        """
        return self._rules_by_lang[lang]

    def __setitem__(self, lang: str, rules: Rules):
        """
        Set a new Rules estimator to the model by its language.
        """
        self._rules_by_lang[lang] = rules

    def __iter__(self):
        yield from self._rules_by_lang.__iter__()

    def __contains__(self, lang: str) -> bool:
        return lang in self._rules_by_lang

    @staticmethod
    def _assemble_classification_report(report: dict) -> dict:
        for key in report:
            if report[key]:
                report[key]["confusion_matrix"] = numpy.array(report[key]["confusion_matrix"])
        return report

    @staticmethod
    def _disassemble_classification_report(report: dict) -> dict:
        report = deepcopy(report)
        for key in report:
            if report[key]:
                report[key]["confusion_matrix"] = report[key]["confusion_matrix"].tolist()
        return report

    @staticmethod
    def _assemble_rules(rules_tree: dict) -> List[Rule]:
        rules = []
        rule_attrs = (RuleAttribute(*params) for params in
                      zip(rules_tree["features"],  rules_tree["cmps"], rules_tree["thresholds"]))
        for cls, conf, support, artificial, length in zip(
                rules_tree["cls"], rules_tree["conf"], rules_tree["support"],
                rules_tree["artificial"], rules_tree["lengths"]):
            rules.append(Rule(attrs=tuple(islice(rule_attrs, int(length))),
                              stats=RuleStats(int(cls), float(conf), int(support)),
                              artificial=bool(artificial)))
        return rules

    @staticmethod
    def _disassemble_rules(rules: Iterable[Rule]) -> Mapping[str, numpy.ndarray]:
        def disassemble_rule(rule: Rule) -> Tuple[int, float, int, bool, numpy.ndarray,
                                                  numpy.ndarray, numpy.ndarray, numpy.ndarray]:
            rule_len = len(rule.attrs)
            features, cmps, thresholds = zip(*rule.attrs)
            features = numpy.fromiter(features, numpy.uint16, rule_len)
            cmps = numpy.fromiter(cmps, numpy.bool, rule_len)
            thresholds = numpy.fromiter(thresholds, numpy.float32, rule_len)
            return (rule.stats.cls, rule.stats.conf, rule.stats.support, rule.artificial, features,
                    cmps, thresholds, rule_len)

        disassembled_rules = list(zip(*[disassemble_rule(rule) for rule in rules]))
        return dict(
            cls=numpy.array(disassembled_rules[0], dtype=numpy.uint16),
            conf=numpy.array(disassembled_rules[1], dtype=numpy.float32),
            support=numpy.array(disassembled_rules[2], dtype=numpy.uint16),
            artificial=numpy.array(disassembled_rules[3], dtype=numpy.bool),
            features=numpy.concatenate(disassembled_rules[4]),
            cmps=numpy.concatenate(disassembled_rules[5]),
            thresholds=numpy.concatenate(disassembled_rules[6]),
            lengths=numpy.array(disassembled_rules[7], dtype=numpy.uint16),
        )
