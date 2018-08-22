import io
from itertools import islice
from typing import Dict, List, Iterable  # noqa: F401

import numpy

from lookout.core.analyzer import AnalyzerModel
from lookout.style.format.rules import Rules, Rule, RuleAttribute, RuleStats


class FormatModel(AnalyzerModel):
    """
    A modelforge model to store Rules instances.
    It is required to store all the Rules for different programming languages in a single model,
    named after each language.
    Note that Rules must be fitted and Rules.base_model is not saved.
    """
    NAME = "code-format"
    VENDOR = "source{d}"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._rules_by_lang = {}  # type: Dict[str, Rules]

    @property
    def languages(self):
        return sorted(self._rules_by_lang)

    def dump(self) -> str:
        result = io.StringIO()
        result.write(super().dump())
        for lang, rules in sorted(self._rules_by_lang.items()):
            result.write("\n\n# %s\n%s" % (lang, rules))
        return result.getvalue()

    def _generate_tree(self) -> dict:
        languages = self.languages
        return dict(
            languages=languages,
            paramss=[self[lang].origin for lang in languages],
            ruless=[self._disassemble_rules(self[lang].rules) for lang in languages],
        )

    def _load_tree(self, tree: dict) -> None:
        for name, params, rules in zip(tree["languages"], tree["paramss"], tree["ruless"]):
            self[name] = Rules(self._assemble_rules(rules), params)

    def __len__(self) -> int:
        return len(self._rules_by_lang)

    def __getitem__(self, lang: str) -> "Rules":
        """
        Get the Rules estimator by its language.
        :param lang: Estimator language.
        :return: Rules estimator instance.
        """
        return self._rules_by_lang[lang]

    def __setitem__(self, lang: str, rules: "Rules"):
        """
        Set a new Rules estimator to the model by its language.
        """
        self._rules_by_lang[lang] = rules

    def __iter__(self):
        yield from self._rules_by_lang.__iter__()

    def __contains__(self, item):
        return item in self._rules_by_lang

    @staticmethod
    def _assemble_rules(rules_tree: dict) -> List[Rule]:
        rules = []
        rule_attrs = (RuleAttribute(*params) for params in
                      zip(rules_tree["features"],  rules_tree["cmps"], rules_tree["thresholds"]))
        for cls, conf, length in zip(rules_tree["cls"], rules_tree["conf"], rules_tree["lengths"]):
            rules.append(Rule(tuple(islice(rule_attrs, int(length))), RuleStats(cls, conf)))
        return rules

    @staticmethod
    def _disassemble_rules(rules: Iterable[Rule]):
        def disassemble_rule(rule: Rule) -> tuple:
            rule_len = len(rule.attrs)
            features, cmps, thresholds = zip(*rule.attrs)
            features = numpy.fromiter(features, numpy.uint16, rule_len)
            cmps = numpy.fromiter(cmps, numpy.bool, rule_len)
            thresholds = numpy.fromiter(thresholds, numpy.float32, rule_len)
            return rule.stats.cls, rule.stats.conf, features, cmps, thresholds, rule_len

        disassembled_rules = list(zip(*[disassemble_rule(rule) for rule in rules]))
        return dict(
            cls=numpy.array(disassembled_rules[0], dtype=numpy.uint16),
            conf=numpy.array(disassembled_rules[1], dtype=numpy.float32),
            features=numpy.concatenate(disassembled_rules[2]),
            cmps=numpy.concatenate(disassembled_rules[3]),
            thresholds=numpy.concatenate(disassembled_rules[4]),
            lengths=numpy.array(disassembled_rules[5], dtype=numpy.uint16),
        )
