"""Features and rules description utils."""
from collections import defaultdict
from copy import copy
from functools import singledispatch
from math import ceil, floor
from typing import List, Tuple

from numpy import flatnonzero, floating, ndarray

from lookout.style.format.feature_extractor import FeatureExtractor, FEATURES_MAX, FEATURES_MIN
from lookout.style.format.feature_utils import (
    CLASS_INDEX, CLASSES, CLS_DOUBLE_QUOTE, CLS_NEWLINE, CLS_NOOP, CLS_SINGLE_QUOTE, CLS_SPACE,
    CLS_SPACE_DEC, CLS_SPACE_INC, CLS_TAB, CLS_TAB_DEC, CLS_TAB_INC)
from lookout.style.format.features import BagFeature, CategoricalFeature, OrdinalFeature
from lookout.style.format.rules import Rule


_CLASS_REPRESENTATIONS_MAPPING = {
    CLS_DOUBLE_QUOTE: '"',
    CLS_NEWLINE: "⏎",
    CLS_NOOP: "∅",
    CLS_SINGLE_QUOTE: "'",
    CLS_SPACE: "␣",
    CLS_SPACE_DEC: "␣⁻",
    CLS_SPACE_INC: "␣⁺",
    CLS_TAB: "⇥",
    CLS_TAB_DEC: "⇥⁻",
    CLS_TAB_INC: "⇥⁺",
}
CLASS_REPRESENTATIONS = [_CLASS_REPRESENTATIONS_MAPPING[cls] for cls in CLASSES]
del _CLASS_REPRESENTATIONS_MAPPING

CLASS_PRINTABLES = copy(CLASS_REPRESENTATIONS)
CLASS_PRINTABLES[CLASS_INDEX[CLS_NEWLINE]] += "\n"


def describe_rules(rules: List[Rule], feature_extractor: FeatureExtractor) -> List[str]:
    """
    Format the rules as a list of human-readable descriptions.

    :param rules: The list of rules to describe.
    :param feature_extractor: The FeatureExtractor used to create those rules.
    :return: A list of rule descriptions.
    """
    return [describe_rule(rule, feature_extractor) for rule in rules]


def describe_rule(rule: Rule, feature_extractor: FeatureExtractor) -> str:
    """
    Format the rule as text.

    We take features metadata to convert the integer indices to human-readable names.

    :param rule: The rule to describe.
    :param feature_extractor: The FeatureExtractor used to create those rules.
    :return: The description of the rule.
    """
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for feature_index, cmp, threshold in rule.attrs:
        group, node_index, feature_name, index = feature_extractor.index_to_feature[feature_index]
        grouped[group][node_index][feature_name].append((cmp, threshold, index))
    descriptions = [
        describe_rule_splits(feature_extractor._features[feature_name],
                             "%s.%s.%s" % (group.name, node, feature_name),
                             splits)
        for group, nodes in grouped.items()
        for node, feature_names in nodes.items()
        for feature_name, splits in feature_names.items()]
    return "%s\n\t→ y = %s (%.2f confidence, %d support)" % (
        "\n\t∧ ".join(descriptions),
        CLASS_REPRESENTATIONS[rule.stats.cls],
        rule.stats.conf,
        rule.stats.support)


@singledispatch
def describe_sample(feature: BagFeature, values: ndarray) -> str:
    """
    Describe a sample given its feature values.

    :param feature: The feature that computed the values to describe.
    :param values: The values to describe.
    :return: A string that describe the values of this feature.
    """
    active = flatnonzero(values)[0]
    counts = ("%s: %d" % (feature.names[i], count) for i, count in zip(active, values[active]))
    return ", ".join(counts) if len(active) else "∅"


@describe_sample.register(CategoricalFeature)
def describe_sample_categorical(feature: CategoricalFeature, values: ndarray) -> str:
    """
    Describe a sample given its feature values.

    :param feature: The feature that computed the values to describe.
    :param values: The values to describe.
    :return: A string that describe the values of this feature.
    """
    active = flatnonzero(values)[0]
    return str(feature.names[values[active[0]]] if len(active) else "∅")


@describe_sample.register(OrdinalFeature)
def describe_sample_ordinal(feature: OrdinalFeature, values: ndarray) -> str:
    """
    Describe a sample given its feature value.

    :param feature: The feature that computed the values to describe.
    :param values: The value to describe, in an array.
    :return: A string that describe the value of this feature.
    """
    return str(values[0])


def _format_int(cmp: bool, threshold: floating, name: str) -> str:
    if cmp:
        if threshold > FEATURES_MAX - 1:
            return "%s = %d" % (name, FEATURES_MAX)
        return "%s ≥ %d" % (name, ceil(threshold))
    elif threshold < FEATURES_MIN + 1:
        return "%s = %d" % (name, FEATURES_MIN)
    return "%s ≤ %d" % (name, floor(threshold))


@singledispatch
def describe_rule_splits(feature: BagFeature, name: str,
                         splits: List[Tuple[bool, floating, int]]) -> str:
    """
    Describe parts of a rule in natural language.

    :param feature: The feature used for the splits to describe.
    :param name: The name to use for the feature used in the split.
    :param splits: List of tuples representing the splits to describe. The tuples contain the \
                   comparison, the threshold and the index of the feature used, useful in case of \
                   multi-values features.
    :return: A string describing the given rule splits.
    """
    return "%s = {%s}" % (name, ", ".join(_format_int(cmp, threshold, feature.names[index])
                                          for cmp, threshold, index in splits))


@describe_rule_splits.register(CategoricalFeature)
def describe_rule_parts_categorical(feature: CategoricalFeature, name: str,
                                    splits: List[Tuple[bool, floating, int]]) -> str:
    """
    Describe parts of a rule in natural language.

    :param feature: The feature used for the splits to describe.
    :param name: The name to use for the feature used in the split.
    :param splits: List of tuples representing the splits to describe. The tuples contain the \
                   comparison, the threshold and the index of the feature used, useful in case of \
                   multi-values features.
    :return: A string describing the given rule splits.
    """
    included = set()
    excluded = set()
    for cmp, _, index in splits:
        if cmp:
            included.add(feature.names[index])
        else:
            excluded.add(feature.names[index])
    description = name
    if included:
        description += " in {%s}" % ", ".join(included)
        if excluded:
            description += " and"
    if excluded:
        description += " not in {%s}" % ", ".join(excluded)
    return description


@describe_rule_splits.register(OrdinalFeature)
def describe_rule_parts_ordinal(feature: OrdinalFeature, name: str,
                                splits: List[Tuple[bool, floating, int]]) -> str:
    """
    Describe a part of a rule in natural language.

    :param feature: The feature used for the splits to describe.
    :param name: The name to use for the feature used in the split.
    :param splits: List of the tuple representing the splits to describe. The tuples contain the \
                   comparison, the threshold and an ignored value here to be consistent with \
                   other types of features. The wrapping list is also needed for this reason.
    :return: A string describing the given rule splits.
    """
    cmp, threshold, _ = splits[0]
    return _format_int(cmp, threshold, name)
