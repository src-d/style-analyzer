"""Features and rules description utils."""
from math import ceil, floor
from typing import List, Sequence

from numpy import flatnonzero, ndarray
from sklearn.exceptions import NotFittedError
import xxhash

from lookout.style.format.classes import CLASS_INDEX, CLS_NEWLINE, CLS_NOOP
from lookout.style.format.feature_extractor import FeatureExtractor, FEATURES_MAX, FEATURES_MIN
from lookout.style.format.features import BagFeature, CategoricalFeature, Feature, OrdinalFeature
from lookout.style.format.model import FormatModel
from lookout.style.format.rules import Rule, RuleAttribute
from lookout.style.format.virtual_node import VirtualNode


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
    if feature_extractor.features is None or feature_extractor.index_to_feature is None:
        raise NotFittedError()
    attr_descriptions = describe_rule_attrs(rule, feature_extractor)
    composite_class_repr = feature_extractor.composite_class_representations
    return "  %s\n⇒ y = %s\nConfidence: %.3f. Support: %d." % (
        "\n\t∧ ".join(attr_descriptions),
        "".join(composite_class_repr[rule.stats.cls]),
        rule.stats.conf,
        rule.stats.support)


def hash_rule(rule: Rule, feature_extractor: FeatureExtractor) -> str:
    """
    Hash rule contents to 8 hex characters. The same content produces the same hash all the time.

    :param rule: The rule to describe.
    :param feature_extractor: The FeatureExtractor used to create those rules.
    :return: String of length 8.
    """
    hasher = xxhash.xxh32(seed=7)
    for attr in describe_rule_attrs(rule, feature_extractor):
        hasher.update(attr.strip())
    hasher.update(feature_extractor.composite_class_representations[rule.stats.cls])
    return hasher.hexdigest()


def describe_rule_attrs(rule: Rule, feature_extractor: FeatureExtractor) -> Sequence[str]:
    """
    Format the rule as text.

    We take features metadata to convert the integer indices to human-readable names.

    :param rule: The rule to describe.
    :param feature_extractor: The FeatureExtractor used to create those rules.
    :return: The description of the rule.
    """
    result = []

    for feature, feature_id, splits, node_index, group in rule.group_features(feature_extractor):
        desc = describe_rule_splits(
            feature, "%s%s" % (group.format(node_index), feature_id.name), splits)
        result.append(desc)

    return result


def describe_sample_bag(feature: BagFeature, values: ndarray) -> str:
    """
    Describe a bag sample given its feature values.

    :param feature: The feature that computed the values to describe.
    :param values: The values to describe.
    :return: A string that describe the values of this feature.
    """
    selected_names = feature.selected_names
    if not selected_names:
        return "unselected"
    active = flatnonzero(values)
    return "{%s}" % ", ".join(selected_names[index]
                              for index in active) if len(active) else "∅"


def describe_sample_categorical(feature: CategoricalFeature, values: ndarray) -> str:
    """
    Describe a categorical sample given its feature values.

    :param feature: The feature that computed the values to describe.
    :param values: The values to describe.
    :return: A string that describe the values of this feature.
    """
    selected_names = feature.selected_names
    if not selected_names:
        return "unselected"
    active = flatnonzero(values)
    return selected_names[active[0]] if len(active) else "∅"


def describe_sample_ordinal(feature: OrdinalFeature, values: ndarray) -> str:
    """
    Describe an ordinal sample given its feature value.

    :param feature: The feature that computed the values to describe.
    :param values: The value to describe, in an array.
    :return: A string that describe the value of this feature.
    """
    if not feature.selected_names:
        return "unselected"
    return str(values[0])


SAMPLE_DESCRIBERS = {
    BagFeature: describe_sample_bag,
    CategoricalFeature: describe_sample_categorical,
    OrdinalFeature: describe_sample_ordinal,
}


def describe_sample(feature: Feature, values: ndarray) -> str:
    """
    Describe a sample given its feature value.

    :param feature: The feature that computed the values to describe.
    :param values: The value to describe, in an array.
    :return: A string that describe the value of this feature.
    """
    for cls in type(feature).__mro__:
        try:
            return SAMPLE_DESCRIBERS[cls](feature, values)
        except KeyError:
            continue
    raise KeyError("no sample describer is registered for %s" % type(feature).__name__)


def _describe_rule_splits_bag(feature: BagFeature, name: str, splits: List[RuleAttribute]) -> str:
    """
    Describe parts of a bag rule in natural language.

    :param feature: The feature used for the splits to describe.
    :param name: The name to use for the feature used in the split.
    :param splits: List of tuples representing the splits to describe. The tuples contain the \
                   comparison, the threshold and the index of the feature used, useful in case of \
                   multi-values features.
    :return: A string describing the given rule splits.
    """
    included = set()
    excluded = set()
    for index, cmp, _ in splits:
        if cmp:
            included.add(feature.names[index])
        else:
            excluded.add(feature.names[index])
    description = name
    if included:
        description += " in {%s}" % ", ".join(sorted(included))
        if excluded:
            description += " and"
    if excluded:
        description += " not in {%s}" % ", ".join(sorted(excluded))
    return description


def _describe_rule_splits_categorical(feature: CategoricalFeature, name: str,
                                      splits: List[RuleAttribute]) -> str:
    """
    Describe parts of a categorical rule in natural language.

    :param feature: The feature used for the splits to describe.
    :param name: The name to use for the feature used in the split.
    :param splits: List of tuples representing the splits to describe. The tuples contain the \
                   comparison, the threshold and the index of the feature used, useful in case of \
                   multi-values features.
    :return: A string describing the given rule splits.
    """
    included = None
    excluded = set()
    for index, cmp, _ in splits:
        if cmp:
            included = feature.names[index]
        else:
            excluded.add(feature.names[index])
    description = name
    if included:
        description += " = %s" % included
    if excluded and not included:
        description += " not in {%s}" % ", ".join(sorted(excluded))
    return description


def _describe_rule_splits_ordinal(_, name: str, splits: List[RuleAttribute]) -> str:
    """
    Describe a part of an ordinal rule in natural language.

    :param _: The feature used for the splits to describe.
    :param name: The name to use for the feature used in the split.
    :param splits: List of the tuple representing the splits to describe. The tuples contain the \
                   comparison, the threshold and an ignored value here to be consistent with \
                   other types of features. The wrapping list is also needed for this reason.
    :return: A string describing the given rule splits.
    """
    _, cmp, threshold = splits[0]
    if cmp:
        if threshold > FEATURES_MAX - 1:
            return "%s = %d" % (name, FEATURES_MAX)
        return "%s ≥ %d" % (name, ceil(threshold))
    elif threshold < FEATURES_MIN + 1:
        return "%s = %d" % (name, FEATURES_MIN)
    return "%s ≤ %d" % (name, floor(threshold))


RULE_SPLITS_DESCRIBERS = {
    BagFeature: _describe_rule_splits_bag,
    CategoricalFeature: _describe_rule_splits_categorical,
    OrdinalFeature: _describe_rule_splits_ordinal,
}


def describe_rule_splits(feature: Feature, name: str, splits: List[RuleAttribute]) -> str:
    """
    Describe a part of a rule in natural language.

    :param feature: The feature used for the splits to describe.
    :param name: The name to use for the feature used in the split.
    :param splits: List of the tuple representing the splits to describe. The tuples contain the \
                   comparison, the threshold and an ignored value here to be consistent with \
                   other types of features. The wrapping list is also needed for this reason.
    :return: A string describing the given rule splits.
    """
    for cls in type(feature).__mro__:
        try:
            return RULE_SPLITS_DESCRIBERS[cls](feature, name, splits)
        except KeyError:
            continue
    raise KeyError("no rule splits describer is registered for %s" % type(feature).__name__)


def get_change_description(vnode: VirtualNode, feature_extractor: FeatureExtractor) -> str:
    """
    Return the comment with regard to the correct node class.

    :param vnode: Changed node. "y" attribute is the predicted node label and \
                  "y_old" is the original one.
    :param feature_extractor: FeatureExtractor used to extract features.
    :return: String comment.
    """
    if not hasattr(vnode, "y_old"):
        raise ValueError("y_old attribute must exist in the supplied vnode")
    column = vnode.start.col
    class_representations = feature_extractor.composite_class_representations
    old_label = class_representations[feature_extractor.class_sequences_to_labels[vnode.y_old]]
    new_label = class_representations[feature_extractor.class_sequences_to_labels[vnode.y]]
    if vnode.y[0] == CLASS_INDEX[CLS_NOOP]:
        if CLASS_INDEX[CLS_NEWLINE] in vnode.y_old:
            return "Redundant line break. Please concatenate with the previous line."
        else:
            return "%s at column %d should be removed." % (old_label, column)
    if vnode.y_old[0] == CLASS_INDEX[CLS_NOOP]:
        return "%s should be inserted at column %d." % (new_label, column)
    if CLASS_INDEX[CLS_NEWLINE] in vnode.y_old:
        return "Replace %s with %s in the beginning of the line" % (old_label, new_label)
    return "Replace %s with %s at column %d." % (old_label, new_label, column)


def get_code_chunk(code_lines: Sequence[str], line_number: int) -> str:
    """
    Return nice code snippet that can be inserted to github message.

    :param code_lines: Sequence of code lines without ending new line character.
    :param line_number: 1-based line number to print.
    :return: Code snippet.
    """
    lines = list(range(max(0, line_number - 2), line_number + 1))
    return "\n".join("%d|%s" % (l, code_lines[l]) for l in lines)


def dump_rule(model: FormatModel, rule_hash: str):
    """
    Print the rule contained in the model by hash.

    :param model: Trained model.
    :param rule_hash: 8-char rule hash.
    :return: Nothing
    """
    for lang in model.languages:
        rules = model[lang]
        fe = FeatureExtractor(language=lang, **rules.origin_config["feature_extractor"])
        for rule in rules.rules:
            h = hash_rule(rule, fe)
            if h == rule_hash:
                print(lang)
                print("    " + describe_rule(rule, fe).replace("\t", "    "))
