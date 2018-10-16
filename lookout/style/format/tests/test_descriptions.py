from math import ceil
from random import randint
import unittest

from lookout.style.format.analyzer import FormatAnalyzer
from lookout.style.format.descriptions import describe_rule, CLASS_REPRESENTATIONS
from lookout.style.format.feature_extractor import FeatureExtractor, FeatureGroup
from lookout.style.format.feature_utils import CLASSES
from lookout.style.format.rules import Rule, RuleStats


class DescriptionsTests(unittest.TestCase):

    @classmethod
    def return_feature(cls, feature_name):
        return ("node.0.%s" % feature_name,
                cls.feature_to_indices[feature_name],
                cls.fe._features[feature_name])

    @classmethod
    def setUpClass(cls):
        config = FormatAnalyzer._load_train_config({
            "javascript": {
                "feature_extractor": {
                    "left_siblings_window": 0,
                    "right_siblings_window": 0,
                    "parents_depth": 0,
                    "node_features": ["start_line", "label", "roles"]
                }
            }
        })
        cls.fe = FeatureExtractor(language="javascript",
                                  **config["javascript"]["feature_extractor"])
        cls.feature_to_indices = cls.fe.feature_to_indices[FeatureGroup.node][0]
        cls.ordinal = cls.return_feature("start_line")
        cls.categorical = cls.return_feature("label")
        cls.bag = cls.return_feature("roles")

    def test_describe_rule_ordinal(self):
        name, indices, feature = self.ordinal
        picked_class = randint(0, len(CLASSES) - 1)
        picked_class_name = CLASS_REPRESENTATIONS[picked_class]
        index = indices[0]
        rule = Rule([(index, True, 4.5)], RuleStats(picked_class, 0.9, 150))
        self.assertEqual(describe_rule(rule, self.fe),
                         "%s ≥ %d\n"
                         "	→ y = %s\n"
                         "	Confidence: 0.900. Support: 150." % (
                             name, ceil(4.5), picked_class_name
                         ))

    def test_describe_rule_categorical(self):
        name, indices, feature = self.categorical
        activated = randint(0, len(indices) - 1)
        activated_name = feature.names[activated]
        picked_class = randint(0, len(CLASSES) - 1)
        picked_class_name = CLASS_REPRESENTATIONS[picked_class]
        index = indices[activated]
        rule = Rule([(index, True, 0.5)], RuleStats(picked_class, 0.9, 150))
        self.assertEqual(describe_rule(rule, self.fe),
                         "%s = %s\n"
                         "	→ y = %s\n"
                         "	Confidence: 0.900. Support: 150." % (
                             name, activated_name, picked_class_name
                         ))

    def test_describe_rule_bag(self):
        name, indices, feature = self.bag
        activated = randint(0, len(indices) - 1)
        activated_name = feature.names[activated]
        not_activated = randint(0, len(indices) - 1)
        not_activated_name = feature.names[not_activated]
        picked_class = randint(0, len(CLASSES) - 1)
        picked_class_name = CLASS_REPRESENTATIONS[picked_class]
        index = indices[activated]
        not_index = indices[not_activated]
        rule = Rule([(index, True, 0.5), (not_index, False, 0.5)],
                    RuleStats(picked_class, 0.9, 150))
        self.assertEqual(describe_rule(rule, self.fe),
                         "%s in {%s} and not in {%s}\n"
                         "	→ y = %s\n"
                         "	Confidence: 0.900. Support: 150." % (
                             name, activated_name, not_activated_name, picked_class_name
                         ))


if __name__ == "__main__":
    unittest.main()
