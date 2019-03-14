import os
import pkgutil
from typing import Set

from lookout.style.format import langs


def get_supported_languages() -> Set[str]:
    """
    Return a list of supported languages by style-analyzer.
    """
    return set(modname for _, modname, ispkg in pkgutil.iter_modules(langs.__path__) if ispkg)


DEFAULT_CONFIG = {
    "common": {
        # Shared settings for train and analyze should be here
    },
    "train": {
        "language_defaults": {
            "feature_extractor": {
                "left_siblings_window": 5,
                "right_siblings_window": 5,
                "parents_depth": 2,
                "node_features": ["start_line", "start_col"],
                "left_features": ["length", "diff_offset", "diff_col", "diff_line",
                                  "internal_type", "label", "reserved", "roles"],
                "right_features": ["length", "internal_type", "reserved", "roles"],
                "parent_features": ["internal_type", "roles"],
                "no_labels_on_right": True,
                "debug_parsing": False,
                "select_features_number": 500,
                "return_sibling_indices": False,
                "cutoff_label_support": 80,
            },
            "trainable_rules": {
                "prune_branches_algorithms": ["reduced-error"],
                "top_down_greedy_budget": [False, .5],
                "prune_attributes": True,
                "attribute_similarity_threshold": 0.98,
                "confidence_threshold": 0.8,
                "prune_dataset_ratio": .2,
                "n_estimators": 10,
            },
            "optimizer": {
                "n_iter": 50,
                "cv": 3,
                "n_jobs": -1,
                "base_model_name_categories": ["sklearn.ensemble.RandomForestClassifier",
                                               "sklearn.tree.DecisionTreeClassifier"],
                "max_depth_categories": [None, 5, 10],
                "max_features_categories": [None, "auto"],
                "min_samples_leaf_min": 90,
                "min_samples_leaf_max": 120,
                "min_samples_split_min": 180,
                "min_samples_split_max": 240,
            },
            "random_state": 42,
            "test_dataset_ratio": 0.0,
            "line_length_limit": 500,
            "lower_bound_instances": 500,
            "overall_size_limit": 5 << 20,  # 5 MB
            "lines_ratio_train_trigger": 0.2,
        },
    },
    "analyze": {
        "language_defaults": {
            "confidence_threshold": 0.92,
            "support_threshold": 80,
            "report_code_lines": False,
            "report_triggered_rules": False,
            "report_parse_failures": False,
            "uast_break_check": True,
        },
        "comment_template": os.path.join(os.path.dirname(__file__), "templates", "comment.jinja2"),
    },
}
