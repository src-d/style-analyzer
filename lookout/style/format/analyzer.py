from collections import defaultdict
import logging
from pprint import pformat

from bblfsh import Node
from skopt import BayesSearchCV
from skopt.space import Categorical, Integer

from lookout.core.analyzer import Analyzer, AnalyzerModel, ReferencePointer
from lookout.core.api.service_analyzer_pb2 import Comment
from lookout.core.api.service_data_pb2_grpc import DataStub
from lookout.core.data_requests import with_changed_uasts_and_contents, with_uasts_and_contents
from lookout.style.format.features import FeatureExtractor
from lookout.style.format.model import FormatModel
from lookout.style.format.rules import TrainableRules


class FormatAnalyzer(Analyzer):
    log = logging.getLogger("FormatAnalyzer")
    model_type = FormatModel
    version = "1"
    description = "Source code formatting: whitespace, new lines, quotes, braces."

    @with_changed_uasts_and_contents
    def analyze(self, ptr_from: ReferencePointer, ptr_to: ReferencePointer,
                data_request_stub: DataStub, **data) -> [Comment]:
        changes = data["changes"]
        comments = []
        for change in changes:
            comment = Comment()
            comment.file = change.head.path
            comment.text = "%s %d > %d" % (change.head.language,
                                           self.count_nodes(change.base.uast),
                                           self.count_nodes(change.head.uast))
            comment.line = 1
            comment.confidence = 100
            comments.append(comment)
        return comments

    @classmethod
    @with_uasts_and_contents
    def train(cls, ptr: ReferencePointer, config: dict, data_request_stub: DataStub,
              **data) -> AnalyzerModel:
        """
        Train a model given the files available.

        :param url: url of the repository
        :param commit: hash of the commit that we analyze
        :param config: configuration dict
        :param data: data sent by the lookout server
        :return: a modelforge.Model containing the learned rules, per language.
        """
        final_config = {
            "siblings_window": 5,
            "parents_depth": 2,
            "lower_bound_instances": 500,
            "prune_branches_algorithms": ("reduced-error",),
            "top_down_greedy_budget": TrainableRules.TopDownGreedyBudget(False, .5),
            "prune_attributes": False,
            "uncertain_attributes": True,
            "prune_dataset_ratio": .2,
            "n_estimators": 10,
            "random_state": 42,
            "n_iter": 30
        }
        final_config.update(config)
        cls.log.info("train %s %s %s with config %s", ptr.url, ptr.commit, data,
                     pformat(final_config, width=4096, compact=True))
        files = data["files"]
        language_to_files = defaultdict(list)
        for file in files:
            if not len(file.uast.children):
                continue
            if file.language != "JavaScript":
                continue
            language_to_files["javascript"].append(file)
            cls.log.info("%s %s %d", file.path, file.language, len(file.uast.children))

        model = FormatModel().construct(cls, ptr)

        for language, files in language_to_files.items():
            cls.log.info("training on %d %s files", len(files), language)
            fe = FeatureExtractor(language=language,
                                  siblings_window=final_config["siblings_window"],
                                  parents_depth=final_config["parents_depth"])
            X, y = fe.extract_features(files)
            lower_bound_instances = final_config["lower_bound_instances"]
            if X.shape[0] < lower_bound_instances:
                cls.log.warning("skipped %s: too few samples (%d/%d)", language, X.shape[0],
                                lower_bound_instances)
                continue
            cls.log.debug("training rules model")
            bscv = BayesSearchCV(
                TrainableRules(
                    prune_branches_algorithms=final_config["prune_branches_algorithms"],
                    prune_attributes=final_config["prune_attributes"],
                    top_down_greedy_budget=final_config["top_down_greedy_budget"],
                    uncertain_attributes=final_config["uncertain_attributes"],
                    prune_dataset_ratio=final_config["prune_dataset_ratio"],
                    n_estimators=final_config["n_estimators"],
                    random_state=final_config["random_state"]),
                {"base_model_name": Categorical(["sklearn.ensemble.RandomForestClassifier",
                                                 "sklearn.tree.DecisionTreeClassifier"]),
                 "max_depth": Categorical([None, 5, 10]),
                 "max_features": Categorical([None, "auto"]),
                 "min_samples_split": Integer(2, 20),
                 "min_samples_leaf": Integer(1, 20)},
                n_jobs=-1,
                n_iter=final_config["n_iter"])
            bscv.fit(X, y)
            cls.log.debug("score of the best estimator found: %.3f", bscv.best_score_)
            cls.log.debug("params of the best estimator found: %s", str(bscv.best_params_))
            cls.log.debug("training the model with complete data")
            trainable_rules = TrainableRules(prune_branches_algorithms=["reduced-error"],
                                             prune_attributes=True, random_state=42,
                                             uncertain_attributes=True, **bscv.best_params_)
            trainable_rules.fit(X, y)
            model[language] = trainable_rules.rules
        cls.log.info("trained %s", model)
        return model

    @staticmethod
    def count_nodes(uast: Node):
        stack = [uast]
        count = 0
        while stack:
            node = stack.pop()
            count += 1
            stack.extend(node.children)
        return count
