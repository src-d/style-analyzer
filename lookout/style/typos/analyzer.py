"""Identifier typos analyzer."""
from collections import defaultdict
import logging
from typing import Any, Dict, List, Mapping, Tuple

import bblfsh
from lookout.core.analyzer import Analyzer, AnalyzerModel, DummyAnalyzerModel, ReferencePointer
from lookout.core.api.service_analyzer_pb2 import Comment
from lookout.core.data_requests import DataService, \
    with_changed_uasts_and_contents, with_uasts_and_contents
from lookout.core.lib import extract_changed_nodes, files_by_language, filter_files, find_new_lines
import pandas
from sourced.ml.algorithms import TokenParser, uast2sequence

from lookout.style.typos.corrector_manager import TyposCorrectorManager
from lookout.style.typos.utils import flatten_data, SPLIT_COLUMN, TYPO_COLUMN


class IdTyposAnalyzer(Analyzer):
    """
    Identifier typos analyzer.
    """

    log = logging.getLogger("IdTyposAnalyzer")
    model_type = None
    version = 1
    description = "Corrector of typos in source code identifiers."
    corrector_manager = TyposCorrectorManager()

    DEFAULT_LINE_LENGTH_LIMIT = 500
    DEFAULT_N_CANDIDATES = 3
    DEFAULT_CONFIDENCE_THRESHOLD = 0.1
    INDEX_COLUMN = "index"

    def __init__(self, model: AnalyzerModel, url: str, config: Mapping[str, Any]):
        """
        Initialize a new instance of IdTyposAnalyzer.

        :param model: The instance of the model loaded from the repository or freshly trained.
        :param url: The analyzed project's Git remote.
        :param config: Configuration of the analyzer of unspecified structure.
        """
        super().__init__(model, url, config)
        self.model = self.corrector_manager.get(config.get("model"))
        self.n_candidates = config.get("n_candidates", self.DEFAULT_N_CANDIDATES)
        self.confidence_threshold = config.get(
            "confidence_threshold", self.DEFAULT_CONFIDENCE_THRESHOLD)
        self.parser = TokenParser(stem_threshold=40, single_shot=True)

    @with_changed_uasts_and_contents
    def analyze(self, ptr_from: ReferencePointer, ptr_to: ReferencePointer,
                data_service: DataService, **data) -> [Comment]:
        """
        Return the list of `Comment`-s - found typo corrections.

        :param ptr_from: The Git revision of the fork point. Exists in both the original and \
                         the forked repositories.
        :param ptr_to: The Git revision to analyze. Exists only in the forked repository.
        :param data_service: The channel to the data service in Lookout server to query for \
                             UASTs, file contents, etc.
        :param data: Extra data passed into the method. Used by the decorators to simplify \
                     the data retrieval.
        :return: List of found review suggestions. Refer to \
                 lookout/core/server/sdk/service_analyzer.proto.
        """
        log = self.log
        comments = []
        changes = list(data["changes"])
        base_files_by_lang = files_by_language(c.base for c in changes)
        head_files_by_lang = files_by_language(c.head for c in changes)
        line_length = self.config.get("line_length_limit", self.DEFAULT_LINE_LENGTH_LIMIT)
        for lang, head_files in head_files_by_lang.items():
            for file in filter_files(head_files, line_length, log):
                try:
                    prev_file = base_files_by_lang[lang][file.path]
                except KeyError:
                    lines = []
                    old_identifiers = set()
                else:
                    lines = find_new_lines(prev_file, file)
                    old_identifiers = {
                        node.token for node in uast2sequence(prev_file.uast)
                        if bblfsh.role_id("IDENTIFIER") in node.roles
                        and bblfsh.role_id("IMPORT") not in node.roles and node.token
                    }
                changed_nodes = extract_changed_nodes(file.uast, lines)
                new_identifiers = [node for node in changed_nodes
                                   if bblfsh.role_id("IDENTIFIER") in node.roles and
                                   bblfsh.role_id("IMPORT") not in node.roles and
                                   node.token and node.token not in old_identifiers]
                if not new_identifiers:
                    continue
                suggestions = self.check_identifiers([n.token for n in new_identifiers])
                for index in suggestions.keys():
                    corrections = suggestions[index]
                    for token in corrections.keys():
                        comment = Comment()
                        comment.file = file.path
                        corrections_line = " " + ", ".join(
                            "%s (%d%%)" % (candidate[0], int(candidate[1] * 100))
                            for candidate in corrections[token])
                        comment.text = """
                            Possible typo in \"%s\". Suggestions:
                        """.strip() % new_identifiers[index].token + corrections_line
                        comment.line = new_identifiers[index].start_position.line
                        comment.confidence = int(corrections[token][0][1] * 100)
                        comments.append(comment)
        return comments

    @classmethod
    @with_uasts_and_contents
    def train(cls, ptr: ReferencePointer, config: dict, data_service: DataService,
              **data) -> AnalyzerModel:
        """
        Generate a new model on top of the specified source code.

        :param ptr: Git repository state pointer.
        :param config: Configuration of the training of unspecified structure.
        :param data_service: The channel to the data service in Lookout server to query for \
                             UASTs, file contents, etc.
        :param data: Extra data passed into the method. Used by the decorators to simplify \
                     the data retrieval.
        :return: Instance of `AnalyzerModel` (`model_type`, to be precise).
        """
        return DummyAnalyzerModel()

    def check_identifiers(self, identifiers: List[str],
                          ) -> Dict[int, Dict[str, List[Tuple[str, float]]]]:
        """
        Check tokens from identifiers for typos.

        :param identifiers: List of identifiers to check.
        :return: Dictionary of corrections grouped by ids of corresponding identifier \
                 in 'identifiers' and typoed tokens which have correction suggestions.
        """
        df = pandas.DataFrame(columns=[self.INDEX_COLUMN, SPLIT_COLUMN])
        df[self.INDEX_COLUMN] = range(len(identifiers))
        df[SPLIT_COLUMN] = [" ".join(self.parser.split(i)) for i in identifiers]
        df = flatten_data(df, new_column_name=TYPO_COLUMN)
        suggestions = self.model.suggest(df, n_candidates=self.n_candidates, return_all=False)
        suggestions = self.filter_suggestions(df, suggestions)
        grouped_suggestions = defaultdict(dict)
        for index, row in df.iterrows():
            if index in suggestions.keys():
                grouped_suggestions[row[self.INDEX_COLUMN]][row[TYPO_COLUMN]] = suggestions[index]
        return grouped_suggestions

    def filter_suggestions(self, test_df: pandas.DataFrame,
                           suggestions: Dict[int, List[Tuple[str, float]]],
                           ) -> Dict[int, List[Tuple[str, float]]]:
        """
        Filter suggestions based on the repo specifics and confidence threshold.

        :param test_df: DataFrame with info about tested tokens.
        :param suggestions: Dictionary of correction suggestions grouped by \
                            typoed token index in test_df.
        :return: Dictionary of filtered suggestions grouped by typoed token index in test_df.
        """
        filtered_suggestions = {}
        tokens = test_df.typo
        for index, candidates in suggestions.items():
            filtered_candidates = []
            for candidate in candidates:
                if candidate[0] == tokens[index] or candidate[1] < self.confidence_threshold:
                    break
                filtered_candidates.append(candidate)
            if filtered_candidates:
                filtered_suggestions[index] = filtered_candidates
        return filtered_suggestions
