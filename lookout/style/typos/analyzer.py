"""Identifier typos analyzer."""
from collections import defaultdict
import logging
from typing import Any, Dict, Iterable, Iterator, List, Mapping, NamedTuple, Sequence, Tuple

import bblfsh
from lookout.core.analyzer import Analyzer, AnalyzerModel, DummyAnalyzerModel, ReferencePointer
from lookout.core.api.service_analyzer_pb2 import Comment
from lookout.core.data_requests import DataService, \
    with_changed_uasts_and_contents, with_uasts_and_contents
from lookout.core.lib import extract_changed_nodes, files_by_language, filter_files, find_new_lines
from lookout.sdk.service_data_pb2 import Change, File
import pandas
from sourced.ml.algorithms import TokenParser, uast2sequence

from lookout.style.common import merge_dicts
from lookout.style.format.utils import generate_comment
from lookout.style.typos.corrector_manager import TyposCorrectorManager
from lookout.style.typos.utils import Candidate, Columns, flatten_df_by_column

TypoFix = NamedTuple("LineFix", (
    ("head_file", File),  # file from head revision
    ("line_number", int),  # line number for the comment
    ("token", str),  # token where typo is found
    ("candidates", List[Candidate]),  # Candidates for token fix
))


class IdTyposAnalyzer(Analyzer):
    """
    Identifier typos analyzer.
    """

    _log = logging.getLogger("IdTyposAnalyzer")
    model_type = DummyAnalyzerModel
    name = "lookout.style.typos"
    version = 1
    description = "Corrector of typos in source code identifiers."
    corrector_manager = TyposCorrectorManager()

    default_config = {
        "line_length_limit": 500,
        "n_candidates": 3,
        "confidence_threshold": 0.1,
        "overall_size_limit": 5 << 20,  # 5 MB
        "model": "d798e898-c6b2-4e39-809f-f502571584e8",
        "index_column": "index",
    }

    def __init__(self, model: AnalyzerModel, url: str, config: Mapping[str, Any]):
        """
        Initialize a new instance of IdTyposAnalyzer.

        :param model: The instance of the model loaded from the repository or freshly trained.
        :param url: The analyzed project's Git remote.
        :param config: Configuration of the analyzer of unspecified structure.
        """
        super().__init__(model, url, config)
        self.config = self._load_config(config)
        self.model = self.corrector_manager.get(self.config["model"])
        self.parser = self.create_token_parser()

    @staticmethod
    def create_token_parser() -> TokenParser:
        """
        Create instance of TokenParser that should be used by IdTyposAnalyzer.

        :return: TokenParser.
        """
        return TokenParser(stem_threshold=1000, single_shot=True, min_split_length=1)

    @with_changed_uasts_and_contents
    def analyze(self, ptr_from: ReferencePointer, ptr_to: ReferencePointer,
                data_service: DataService, changes: Iterable[Change], **data) -> List[Comment]:
        """
        Return the list of `Comment`-s - found typo corrections.

        :param ptr_from: The Git revision of the fork point. Exists in both the original and \
                         the forked repositories.
        :param ptr_to: The Git revision to analyze. Exists only in the forked repository.
        :param data_service: The channel to the data service in Lookout server to query for \
                             UASTs, file contents, etc.
        :param changes: Iterator of changes from the data service.
        :param data: Extra data passed into the method. Used by the decorators to simplify \
                     the data retrieval.
        :return: List of found review suggestions. Refer to \
                 lookout/core/server/sdk/service_analyzer.proto.
        """
        return [generate_comment(
            filename=typo_fix.head_file.path,
            line=typo_fix.line_number,
            text=self.render_comment_text(typo_fix),
            confidence=self._get_comment_confidence(typo_fix.candidates))
            for typo_fix in self.generate_typos_fixes(list(changes))]

    def generate_typos_fixes(self, changes: Sequence[Change]) -> Iterator[TypoFix]:
        """
        Generate all data about typo fix required for any type of further processing.

        The processing can be comment generation or performance report generation.

        :param changes: The list of changes in the pointed state.
        :return: Iterator with unrendered data per comment.
        """
        base_files_by_lang = files_by_language(c.base for c in changes)
        head_files_by_lang = files_by_language(c.head for c in changes)
        for lang, head_files in head_files_by_lang.items():
            for file in filter_files(
                    files=head_files, line_length_limit=self.config["line_length_limit"],
                    overall_size_limit=self.config["overall_size_limit"], log=self._log):
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
                        yield TypoFix(
                            head_file=file,
                            token=new_identifiers[index].token,
                            candidates=[Candidate(*c[:2]) for c in corrections[token]],
                            line_number=new_identifiers[index].start_position.line,
                        )

    def render_comment_text(self, typo_fix: TypoFix) -> str:
        """
        Generate the text of the comment for the specified typo fix.

        :param typo_fix: Information about typo fix required to render a comment text.
        :return: string with the generated comment.
        """
        # TODO: move this logic to template. See FormatAnalyzer.render_comment_text()
        corrections_line = ", ".join("%s (%d%%)" % (
            c.token, int(c.confidence * 100)) for c in typo_fix.candidates)
        return "Possible typo in \"%s\". Suggestions: %s" % (typo_fix.token, corrections_line)

    @staticmethod
    def _get_comment_confidence(suggestions: Sequence[Candidate]) -> int:
        invert_confidence = 1
        for suggestion in suggestions:
            invert_confidence *= (1 - suggestion.confidence)
        return int(100 * (1 - invert_confidence))

    @staticmethod
    def reconstruct_identifier(tokenizer: TokenParser, pred_tokens: List[str], identifier: str) \
            -> str:
        """
        Reconstruct identifier given predicted tokens and initial identifier.

        :param tokenizer: tokenizer - instance of TokenParser.
        :param pred_tokens: list of predicted tokens.
        :param identifier: identifier.
        :return: reconstructed identifier based on predicted tokens.
        """
        identifier_l = identifier.lower()
        # setup required parameters
        assert tokenizer._single_shot, "TokenParser should be initialized with " \
                                       "`single_shot=True` for IdTyposAnalyzer"
        # sanity checking
        initial_tokens = list(tokenizer.split(identifier))
        err = "Number of predicted tokens (%s) not equal to the number of tokens in the " \
              "identifier (%s) for identifier '%s', predicted_tokens '%s', tokens in identifier " \
              "'%s'"
        assert len(initial_tokens) == len(pred_tokens), \
            err % (len(initial_tokens), len(pred_tokens), identifier, pred_tokens, initial_tokens)
        # reconstruction
        res = []
        prev_end = 0
        for token, pred_token in zip(initial_tokens, pred_tokens):
            curr = identifier_l.find(token, prev_end)
            assert curr != -1, "TokenParser is broken, the subtoken `%s` was not found in the " \
                               "identifier `%s`" % (token, identifier)
            if curr != prev_end:
                # delimiter found
                res.append(identifier[prev_end:curr])
            if identifier[curr:curr + len(token)].isupper():
                # upper case
                res.append(pred_token.upper())
            elif identifier[curr:curr + len(token)][0].isupper():
                # capitalized
                res.append(pred_token[0].upper() + pred_token[1:])
            else:
                res.append(pred_token)
            prev_end = curr + len(token)
        if prev_end != len(identifier):
            # suffix
            res.append(identifier[prev_end:])
        return "".join(res)

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
        df = pandas.DataFrame(columns=[self.config["index_column"], Columns.Split])
        df[self.config["index_column"]] = range(len(identifiers))
        df[Columns.Split] = [" ".join(self.parser.split(i)) for i in identifiers]
        df = flatten_df_by_column(df, Columns.Split, Columns.Token, str.split)
        suggestions = self.model.suggest(df, n_candidates=self.config["n_candidates"],
                                         return_all=False)
        suggestions = self.filter_suggestions(df, suggestions)
        grouped_suggestions = defaultdict(dict)
        for index, row in df.iterrows():
            if index in suggestions.keys():
                grouped_suggestions[row[self.config["index_column"]]][row[Columns.Token]] = \
                    suggestions[index]
        return grouped_suggestions

    def filter_suggestions(self, test_df: pandas.DataFrame,
                           suggestions: Dict[int, List[Tuple[str, float]]],
                           ) -> Dict[int, List[Tuple[str, float]]]:
        """
        Filter suggestions based on the repo specifics and confidence threshold.

        :param test_df: DataFrame with info about tested tokens.
        :param suggestions: Dictionary of correction suggestions grouped by \
                            typoed token index in test_df.
        :return: Dictionary of filtered suggestions grouped by checked token's index in test_df.
        """
        filtered_suggestions = {}
        tokens = test_df[Columns.Token]
        for index, candidates in suggestions.items():
            filtered_candidates = []
            for candidate in candidates:
                if candidate[0] == tokens[index] or \
                        candidate[1] < self.config["confidence_threshold"]:
                    break
                filtered_candidates.append(candidate)
            if filtered_candidates:
                filtered_suggestions[index] = filtered_candidates
        return filtered_suggestions

    @classmethod
    def _load_config(cls, config: Mapping[str, Any]) -> Mapping[str, Any]:
        """
        Merge provided config with the default values.

        :param config: User-defined config.
        :return: Full config.
        """
        return merge_dicts(cls.default_config, config)
