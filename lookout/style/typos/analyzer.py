"""Identifier typos analyzer."""
from collections import defaultdict
import itertools
import logging
import os
from pprint import pformat
from typing import Any, Dict, Iterable, Iterator, List, Mapping, NamedTuple, Sequence, Tuple

import bblfsh
from lookout.core.analyzer import Analyzer, ReferencePointer
from lookout.core.api.service_analyzer_pb2 import Comment
from lookout.core.api.service_data_pb2 import File
from lookout.core.data_requests import DataService, with_changed_uasts_and_contents, \
    with_uasts_and_contents
from lookout.core.lib import extract_changed_nodes, files_by_language, filter_files, find_new_lines
from lookout.sdk.service_data_pb2 import Change
import numpy
import pandas
from sourced.ml.core.algorithms.token_parser import TokenParser

from lookout.style import __version__
from lookout.style.common import load_jinja2_template, merge_dicts
from lookout.style.format.utils import generate_comment
from lookout.style.typos.corrector_manager import TyposCorrectorManager
from lookout.style.typos.model import IdTyposModel
from lookout.style.typos.utils import Candidate, Columns, flatten_df_by_column, TEMPLATE_DIR

# TODO(zurk): Split TypoFix to FileFixes and TypoFix. content, path and identifiers_number should
# be in the FileFixes.
TypoFix = NamedTuple("TypoFix", (
    ("content", str),                                       # file content from head revision
    ("path", str),                                          # file path from head revision
    ("line_number", int),                                   # line number for the comment
    ("identifier", str),                                    # identifier where typo is found
    ("candidates", Iterable[Candidate]),                    # suggested identifiers
    ("identifiers_number", int),                            # number of unique analyzed identifiers
))

IDENTIFIER = bblfsh.role_id("IDENTIFIER")
IMPORT = bblfsh.role_id("IMPORT")
IDENTIFIER_INDEX_COLUMN = "identifier_index"


class IdTyposAnalyzer(Analyzer):
    """
    Identifier typos analyzer.
    """

    _log = logging.getLogger("IdTyposAnalyzer")
    model_type = IdTyposModel
    name = "lookout.style.typos"
    vendor = "source{d}"
    version = 1
    description = "Corrector of typos in source code identifiers."
    corrector_manager = TyposCorrectorManager()

    default_config = {
        "check_all_identifiers": False,
        "line_length_limit": 500,
        "min_token_length": 1,
        "processes_number": 1,
        "n_candidates": 3,
        "confidence_threshold": 0.1,
        "overall_size_limit": 5 << 20,  # 5 MB
        "corrector": "16577a2c-7f17-4a6f-a759-92f3a00cf339",
        "comment_template": os.path.join(TEMPLATE_DIR, "comment.md.jinja2"),
    }

    def __init__(self, model: IdTyposModel, url: str, config: Mapping[str, Any]):
        """
        Initialize a new instance of IdTyposAnalyzer.

        :param model: The instance of the model loaded from the repository or freshly trained.
        :param url: The analyzed project's Git remote.
        :param config: Configuration of the analyzer of unspecified structure.
        """
        super().__init__(model, url, config)
        self.config = self._load_config(config)
        self.corrector = self.corrector_manager.get(self.config["corrector"],
                                                    self.config["processes_number"])
        self.parser = self.create_token_parser()
        self.comment_template = load_jinja2_template(self.config["comment_template"])
        for identifier in model.identifiers:
            self.corrector.expand_vocabulary(set(self.parser.split(identifier)))
        self.allowed_identifiers = set() if self.config["check_all_identifiers"] else \
            model.identifiers

    @staticmethod
    def create_token_parser() -> TokenParser:
        """
        Create instance of TokenParser that should be used by IdTyposAnalyzer.

        :return: TokenParser.
        """
        return TokenParser(stem_threshold=1000, single_shot=True, min_split_length=1)

    @with_changed_uasts_and_contents(unicode=False)
    def analyze(self, ptr_from: ReferencePointer, ptr_to: ReferencePointer,
                data_service: DataService, changes: Iterable[Change],
                **data) -> List[Comment]:
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
        comments = []
        for typo_fix in self.generate_typos_fixes(list(changes)):
            text = self.render_comment_text(typo_fix)
            confidence = 1 - numpy.prod(
                1 - numpy.fromiter((x.confidence for x in typo_fix.candidates),
                                   dtype=numpy.float),
            )
            comment = generate_comment(text=text, confidence=int(100 * confidence),
                                       filename=typo_fix.path,
                                       line=typo_fix.line_number)
            comments.append(comment)
        return comments

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
                else:
                    lines = self._find_new_lines(prev_file.content, file.content)
                identifiers = self._get_identifiers(file.uast, lines)
                new_identifiers = [node for node in identifiers
                                   if node.token not in self.allowed_identifiers]
                if not new_identifiers:
                    continue
                self._log.debug("found %d new identifiers" % len(new_identifiers))
                suggestions = self.check_identifiers([n.token for n in new_identifiers])
                if not suggestions:
                    continue
                for index in suggestions.keys():
                    identifier = new_identifiers[index].token
                    candidates = {token: [Candidate(*sugg) for sugg in suggestions[index][token]]
                                  for token in suggestions[index]}
                    sugg_identifiers, id_confidences = [], []
                    for final_sugg, conf in self.generate_identifier_suggestions(
                            candidates, identifier):
                        sugg_identifiers.append(final_sugg)
                        id_confidences.append(conf)

                    identifier_candidates = [
                        Candidate(i, c) for i, c in zip(
                            sugg_identifiers, self._normalize_confidences(id_confidences),
                        ) if i != identifier
                    ]
                    if identifier_candidates:
                        yield TypoFix(
                            content=file.content.decode("utf-8", "replace"),
                            path=file.path,
                            identifier=identifier,
                            line_number=new_identifiers[index].start_position.line,
                            candidates=identifier_candidates,
                            identifiers_number=len(set(n.token for n in new_identifiers)),
                        )

    @staticmethod
    def _get_identifiers(uast, lines):
        return [node for node in extract_changed_nodes(uast, lines)
                if (IDENTIFIER in node.roles and IMPORT not in node.roles and node.token)]

    def _find_new_lines(self, prev_content: str, content: str) -> List[int]:
        return find_new_lines(prev_content, content)

    def render_comment_text(self, typo_fix: TypoFix) -> str:
        """
        Generate the text of the comment for the specified typo fix.

        :param typo_fix: Information about typo fix required to render a comment text.
        :return: string with the generated comment.
        """
        file_lines = typo_fix.content.splitlines()
        confidences = [candidate.confidence for candidate in typo_fix.candidates]
        return self.comment_template.render(
            identifier=typo_fix.identifier,
            suggestions=[candidate.token for candidate in typo_fix.candidates],
            old_code_line=file_lines[typo_fix.line_number - 1],
            confidences=confidences,
            zip=zip)

    @staticmethod
    def _normalize_confidences(confidences: Sequence[float]):
        s = sum(confidences)
        return [conf / s for conf in confidences]

    def generate_identifier_suggestions(self, suggestions: Mapping[str, Iterable[Candidate]],
                                        identifier: str) \
            -> Iterator[Tuple[str, float]]:
        """
        Generate suggestions for the identifier and compute the probability of suggestion.

        :param suggestions: suggestions are a mapping from a token to the list of candidates.
        :param identifier: initial identifier.
        :return: a generator of tuples with a suggestion for the identifier and probability.
        """
        if not suggestions:
            # do nothing in case of empty suggestions
            return
        token_candidates = []
        # split identifier
        initial_tokens = self.parser.split(identifier)

        # prepare suggestions per initial token
        for token in initial_tokens:
            # if no suggestion is provided, the token itself should be taken
            token_candidates.append(suggestions.get(token, [Candidate(token, 1.0)]))

        # sort candidates by resulting identifier probability
        suggestion_candidates = list(reversed(sorted(itertools.product(*token_candidates),
                                                     key=self._proba)))

        for i in range(min(self.config["n_candidates"], len(suggestion_candidates))):
            final_candidate = self.reconstruct_identifier(
                tokenizer=self.parser,
                pred_tokens=[c.token for c in suggestion_candidates[i]],
                identifier=identifier,
            )
            if final_candidate == identifier:
                # don't suggest identifier itself and anything with lesser probability
                return
            yield final_candidate, self._proba(suggestion_candidates[i])

    @staticmethod
    def _proba(candidates: Iterable[Candidate]):
        return float(numpy.prod([c.confidence for c in candidates]))

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
        # check required parameters
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
    @with_uasts_and_contents(unicode=False)
    def train(cls, ptr: ReferencePointer, config: Mapping[str, Any], data_service: DataService,
              files: Iterator[File], **data) -> IdTyposModel:
        """
        Generate a new model on top of the specified source code.

        :param ptr: Git repository state pointer.
        :param config: Configuration of the training of unspecified structure.
        :param data_service: The channel to the data service in Lookout server to query for \
                             UASTs, file contents, etc.
        :param files: iterator of File records from the data service.
        :param data: Extra data passed into the method. Used by the decorators to simplify \
                     the data retrieval.
        :return: Instance of `AnalyzerModel` (`model_type`, to be precise).
        """
        _log = logging.getLogger(cls.__name__)
        train_config = cls._load_config(config)
        _log.info("train %s %s %s %s", __version__, ptr.url, ptr.commit,
                  pformat(train_config, width=4096, compact=True))
        model = IdTyposModel()
        for _, files in files_by_language(files).items():
            for file in filter_files(
                    files=files, line_length_limit=train_config["line_length_limit"],
                    overall_size_limit=train_config["overall_size_limit"], log=_log):
                model.identifiers.update({
                    node.token for node in cls._get_identifiers(file.uast, [])})
        return model

    def check_identifiers(self, identifiers: List[str],
                          ) -> Dict[int, Dict[str, List[Candidate]]]:
        """
        Check tokens from identifiers for typos.

        :param identifiers: List of identifiers to check.
        :return: Dictionary of corrections grouped by ids of corresponding identifier \
                 in 'identifiers' and typoed tokens which have correction suggestions.
        """
        identifiers_positions = defaultdict(list)
        for i, identifier in enumerate(identifiers):
            identifiers_positions[identifier].append(i)
        unique_identifiers = sorted(identifiers_positions.keys())
        df = pandas.DataFrame(columns=[IDENTIFIER_INDEX_COLUMN, Columns.Split])
        df[IDENTIFIER_INDEX_COLUMN] = range(len(unique_identifiers))
        df[Columns.Split] = [" ".join(self.parser.split(identifier))
                             for identifier in unique_identifiers]
        df = flatten_df_by_column(df, Columns.Split, Columns.Token, str.split)
        df = df[df[Columns.Token].str.len() >= self.config["min_token_length"]]
        suggestions = self.corrector.suggest(df, n_candidates=self.config["n_candidates"],
                                             return_all=False)
        suggestions = self.filter_suggestions(df, suggestions)
        grouped_suggestions = defaultdict(dict)
        for index, row in df.iterrows():
            if index in suggestions.keys():
                for pos in identifiers_positions[unique_identifiers[row[IDENTIFIER_INDEX_COLUMN]]]:
                    grouped_suggestions[pos][row[Columns.Token]] = \
                        suggestions[index]
        return grouped_suggestions

    def filter_suggestions(self, test_df: pandas.DataFrame,
                           suggestions: Dict[int, List[Candidate]],
                           ) -> Dict[int, List[Candidate]]:
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
                if candidate.confidence < self.config["confidence_threshold"]:
                    break
                filtered_candidates.append(candidate)
                if candidate.token == tokens[index]:
                    break
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
