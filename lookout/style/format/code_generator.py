"""Code generator that able to generate new source code from format model style suggestions."""
import logging
from typing import Iterable, List, Optional, Sequence, Tuple

from numpy import ndarray

from lookout.style.format.classes import (
    CLASS_INDEX, CLASS_REPRESENTATIONS, CLASSES, CLS_NEWLINE, CLS_NOOP, CLS_SPACE_DEC,
    CLS_SPACE_INC, CLS_TAB_DEC, CLS_TAB_INC, CLS_TO_STR, QUOTES_INDEX)
from lookout.style.format.feature_extractor import FeatureExtractor
from lookout.style.format.rules import Rule, Rules
from lookout.style.format.virtual_node import VirtualNode


class CodeGenerationError(Exception):
    """
    CodeGenerator.generate() can raise this exception if it fails to generate new file.

    To avoid this you can use `skip_errors=True` while creating CodeGenerator instance.
    """


class CodeGenerator:
    """
    Generate new source code from format model suggestions.

    Use `generate()` function to get a result.
    """

    log = logging.getLogger("FormatAnalyzer")
    INDENTATIONS_DEC = {CLASS_INDEX[x] for x in [CLS_SPACE_DEC, CLS_TAB_DEC]}
    INDENTATIONS_INC = {CLASS_INDEX[x] for x in [CLS_SPACE_INC, CLS_TAB_INC]}
    DEC_TO_INC = {
        CLS_SPACE_DEC: CLS_SPACE_INC,
        CLS_TAB_DEC: CLS_TAB_INC,
    }
    INDENTATIONS = INDENTATIONS_DEC | INDENTATIONS_INC
    NEWLINE_RELATED = {CLASS_INDEX[x] for x in
                       [CLS_SPACE_INC, CLS_TAB_INC, CLS_SPACE_DEC, CLS_TAB_DEC, CLS_NEWLINE]}
    NEWLINE_INDEX = CLASS_INDEX[CLS_NEWLINE]

    def __init__(self,
                 feature_extractor: FeatureExtractor,
                 skip_errors: bool = False,
                 verbose: bool = False,
                 url: str = "<N/A>",
                 commit: str = "<N/A>"):
        """
        Construct a CodeGenerator.

        :param feature_extractor: Feature extraction class that was used to generate \
                                  corresponding data.
        :param skip_errors: Raise an exception in case of code generation failure if skip_errors \
                            is False. If skip_errors is True, ignore such model suggestions.
        :param verbose: Run code with additional verbose debug logging.
        :param url: Repository url if applicable. Useful for more informative warning messages.
        :param commit: Commit hash if applicable. Useful for more informative warning messages.

        """
        self.feature_extractor = feature_extractor
        self.skip_errors = skip_errors
        self.verbose = verbose
        self.url = url
        self.commit = commit

    def generate_new_line(self, line_vnodes: List[VirtualNode]) -> str:
        """
        Generate new code line for giving vnodes.

        :param line_vnodes: corresponding virtual nodes sequence. It should be started from token \
                            with indentation change or newline token from the prevoius line if \
                            exists. The final line break of the line should not be included.
        :return: Code line.
        """
        if not line_vnodes:
            return ""

        first_y = line_vnodes[0].y

        generated = self.generate(line_vnodes)
        if line_vnodes[0].y is None:
            # we add " " because we want to count probably empty last line
            lines_no = len((line_vnodes[0].value + " ").splitlines())
            return "".join(generated.splitlines()[lines_no:])

        if first_y is not None and self.NEWLINE_INDEX not in first_y:
            return generated

        # First line is always removed because it is an end line from the previous line.
        generated_lines = generated.splitlines(keepends=True)[1:]
        for i, line in enumerate(generated_lines):  # noqa B007
            if line.splitlines()[0]:
                # Line is not empty
                break
        lines_num = 0
        if line_vnodes and hasattr(line_vnodes[0], "y_old") and self.NEWLINE_INDEX in first_y:
            lines_num = line_vnodes[0].y.count(self.NEWLINE_INDEX) - \
                        line_vnodes[0].y_old.count(self.NEWLINE_INDEX)
        return "".join(generated_lines[i - max(0, lines_num):])

    def apply_predicted_y(self, vnodes: Sequence[VirtualNode],
                          vnodes_y: Sequence[VirtualNode],
                          rule_winners: Sequence[int],
                          rules: Rules,
                          ) -> List[VirtualNode]:
        """
        Update labels for the sequence of VirtualNode-s. We also discard NOOPs.

        :param vnodes: Sequence of all the `VirtualNode`-s corresponding to the input code file. \
                       Should be ordered by position.
        :param vnodes_y: Sequence of the labeled `VirtualNode`-s corresponding to labeled samples.\
                         Should be ordered by start position value.
        :param rule_winners: Sequence of applied rules to generate y_pred.
        :param rules: Rules that were used for prediction.
        :return: The list of VirtualNode-s with adjusted labels. "y_old" attribute of each node \
                 contains the previous label.
        """
        result = []
        for vnode, y_new, applied_rule in \
                self._iterate_vnodes(vnodes, vnodes_y, rule_winners, rules):
            vnode = vnode.copy()
            if y_new != vnode.y:
                vnode.y, vnode.y_old = y_new, vnode.y
                if applied_rule:
                    vnode.applied_rule = applied_rule
            result.append(vnode)
        return result

    def generate(self, vnodes: Sequence[VirtualNode]) -> str:
        """
        Generate new source code from format model suggestions.

        :param vnodes: Sequence of all the `VirtualNode`-s corresponding to the input code file. \
                       Should be ordered by position.
        :return: New source code file content.
        """
        def split_by_set(to_split: tuple, s: set) -> Tuple[tuple, tuple]:
            index = next((i for i, ch in enumerate(to_split) if ch in s), len(to_split))
            return to_split[:index], to_split[index:]

        tokens = [""]
        last_indentation = ""
        for vnode in vnodes:
            y_new = vnode.y
            y_old = getattr(vnode, "y_old", y_new)
            if y_new == y_old:
                # Nothing should be changed
                tokens.append(vnode.value)
                if y_new and self.NEWLINE_INDEX in y_new:
                    last_indentation = vnode.value.splitlines()[-1]
            elif not self._can_set_label(y_new, y_old, repr(vnode)):
                # Check unexpected situations.
                # If skip_errors is False `self.check()` raises an exception.
                tokens.append(vnode.value)
                if y_new and self.NEWLINE_INDEX in y_new:
                    last_indentation = vnode.value.splitlines()[-1]
            else:
                # Something was changed
                assert not (y_old is None and y_new is None)
                if self.NEWLINE_INDEX not in set(y_new):
                    tokens.append("".join(CLS_TO_STR[CLASSES[yi]] for yi in y_new))
                else:
                    # assume that all indentation changes are at the end of a vnode label
                    y_new_no_indentation, new_indentation_change = \
                        split_by_set(y_new, self.INDENTATIONS)
                    _, old_indentation_change = split_by_set(y_old, self.INDENTATIONS)
                    tokens.append("".join(CLS_TO_STR[CLASSES[yi]] for yi in y_new_no_indentation))
                    if self.NEWLINE_INDEX not in set(y_old):
                        cur_indentation = last_indentation
                    else:
                        cur_indentation = vnode.value.splitlines()[-1]
                        # Revert indentation changes
                        for y_old_i in old_indentation_change:
                            if y_old_i in self.INDENTATIONS_DEC:
                                cur_indentation += CLS_TO_STR[self.DEC_TO_INC[CLASSES[y_old_i]]]
                            elif y_old_i in self.INDENTATIONS_INC:
                                assert cur_indentation[-1] == CLS_TO_STR[CLASSES[y_old_i]]
                                cur_indentation = cur_indentation[:-1]
                            else:
                                raise ValueError("Unexpected character in y_old: %s" %
                                                 CLASS_REPRESENTATIONS[y_old_i])
                    for y_new_i in new_indentation_change:
                        if y_new_i in self.INDENTATIONS_DEC:
                            if len(cur_indentation) == 0:
                                self.log.warning(
                                    "There is no indentations chanarters left to decrease for "
                                    "vnode %s and y_old %s", repr(vnode), str(y_old))
                                continue
                            assert cur_indentation[-1] == \
                                CLS_TO_STR[self.DEC_TO_INC[CLASSES[y_new_i]]], \
                                "Unexpected character in cur_indentation: `%s` != `%s`" % (
                                    cur_indentation[-1], CLS_TO_STR[CLASSES[y_new_i]])
                            cur_indentation = cur_indentation[:-1]
                        elif y_new_i in self.INDENTATIONS_INC:
                            cur_indentation += CLS_TO_STR[CLASSES[y_new_i]]
                        else:
                            raise ValueError("Unexpected character in y_new: %s" %
                                             CLASS_REPRESENTATIONS[y_new_i])
                    last_indentation = cur_indentation
                    tokens.append(cur_indentation)

        return "".join(tokens)

    def _can_set_label(self, y_new: Optional[Sequence[int]], y_old: Sequence[int],
                       node_repr: str = "") -> bool:
        """
        Check if a new label is applicable to a VirtualNode.

        Return False or raises FileGenerationError if it is not applicable.
        True means nothing because applicability depends also on the context.
        """
        if (y_new is None or
                y_new == y_old or
                y_new[0] == CLASS_INDEX[CLS_NOOP] or
                y_old[0] == CLASS_INDEX[CLS_NOOP]):
            return True
        if len(set(y_old) & QUOTES_INDEX) > 0 and len(set(y_new) & QUOTES_INDEX) == 0:
            y_new_repr = "".join([CLASS_REPRESENTATIONS[y] for y in y_new])
            return self._handle_error("Quotes cannot be changed to non-quote tokens.\n"
                                      "vnode: %s, y_new: %s" % (node_repr, y_new_repr))
        if len(set(y_old) & QUOTES_INDEX) == 0 and len(set(y_new) & QUOTES_INDEX) > 0:
            y_new_repr = "".join([CLASS_REPRESENTATIONS[y] for y in y_new])
            return self._handle_error("Non-Quote tokens cannot be changed to quote tokens.\n"
                                      "vnode: %s, y_new: %s" % (node_repr, y_new_repr))
        return True

    def _iterate_vnodes(self, vnodes: Sequence[VirtualNode], vnodes_y: Sequence[VirtualNode],
                        rule_winners: ndarray, rules: Rules,
                        ) -> Iterable[Tuple[VirtualNode, Optional[Tuple[int, ...]],
                                            Optional[Rule]]]:
        y_index = 0
        j = 0
        for i, vnode in enumerate(vnodes):
            if self.verbose:
                if vnode.y is None:
                    self.log.debug("Node #%d: %s", i, repr(vnode))
                else:
                    self.log.debug("Node #%d, y #%d: %s", i, j, repr(vnode))
                    j += 1
            if y_index >= len(vnodes_y) or id(vnode) != id(vnodes_y[y_index]):
                yield vnode, vnode.y, None
            else:
                rule_winner = rule_winners[y_index]
                if rule_winner < 0:
                    yield vnode, vnode.y, None
                else:
                    rule = rules.rules[rule_winner]
                    yield (vnode, self.feature_extractor.labels_to_class_sequences[rule.stats.cls],
                           rule)
                y_index += 1

    def _handle_error(self, msg: str) -> bool:
        self.log.debug("%s url: %s, commit: %s. %s.",
                       msg, self.url, self.commit, "Skipping" if self.skip_errors else
                       "Set skip_errors=True to skip faulty suggestions")
        if not self.skip_errors:
            raise CodeGenerationError(msg)
        return False
