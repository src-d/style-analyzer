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


class CodeGenerationBaseError(Exception):
    """
    Base class for CodeGenerator exceptions.
    """


class CodeGenerationError(CodeGenerationBaseError):
    """
    CodeGenerator.generate() can raise this exception if it fails to generate new file.

    To avoid this you can use `skip_errors=True` while creating CodeGenerator instance.
    """


class InapplicableIndentation(CodeGenerationBaseError):
    """
    InapplicableIndentation raises if you try to decrease tab indentation and have space \
    indentation. And vice versa.
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
        if not generated_lines:
            # Usually it means that you replace several newlines with only one.
            return ""
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
            elif not self._can_set_label(vnode, last_indentation):
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
                    y_new_no_indentation, new_indentation = \
                        self.apply_new_indentation(vnode, last_indentation)
                    tokens.append(y_new_no_indentation)
                    tokens.append(new_indentation)
                    last_indentation = new_indentation
        return "".join(tokens)

    def _can_set_label(self, vnode: VirtualNode, last_indentation: str) -> bool:
        """
        Check if a new label is applicable to a VirtualNode.

        Return False or raises FileGenerationError if it is not applicable.
        True means nothing because applicability depends also on the context.
        """
        y_new = vnode.y
        y_old = getattr(vnode, "y_old", y_new)
        if (y_new is None or
                y_new == y_old or
                y_new[0] == CLASS_INDEX[CLS_NOOP] or
                y_old[0] == CLASS_INDEX[CLS_NOOP]):
            return True
        set_y_old = set(y_old)
        set_y_new = set(y_new)
        if len(set_y_old & QUOTES_INDEX) > 0 and len(set_y_new & QUOTES_INDEX) == 0:
            y_new_repr = "".join([CLASS_REPRESENTATIONS[y] for y in y_new])
            return self._handle_error("Quotes cannot be changed to non-quote tokens.\n"
                                      "vnode: %s, y_new: %s" % (repr(vnode), y_new_repr))
        if len(set_y_old & QUOTES_INDEX) == 0 and len(set_y_new & QUOTES_INDEX) > 0:
            y_new_repr = "".join([CLASS_REPRESENTATIONS[y] for y in y_new])
            return self._handle_error("Non-Quote tokens cannot be changed to quote tokens.\n"
                                      "vnode: %s, y_new: %s" % (repr(vnode), y_new_repr))
        if len(set_y_new & self.INDENTATIONS_DEC):
            try:
                self.apply_new_indentation(vnode, last_indentation)
            except InapplicableIndentation as e:
                return self._handle_error(e.args[0])
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

    @staticmethod
    def apply_new_indentation(vnode: VirtualNode, last_indentation: str) -> Tuple[str, str]:
        """
        Apply new indentation token `vnode.y` to the `vnode.value`.

        Old value for `vnode.y` must be saved in `vnode.y_old`. If it is not possible to apply new
        indentation InapplicableIndentation() is raised.

        :param vnode: VirtualNode with new indentation label `y` and old label `y_old`.
        :param last_indentation: Last indentation in code. Usually indentation from the previous \
                                 line. If you do not expect additional line breaks in comparision \
                                 with `vnode.y_old` you can set to empty line.
        :return: string with values before indentation and string with a new indentation.
        """
        y_new = vnode.y
        y_old = getattr(vnode, "y_old", y_new)
        if y_new == y_old:
            return vnode.value
        indentation = (last_indentation if CodeGenerator.NEWLINE_INDEX not in set(y_old) else
                       CodeGenerator.revert_indentation_change(vnode).splitlines()[-1])

        # assume that all indentation changes are at the end of a vnode label
        no_indentation, indentation_change = \
            CodeGenerator._split_by_set(y_new, CodeGenerator.INDENTATIONS)
        for yi in indentation_change:
            if yi in CodeGenerator.INDENTATIONS_DEC:
                if len(indentation) == 0:
                    CodeGenerator.log.warning(
                        "There is no indentation characters left to decrease for "
                        "vnode %s and y_old %s", repr(vnode), str(y_old))
                elif indentation[-1] != CLS_TO_STR[CodeGenerator.DEC_TO_INC[CLASSES[yi]]]:
                    raise InapplicableIndentation(
                        "Indentation change is not applicable for %s" % repr(vnode))
                else:
                    indentation = indentation[:-1]
            elif yi in CodeGenerator.INDENTATIONS_INC:
                indentation += CLS_TO_STR[CLASSES[yi]]
            else:
                raise ValueError("Unexpected character in y_new: %s. %s" % (
                    CLASS_REPRESENTATIONS[yi], repr(vnode)))
        return "".join(CLS_TO_STR[CLASSES[yi]] for yi in no_indentation), indentation

    @staticmethod
    def revert_indentation_change(vnode: VirtualNode) -> str:
        """
        Reverts original change for provided VirtualNode.

        Example:
        for vnode
        ```
        VirtualNode(value='⏎␣␣␣␣', y=⏎␣⁺␣⁺, start=(852, 26, 20), end=(857, 27, 5), node=None,
                    path="lib/find-chrome.js")
        ```
        revert_indentation_change() removes two spaces that were added. The result is `⏎␣␣`.

        :param vnode: VirtualNode with indentation label `y`.
        :return: new value for VirtualNode.value.
        """
        y = getattr(vnode, "y_old", vnode.y)
        value = vnode.value
        for y_i in y[::-1]:
            if y_i in CodeGenerator.INDENTATIONS_DEC:
                value += CLS_TO_STR[CodeGenerator.DEC_TO_INC[CLASSES[y_i]]]
            elif y_i in CodeGenerator.INDENTATIONS_INC:
                if value[-1] != CLS_TO_STR[CLASSES[y_i]]:
                    raise InapplicableIndentation("%s has inconsistent value and y" % repr(vnode))
                value = value[:-1]
            elif y_i == CodeGenerator.NEWLINE_INDEX:
                break
            else:
                raise ValueError("%s has unexpected character in y: %s" % (
                    repr(vnode), CLASS_REPRESENTATIONS[y_i]))
        return value

    @staticmethod
    def _split_by_set(to_split: tuple, s: set) -> Tuple[tuple, tuple]:
        index = next((i for i, ch in enumerate(to_split) if ch in s), len(to_split))
        return to_split[:index], to_split[index:]
