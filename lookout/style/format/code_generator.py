"""Code generator that able to generate new source code from format model style suggestions."""
import logging
from typing import Iterable, List, Optional, Sequence, Tuple

from numpy import ndarray

from lookout.style.format.classes import (
    CLASS_INDEX, CLASS_REPRESENTATIONS, CLASSES, CLS_DOUBLE_QUOTE, CLS_NEWLINE, CLS_NOOP,
    CLS_SINGLE_QUOTE, CLS_SPACE_DEC, CLS_SPACE_INC, CLS_TAB_DEC, CLS_TAB_INC, CLS_TO_STR)
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
    QUOTES = {CLASS_INDEX[x] for x in [CLS_SINGLE_QUOTE, CLS_DOUBLE_QUOTE]}
    INDENTATIONS = {CLASS_INDEX[x] for x in
                    [CLS_SPACE_INC, CLS_TAB_INC, CLS_SPACE_DEC, CLS_TAB_DEC]}
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

    def generate(self, vnodes: Sequence[VirtualNode], indentation: str) -> str:
        """
        Generate new source code from format model suggestions.

        :param vnodes: Sequence of all the `VirtualNode`-s corresponding to the input code file. \
                       Should be ordered by position.
        :param indentation: Can be either "local" or "global"; in "local" mode \
                            indentation changes do not propagate to the following lines.
        :return: New source code file content.
        """
        assert indentation in ("local", "global")
        tokens = [""]
        state = _State(change_locally=indentation == "local")
        for vnode in vnodes:
            y_new = vnode.y
            y_old = getattr(vnode, "y_old", y_new)
            y_changed = y_new != y_old
            if not self._can_set_label(y_new, y_old, repr(vnode)):
                # Check unexpected situations.
                # If skip_errors is False `self.check()` raises an exception.
                tokens.append(vnode.value)
            elif state.line_removed and vnode.is_accumulated_indentation:
                # Skip accumulated indentation for the line that was removed
                pass
            elif state.line_added:
                assert y_new is None
                # Add last accumulated indentation with respect to indent_delta value
                if state.indent_delta < 0:
                    tokens.append(state.accumulated_indentation[:state.indent_delta])
                else:
                    tokens.append(state.accumulated_indentation)
                    tokens.extend(state.indent_increase_tokens)
                tokens.append(vnode.value)
            elif not (y_changed or vnode.is_accumulated_indentation or state.line_beginning):
                # Nothing should be changed
                tokens.append(vnode.value)
            elif state.line_beginning and y_new is None:
                # line beginning should be handled super carefully
                if state.indent_delta == 0:
                    # Nothing should be changed, just save accumulated_indentation
                    if vnode.is_accumulated_indentation:
                        state.accumulated_indentation = vnode.value
                    tokens.append(vnode.value)
                else:
                    if vnode.is_accumulated_indentation:
                        # We should modify existing indentation
                        if state.indent_delta < 0:
                            if len(vnode.value) < -state.indent_delta:
                                self._handle_error("Indentation decrease excess.")
                                state.reset_indentation()
                            else:
                                state.accumulated_indentation = vnode.value[:state.indent_delta]
                                tokens.append(state.accumulated_indentation)
                        else:
                            state.accumulated_indentation = \
                                vnode.value + "".join(state.indent_increase_tokens)
                            tokens.append(state.accumulated_indentation)
                    else:
                        state.accumulated_indentation = ""
                        if state.indent_delta > 0:
                            # We should insert new indentation.
                            # It happens if you increase indentation
                            # for a line with zero indentation
                            state.accumulated_indentation = "".join(state.indent_increase_tokens)
                        # For indent_delta < 0
                        # we already decrease an indentation on the previous step
                        tokens.append(state.accumulated_indentation)
                        tokens.append(vnode.value)
            else:
                # Indentation changes handling
                assert not (y_old is None or y_new is None)
                if y_changed and (self.INDENTATIONS & set(y_old) or
                                  self.INDENTATIONS & set(y_new)):
                    state.handle_indentation_changes(y_old, y_new)
                    if self.NEWLINE_INDEX in y_new:
                        tokens.append("\n" + " " * (y_old.count(CLASS_INDEX[CLS_SPACE_INC]) -
                                                    y_new.count(CLASS_INDEX[CLS_SPACE_INC])))
                        state.update(y_old, y_new)
                        continue

                # If we are here we need just to replace old value with a new one.
                assert y_new is not None
                tokens.append("".join(CLS_TO_STR[CLASSES[yi]] for yi in y_new))
            state.update(y_old, y_new)

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
        if len(set(y_old) & self.QUOTES) > 0 and len(set(y_new) & self.QUOTES) == 0:
            y_new_repr = "".join([CLASS_REPRESENTATIONS[y] for y in y_new])
            return self._handle_error("Quotes cannot be changed to non-quote tokens.\n"
                                      "vnode: %s, y_new: %s" % (node_repr, y_new_repr))
        if len(set(y_old) & self.QUOTES) == 0 and len(set(y_new) & self.QUOTES) > 0:
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


class _State:
    def __init__(self, change_locally: bool) -> None:
        self.change_locally = change_locally
        self.indent_delta = 0
        self.indent_increase_tokens = []
        self.line_beginning = True
        self.line_removed = False
        self.line_added = False
        self.accumulated_indentation = ""

    def update(self, y_old: Sequence[int], y_new: Optional[Sequence[int]]) -> None:
        self.line_beginning = (y_new is not None and
                               CodeGenerator.NEWLINE_INDEX in y_new)
        self.line_added = (y_new is not None and
                           CodeGenerator.NEWLINE_INDEX in y_new and
                           CodeGenerator.NEWLINE_INDEX not in y_old)
        self.line_removed = (y_new is not None and
                             CodeGenerator.NEWLINE_INDEX not in y_new and
                             CodeGenerator.NEWLINE_INDEX in y_old)
        if self.change_locally and not self.line_beginning:
            self.indent_delta = 0

    def reset_indentation(self) -> None:
        self.indent_delta = 0
        self.accumulated_indentation = ""

    def handle_indentation_changes(self, y_old: Sequence[int], y_new: Sequence[int]) -> None:
        n_y_space_inc = y_old.count(CLASS_INDEX[CLS_SPACE_INC])
        n_y_space_dec = y_old.count(CLASS_INDEX[CLS_SPACE_DEC])
        n_y_new_space_inc = y_new.count(CLASS_INDEX[CLS_SPACE_INC])
        n_y_new_space_dec = y_new.count(CLASS_INDEX[CLS_SPACE_DEC])
        indent_delta_space_change = n_y_new_space_inc - n_y_space_inc
        indent_delta_space_change -= n_y_new_space_dec - n_y_space_dec

        n_y_tab_inc = y_old.count(CLASS_INDEX[CLS_TAB_INC])
        n_y_tab_dec = y_old.count(CLASS_INDEX[CLS_TAB_DEC])
        n_y_new_tab_inc = y_new.count(CLASS_INDEX[CLS_TAB_INC])
        n_y_new_tab_dec = y_new.count(CLASS_INDEX[CLS_TAB_DEC])
        indent_delta_tab_change = n_y_new_tab_inc - n_y_tab_inc
        indent_delta_tab_change -= n_y_new_tab_dec - n_y_tab_dec

        indent_delta_change = indent_delta_space_change + indent_delta_tab_change

        self.indent_delta += indent_delta_change
        if indent_delta_change > 0:
            self.indent_increase_tokens += (
                [CLS_TO_STR[CLS_SPACE_INC]] * indent_delta_space_change +
                [CLS_TO_STR[CLS_TAB_INC]] * indent_delta_tab_change)
        else:
            self.indent_increase_tokens = self.indent_increase_tokens[:self.indent_delta]
