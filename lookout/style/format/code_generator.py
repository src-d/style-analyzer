"""Code generator that able to generate new source code from format model style suggestions."""
import logging
from typing import Sequence, Union

import numpy

from lookout.style.format.feature_extractor import FeatureExtractor
from lookout.style.format.feature_utils import (
    CLASS_INDEX, CLASS_REPRESENTATIONS, CLASSES, CLS_DOUBLE_QUOTE, CLS_NEWLINE, CLS_NOOP,
    CLS_SINGLE_QUOTE, CLS_SPACE_DEC, CLS_SPACE_INC, CLS_TAB_DEC, CLS_TAB_INC, CLS_TO_STR,
    VirtualNode
)


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
    NEWLINE_INDX = CLASS_INDEX[CLS_NEWLINE]

    def __init__(self,
                 feature_extractor: FeatureExtractor,
                 change_locally: bool = False,
                 skip_errors: bool = False,
                 verbose: bool = False,
                 url: str = "<NA>",
                 commit: str = "<NA>"):
        """
        Construct a CodeGenerator.

        :param feature_extractor: Feature extraction class that was used to generate \
                                  corresponding data.
        :param change_locally: Apply style analyser changes only locally. For example, \
                               indentation changes do not propagate to the next lines.
        :param skip_errors: Raise an exception in case of code generation failure if skip_errors \
                            is False. If skip_errors is True, ignore such model suggestions.
        :param verbose: Run code with additional verbose debug logging.
        :param url: Repository url if applicable. Useful for more informative warning messages.
        :param commit: Commit hash if applicable. Useful for more informative warning messages.

        """
        self.feature_extractor = feature_extractor
        self.change_locally = change_locally
        self.skip_errors = skip_errors
        self.verbose = verbose
        self.url = url
        self.commit = commit

    def generate(self,
                 vnodes_y: Sequence[VirtualNode],
                 y_pred: numpy.ndarray,
                 vnodes: Sequence[VirtualNode],
                 ) -> str:
        """
        Generate new source code from format model suggestions.

        :param vnodes_y: Sequence of the labeled `VirtualNode`-s corresponding to labeled samples.\
                         Should be ordered by start position value.
        :param y_pred: The model predictions for `vnodes_y` `VirtualNode`-s.
        :param vnodes: Sequence of all the `VirtualNode`-s corresponding to the input code file. \
                       Should be ordered by start position value.
        :return: New source code file content.
        """
        tokens = [""]
        state = _State(change_locally=self.change_locally)
        for vnode, y_new in self._iterate_vnodes(vnodes, vnodes_y, y_pred):
            # if there is no prediction (y_new) for current VirtualNode (vnode)
            # its value is set to None.
            y_change = y_new != vnode.y
            if not y_change and not vnode.value:
                # Skipping all unmodified NOOPS and empty tokens
                # No need to finalize
                continue
            elif not self.check(vnode, y_new):
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
            elif not (y_change or vnode.is_accumulated_indentation or state.line_beginning):
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
                assert not (vnode.y is None or y_new is None)
                if y_change and (self.INDENTATIONS & set(vnode.y) or
                                 self.INDENTATIONS & set(y_new)):
                    state.handle_indentation_changes(vnode, y_new)
                    if self.NEWLINE_INDX in y_new:
                        tokens.append("\n" + " " * (vnode.y.count(CLASS_INDEX[CLS_SPACE_INC]) -
                                                    y_new.count(CLASS_INDEX[CLS_SPACE_INC])))
                        state.update(vnode, y_new)
                        continue

                # If we are here we need just to replace old value with a new one.
                assert y_new is not None
                tokens.append("".join(CLS_TO_STR[CLASSES[yi]] for yi in y_new))
            state.update(vnode, y_new)
            continue

        return "".join(tokens)

    def check(self, vnode: VirtualNode, y_new: Union[int, Sequence[int]]):
        """
        Check if y_new is applicable to vnode.

        Return False or raises FileGenerationError if it is not applicable.
        True means nothing because applicability depends on context also.
        """
        if (y_new is None or
                y_new == vnode.y or
                y_new[0] == CLASS_INDEX[CLS_NOOP] or
                vnode.y[0] == CLASS_INDEX[CLS_NOOP]):
            return True
        if len(set(vnode.y) & self.QUOTES) > 0 and len(set(y_new) & self.QUOTES) == 0:
            y_new_repr = "".join([CLASS_REPRESENTATIONS[y] for y in y_new])
            return self._handle_error("Quotes cannot be changed to non-quote tokens.\n"
                                      "vnode: %s, y_new: %s" % (repr(vnode), y_new_repr))
        if len(set(vnode.y) & self.QUOTES) == 0 and len(set(y_new) & self.QUOTES) > 0:
            y_new_repr = "".join([CLASS_REPRESENTATIONS[y] for y in y_new])
            return self._handle_error("Non-Quote tokens cannot be changed to quote tokens.\n"
                                      "vnode: %s, y_new: %s" % (repr(vnode), y_new_repr))
        return True

    def _iterate_vnodes(self, vnodes, vnodes_y, y_pred):
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
                yield vnode, vnode.y
            else:
                yield vnode, self.feature_extractor.labels_to_class_sequences[y_pred[y_index]]
                y_index += 1

    def _handle_error(self, msg):
        self.log.warning("%s url: %s, commit: %s. %s.",
                         msg, self.url, self.commit, "Skipping" if self.skip_errors else
                         "Set skip_errors=True to skip faulty suggestions")
        if not self.skip_errors:
            raise CodeGenerationError(msg)
        return False


class _State:
    def __init__(self, change_locally):
        self.change_locally = change_locally
        self.indent_delta = 0
        self.indent_increase_tokens = []
        self.line_beginning = True
        self.line_removed = False
        self.line_added = False
        self.accumulated_indentation = ""

    def update(self, vnode, y_new):
        y = vnode.y
        self.line_beginning = (y_new is not None and
                               CodeGenerator.NEWLINE_INDX in y_new)
        self.line_added = (y_new is not None and
                           CodeGenerator.NEWLINE_INDX in y_new and
                           CodeGenerator.NEWLINE_INDX not in y)
        self.line_removed = (y_new is not None and
                             CodeGenerator.NEWLINE_INDX not in y_new and
                             CodeGenerator.NEWLINE_INDX in y)
        if self.change_locally and not self.line_beginning:
            self.indent_delta = 0

    def reset_indentation(self):
        self.indent_delta = 0
        self.accumulated_indentation = ""

    def handle_indentation_changes(self, vnode, y_new):
        n_y_space_inc = vnode.y.count(CLASS_INDEX[CLS_SPACE_INC])
        n_y_space_dec = vnode.y.count(CLASS_INDEX[CLS_SPACE_DEC])
        n_y_new_space_inc = y_new.count(CLASS_INDEX[CLS_SPACE_INC])
        n_y_new_space_dec = y_new.count(CLASS_INDEX[CLS_SPACE_DEC])
        indent_delta_space_change = n_y_new_space_inc - n_y_space_inc
        indent_delta_space_change -= n_y_new_space_dec - n_y_space_dec

        n_y_tab_inc = vnode.y.count(CLASS_INDEX[CLS_TAB_INC])
        n_y_tab_dec = vnode.y.count(CLASS_INDEX[CLS_TAB_DEC])
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
