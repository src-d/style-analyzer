"""Predicted class definitions."""
# TODO(zurk): refactor CLS related code into one class. Related issue:
# https://github.com/src-d/style-analyzer/issues/286

CLS_SPACE = "<space>"
CLS_TAB = "<tab>"
CLS_NEWLINE = "<newline>"
CLS_SPACE_INC = "<+space>"
CLS_SPACE_DEC = "<-space>"
CLS_TAB_INC = "<+tab>"
CLS_TAB_DEC = "<-tab>"
CLS_SINGLE_QUOTE = "'"
CLS_DOUBLE_QUOTE = '"'
CLS_NOOP = "<noop>"
CLASSES = (CLS_SPACE, CLS_TAB, CLS_NEWLINE, CLS_SPACE_INC, CLS_SPACE_DEC,
           CLS_TAB_INC, CLS_TAB_DEC, CLS_SINGLE_QUOTE, CLS_DOUBLE_QUOTE, CLS_NOOP)
CLASS_INDEX = {cls: i for i, cls in enumerate(CLASSES)}
EMPTY_CLS = frozenset([CLASS_INDEX[CLS_TAB_DEC], CLASS_INDEX[CLS_SPACE_DEC],
                       CLASS_INDEX[CLS_NOOP]])
QUOTES_INDEX = {CLASS_INDEX[x] for x in [CLS_SINGLE_QUOTE, CLS_DOUBLE_QUOTE]}
NEWLINE_INDEX = CLASS_INDEX[CLS_NEWLINE]
CLS_TO_STR = {
    CLS_DOUBLE_QUOTE: '"',
    CLS_NEWLINE: "\n",  # FIXME(zurk) Not usable for \r\n endings
    CLS_NOOP: "",
    CLS_SINGLE_QUOTE: "'",
    CLS_SPACE: " ",
    CLS_SPACE_DEC: "",
    CLS_SPACE_INC: " ",
    CLS_TAB: "\t",
    CLS_TAB_DEC: "",
    CLS_TAB_INC: "\t",
}
INDEX_CLS_TO_STR = tuple(CLS_TO_STR[c] for c in CLASSES)

_CLASS_REPRESENTATIONS_MAPPING = {
    CLS_DOUBLE_QUOTE: '"',
    CLS_NEWLINE: "⏎",
    CLS_NOOP: "∅",
    CLS_SINGLE_QUOTE: "'",
    CLS_SPACE: "␣",
    CLS_SPACE_DEC: "␣⁻",
    CLS_SPACE_INC: "␣⁺",
    CLS_TAB: "⇥",
    CLS_TAB_DEC: "⇥⁻",
    CLS_TAB_INC: "⇥⁺",
}
CLASS_REPRESENTATIONS = [_CLASS_REPRESENTATIONS_MAPPING[cls] for cls in CLASSES]
del _CLASS_REPRESENTATIONS_MAPPING
CLASS_PRINTABLES = CLASS_REPRESENTATIONS[:]
CLASS_PRINTABLES[CLASS_INDEX[CLS_NEWLINE]] += "\n"
