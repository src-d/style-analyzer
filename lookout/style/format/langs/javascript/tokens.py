import re

RESERVED = [
    "abstract",
    "any",
    "as",
    "async",
    "await",
    "boolean",
    "break",
    "byte",
    "case",
    "catch",
    "char",
    "class",
    "const",
    "continue",
    "debugger",
    "declare",
    "default",
    "delete",
    "do",
    "double",
    "else",
    "enum",
    "export",
    "exports",
    "extends",
    "false",
    "final",
    "finally",
    "float",
    "for",
    "from",
    "function",
    "get",
    "goto",
    "of",
    "opaque",
    "if",
    "implements",
    "import",
    "in",
    "instanceof",
    "int",
    "interface",
    "let",
    "long",
    "mixed",
    "module",
    "native",
    "new",
    "number",
    "null",
    "package",
    "private",
    "protected",
    "public",
    "return",
    "set",
    "short",
    "static",
    "string",
    "super",
    "switch",
    "synchronized",
    "this",
    "throw",
    "throws",
    "transient",
    "true",
    "try",
    "type",
    "typeof",
    "yield",
    "var",
    "void",
    "volatile",
    "while",
    "with",
    "+",
    "-",
    "*",
    "/",
    "%",
    "++",
    "--",
    "=",
    "+=",
    "-=",
    "/=",
    "%=",
    "==",
    "===",
    "!=",
    "!==",
    ">",
    "<",
    ">=",
    "<=",
    "?",
    ":",
    "&&",
    "||",
    "!",
    "&",
    "|",
    "~",
    "^",
    ">>",
    "<<",
    "(",
    ")",
    "{",
    "}",
    ".",
    "...",
    "[",
    "]",
    ">>>",
    ",",
    ";",
    "'",
    '"',
    "`",
    "${",
    "\\",
]

# The longest keywords should come first for the regex below to be usable with finditer
RESERVED.sort(reverse=True)

RESERVED_INDEX = {r: i for i, r in enumerate(RESERVED)}

PARSER = re.compile("|".join(re.escape(i) for i in RESERVED) + r"|\s+")
