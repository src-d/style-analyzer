"""Javascript specific methods to unwrap a token from its delimiters."""
from typing import Tuple


def unwrap_string_literal(outer_token: str) -> Tuple[str, str]:
    """
    Unwrap a string literal.

    :param outer_token: Complete token.
    :return: Inner token (without delimiters), complete token.
    """
    assert len(outer_token) >= 2, "string literal should have at least 2 characters"
    return outer_token[1:-1], outer_token


def unwrap_use_outer_token(outer_token: str) -> Tuple[str, str]:
    """
    Unwrap a token that has no delimiters (noop).

    :param outer_token: Complete token.
    :return: Inner token (without delimiters), complete token.
    """
    return outer_token, outer_token


TOKEN_UNWRAPPERS = {
    "StringLiteral": unwrap_string_literal,
    "NumericLiteral": unwrap_use_outer_token,
}
