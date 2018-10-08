"""Facilities to filter files during debugging."""
from os.path import isfile
import re
from typing import Iterable, Optional

from lookout.core.garbage_exclusion import GARBAGE_PATTERN


def filter_filepaths(filepaths: Iterable[str], line_length_limit: int = 500,
                     exclude_pattern: Optional[str] = None) -> Iterable[str]:
    """
    Mirror of the file filtering used in the format analyzer for use by debugging tools.

    :param filepaths: Iterable of paths to filter.
    :param line_length_limit: Maximum length of lines to keep a file.
    :param exclude_pattern: Pattern to reject files based on their path. If None, uses the pattern
                            currently in use in lookout.core. Use "" to not filter anything.
    :return Iterable of paths, filtered.
    """
    if exclude_pattern is None:
        exclude_pattern = GARBAGE_PATTERN
    exclude_compiled_pattern = re.compile(exclude_pattern) if exclude_pattern else None
    for filepath in filepaths:
        if not isfile(filepath):
            continue
        if exclude_compiled_pattern and exclude_compiled_pattern.search(filepath):
            continue
        with open(filepath, "rb") as fh:
            if len(max(fh, key=len, default=b"")) <= line_length_limit:
                yield filepath
