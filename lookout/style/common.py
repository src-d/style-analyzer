from copy import deepcopy
from datetime import datetime
import logging
import lzma
import os
import pprint
from typing import Callable, Iterator, Mapping, Sequence

import jinja2
from smart_open import register_compressor


def add_xz_to_smart_open():
    """Add .xz support to smart_open."""
    def _handle_xz(file_obj, mode):
        return lzma.LZMAFile(filename=file_obj, mode=mode, format=lzma.FORMAT_XZ)
    register_compressor(".xz", _handle_xz)


def merge_dicts(*dicts: Mapping) -> dict:
    """
    Merge several mappings together; nested values are merged recursively.

    Operation is not commutative, each next dictionary overrides values of the previous one for
    the same keys sequence.
    (see example). Example:
    >>> a = {1: 1, 2: {3: 3, 4: 4}}
    >>> b = {1: 10, 2: {3: 30, 4: 4, 5: 5}}
    >>> merge_dicts(a, b)
    >>> {1: 10, 2: {3: 30, 4: 4, 5: 5}}
    >>> merge_dicts(b, a)
    >>> {1: 1, 2: {3: 3, 4: 4, 5: 5}}

    :return: New merged dictionary.
    """
    if len(dicts) == 0:
        raise ValueError("At least one argument is required.")
    if len(dicts) == 1:
        return dict(deepcopy(dicts[0]))
    res = dict(deepcopy(dicts[0]))
    stack = [(res, d) for d in dicts[:0:-1]]
    while stack:
        d1, d2 = stack.pop()
        for key, value in d2.items():
            if isinstance(value, dict):
                stack.append((d1.setdefault(key, {}), value))
            else:
                d1[key] = value
    return res


def load_jinja2_template(path: str) -> jinja2.Template:
    """Return a loaded template by the specified file path."""
    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=True,
                             extensions=["jinja2.ext.do"])
    env.filters.update({
        "pformat": pprint.pformat,
        "deepcopy": deepcopy,
        "intersect": lambda x, y: set(x).intersection(set(y)),
    })
    env.globals.update({
        "zip": zip,
    })
    root, name = os.path.split(path)
    loader = jinja2.FileSystemLoader((root,), followlinks=True)
    template = loader.load(env, name)
    # the following is really needed, otherwise e.g. range is undefined
    template.globals = template.environment.globals
    return template


def huge_progress_bar(sequence: Sequence, log: logging.Logger, get_iter_name: Callable,
                      ) -> Iterator:
    """Create big multi-line progress bar with the logs."""
    progress_bar_template = ("\n%s\n"
                             "= %-76s =\n"
                             "= %2d / %2d%s=\n"
                             "= Now:  %-60s%s=\n"
                             "= Left: %-40s%s=\n"
                             "= Ends: %-60s%s=\n"
                             "%s")
    start_time = datetime.now()
    index = -1
    for index, item in enumerate(sequence):
        now = datetime.now()
        if index > 0:
            left = (len(sequence) - index) / index * (now - start_time)
        else:
            left = None
        log.info(progress_bar_template,
                 "=" * 80,
                 get_iter_name(item),
                 index + 1, len(sequence), " " * 70,
                 now, " " * 11,
                 left, " " * 31,
                 now + left if left is not None else None, " " * 11,
                 "=" * 80,
                 )
        yield item
    now = datetime.now()
    log.info(progress_bar_template,
             "=" * 80,
             "Done",
             index + 1, len(sequence), " " * 70,
             now, " " * 11,
             0, " " * 31,
             now, " " * 11,
             "=" * 80,
             )
