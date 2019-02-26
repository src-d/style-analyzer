from copy import deepcopy
from typing import Mapping


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
