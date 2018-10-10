"""Commonly used utils."""
import cProfile
from functools import wraps
from copy import deepcopy
import glob
import io
import pstats
from typing import Mapping, Callable, Iterable

from bblfsh import BblfshClient
from bblfsh.client import NonUTF8ContentException
from tqdm import tqdm

from lookout.core.api.service_data_pb2 import File
from lookout.style.format.files_filtering import filter_filepaths


def prepare_files(folder: str, client: BblfshClient, language: str) -> Iterable[File]:
    """
    Prepare the given folder for analysis by extracting UASTs and creating the gRPC wrappers.

    :param folder: Path to the folder to analyze.
    :param client: Babelfish client. Babelfish server should be started accordingly.
    :param language: Language to consider. Will discard the other languages.
    :return: Iterator of File-s with content, uast, path and language set.
    """
    files = []

    # collect filenames with full path
    filenames = glob.glob(folder, recursive=True)

    for file in tqdm(filter_filepaths(filenames)):
        try:
            res = client.parse(file)
        except NonUTF8ContentException:
            # skip files that can't be parsed because of UTF-8 decoding errors.
            continue
        if res.status == 0 and res.language.lower() == language.lower():
            uast = res.uast
            path = file
            with open(file) as f:
                content = f.read().encode("utf-8")
            files.append(File(content=content, uast=uast, path=path,
                              language=res.language.lower()))
    return files


def profile(func: Callable) -> Callable:
    """
    Profiling decorator.

    :param func: Function to be wrapped.
    :return: Wrapped function.
    """
    @wraps(func)
    def wrapped_profile(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        try:
            res = func(*args, **kwargs)
        finally:
            pr.disable()
            s = io.StringIO()
            ps = pstats.Stats(pr, stream=s).strip_dirs().sort_stats("cumulative", "time")
            ps.print_stats(20)
            print(s.getvalue())
        return res
    return wrapped_profile


def merge_dicts(*dicts: Mapping) -> Mapping:
    """
    Deep merge of nested dictionaries.

    Operation is not commutative, each next dictionary overrides values of the previous one for
    the same keys sequence.
    (see example). Example:
    >>> a = {1: 1, 2: {3: 3, 4: 4}}
    >>> b = {1: 10, 2: {3: 30, 4: 4, 5: 5}}
    >>> merge_dicts(a, b)
    {1: 10, 2: {3: 30, 4: 4, 5: 5}}
    >>> merge_dicts(b, a)
    {1: 1, 2: {3: 3, 4: 4, 5: 5}}

    :return: New merged dictionary.
    """
    if len(dicts) == 0:
        raise ValueError("At least one argument is required.")
    if len(dicts) == 1:
        return deepcopy(dicts[0])
    res = deepcopy(dicts[0])
    stack = [(res, d) for d in dicts[:0:-1]]
    while stack:
        d1, d2 = stack.pop()
        for key, value in d2.items():
            if isinstance(value, dict):
                stack.append((d1.setdefault(key, {}), value))
            else:
                d1[key] = value
    return res
