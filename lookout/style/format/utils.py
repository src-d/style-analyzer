"""Commonly used utils."""
import cProfile
from functools import wraps
import io
import pstats
from typing import Callable, Iterable

from bblfsh import BblfshClient
from bblfsh.client import NonUTF8ContentException
from tqdm import tqdm

from lookout.core.api.service_data_pb2 import File
from lookout.style.format.files_filtering import filter_filepaths


def prepare_files(filenames: Iterable[str], client: BblfshClient,
                  language: str) -> Iterable[File]:
    """
    Prepare the given folder for analysis by extracting UASTs and creating the gRPC wrappers.

    :param filenames: List of paths to files to analyze.
    :param client: Babelfish client. Babelfish server should be started accordingly.
    :param language: Language to consider. Will discard the other languages
    :return: Iterator of File-s with content, uast, path and language set.
    """
    files = []
    for file in tqdm(filter_filepaths(list(filenames))):
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

    :param func: function to be wrapped.
    :return: wrapped function.
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
