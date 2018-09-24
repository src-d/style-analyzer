"""Utils for the debugging scripts."""
from glob import glob
from os.path import isfile
import re
from typing import Iterable, Optional

from bblfsh.client import BblfshClient, NonUTF8ContentException
from tqdm import tqdm

from lookout.core.api.service_data_pb2 import File
from lookout.core.garbage_exclusion import GARBAGE_PATTERN


def filter_filepaths(filepaths: Iterable[str], line_length_limit: int = 500,
                     exclude_pattern: Optional[str] = None) -> Iterable[str]:
    """
    Mirror of the file filtering used in the format analyzer for use by debugging tools.

    :param paths: Iterable of paths to filter.
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
        with open(filepath, 'rb') as fh:
            if len(max(fh, key=len, default=b"")) <= line_length_limit:
                yield filepath


def prepare_file(filename: str, client: BblfshClient, language: str) -> File:
    """
    Prepare the given file for analysis by extracting UAST and creating the gRPC wrapper.

    :param filename: Path to the filename to analyze.
    :param client: Babelfish client. Babelfish server should be started accordingly.
    :param language: Language to consider. Will discard the other languages
    """
    assert isfile(filename), "\"%s\" should be a file" % filename
    res = client.parse(filename, language)
    assert res.status == 0, "Parse returned status %s for file %s" % (res.status, filename)
    error_log = "Language for % should be %s instead of %s"
    assert res.language.lower() == language.lower(), error_log % (filename, language, res.language)

    with open(filename) as f:
        content = f.read().encode("utf-8")

    return File(content=content, uast=res.uast, path=filename)


def prepare_files(folder: str, client: BblfshClient, language: str) -> Iterable[File]:
    """
    Prepare the given folder for analysis by extracting UASTs and creating the gRPC wrappers.

    :param folder: Path to the folder to analyze.
    :param client: Babelfish client. Babelfish server should be started accordingly.
    :param language: Language to consider. Will discard the other languages
    """
    files = []

    # collect filenames with full path
    filenames = glob(folder, recursive=True)

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
            files.append(File(content=content, uast=uast, path=path, language=res.language))
    return files
