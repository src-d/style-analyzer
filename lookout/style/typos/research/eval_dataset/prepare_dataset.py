"""
Filter and prepare dataset for evaluation.
It should be launched on dataset prepared by `typos_preprocessing.ipynb`.
"""
import argparse
from collections import defaultdict
import datetime
from difflib import SequenceMatcher
import os
import logging as log
import re
import subprocess
import tempfile
import time
from typing import NamedTuple, Optional
from tqdm import tqdm

import pandas as pd
from joblib import Parallel, delayed, Memory


Changes = NamedTuple("Changes", (
    ("date", datetime.datetime),  # date of change
    ("line", str),  # line from blame
    ("hash", str),  # hash of commit from blame
))

COLUMNS = ["identifier", "correct_id", "filename", "line", "commit", "repository"]
NEW_COLUMNS = COLUMNS + ["first_hash", "blame_filename"]
COL2IND = {c: i for i, c in enumerate(COLUMNS)}
NEW_COL2IND = {c: i for i, c in enumerate(NEW_COLUMNS)}


class IdentifierFileCommitRanger:
    """Find first commit where identifier was added to the file."""
    _log = log.getLogger("CommitRange")

    def __init__(self, *, filename: str, repository: str, identifier: str,
                 commit: str, directory: Optional[str] = None):
        """
        :param filename: name of file to check.
        :param repository: repository in `org/name` format.
        :param directory: directory to store results.
        :param identifier: identifier to search.
        :param commit: commit where identifier was fixed.
        """
        self.filename = filename
        self.repository = "https://github.com/" + repository
        self.directory = directory
        self.identifier = identifier
        self.commit = commit

    def _run_cmd(self, cmd, step, cwd=None, env=None):
        try:
            return subprocess.check_output(cmd, cwd=cwd, env=env, stderr=subprocess.STDOUT)\
                .decode()
        except subprocess.CalledProcessError as e:
            err = "Repository %s failed with exception '''%s''' at %s step" % (self.repository, e,
                                                                               step)
            self._log.error(err)
            raise e

    def _clone(self):
        os.makedirs(self.directory, exist_ok=True)
        cmd = "git clone --quiet".split() + [self.repository, self.directory]
        self._run_cmd(cmd, "cloning", env=dict(os.environ, GIT_TERMINAL_PROMPT="0"))
        return self

    def _checkout(self):
        cmd = "git checkout --quiet".split() + [self.commit]
        self._run_cmd(cmd, "checkout", cwd=self.directory)
        return self

    def _blame(self, filename=None):
        if not filename:
            filename = self.filename
        cmd = "git blame HEAD^ --".split() + [filename]
        return self._run_cmd(cmd, "blame", cwd=self.directory)

    @staticmethod
    def _validate_date(text):
        try:
            return datetime.datetime.strptime(text, "%Y-%m-%d")
        except ValueError:
            return None

    def _get_full_hash(self, short_hash):
        cmd = "git rev-parse".split() + [short_hash]
        return self._run_cmd(cmd, "full_hash", cwd=self.directory).strip()

    def _get_diff(self):
        cmd = "git diff HEAD^ HEAD".split()
        return self._run_cmd(cmd, "diff", cwd=self.directory).strip()

    def _to_changes(self, line):
        parts = line.strip().split()
        full_hash = self._get_full_hash(parts[0])
        for part in parts[1:]:
            date = self._validate_date(part)
            if date:
                return Changes(hash=full_hash, date=date, line=line)
        else:
            raise ValueError("Line doesn't contain date: %s" % line)

    def _pipeline(self):
        blame = self._clone()._checkout()
        try:
            blame = blame._blame()
            filename = self.filename
        except Exception as exception:
            fatal = "fatal: no such path %s in HEAD^" % self.filename
            if exception.stdout.decode().strip() == fatal:
                # search in diffs for removed file
                diff = self._get_diff()
                filename = self._find_deleted_file(diff, self.filename)
                blame = blame._blame(filename=filename)
            else:
                raise exception
        res = [self._to_changes(line) for line in blame.splitlines() if self.identifier in line]
        # return hash of the earliest date
        return sorted(res, key=lambda x: x.date)[0].hash, filename

    def __call__(self):
        if self.directory:
            return self._pipeline()
        else:
            with tempfile.TemporaryDirectory() as tmpdirname:
                self.directory = tmpdirname
                return self._pipeline()
                self.directory = None

    @staticmethod
    def _find_deleted_file(text, filename=None):
        diff_pattern = "diff --git a/%s b/%s"
        pattern = r"%s" % (diff_pattern % ("(.*)", "(.*)"))

        # find all changed files
        m = re.findall(pattern, text)
        pos = 0
        res = []
        for f, l in zip(m[:-1], m[1:]):
            f_pattern = diff_pattern % f
            l_pattern = diff_pattern % l
            f_pos = text.find(f_pattern, pos) + len(f_pattern)
            l_pos = text.find(l_pattern, f_pos)
            res.append((f, text[f_pos:l_pos]))
            pos = l_pos
        res.append((l, text[l_pos:]))
        # select only deleted and created
        created, deleted = None, []
        for names, txt in res:
            lines = txt.splitlines()
            if lines[4] == "+++ /dev/null":
                # deleted file
                assert names[0] == names[1]
                deleted.append((names[0], txt[txt.find("@@", txt.find("@@") + 2) + 2:],
                                "\n".join(line[1:] for line in lines[6:])))
            elif lines[3] == "--- /dev/null" and names[0] == filename:
                # created file
                assert names[0] == names[1]
                assert created is None
                created = (names[0], txt[txt.find("@@", txt.find("@@") + 2) + 2:],
                           "\n".join(line[1:] for line in lines[6:]))

        deleted = sorted(deleted, key=lambda x: SequenceMatcher(a=x[2], b=created[-1]).ratio())
        return deleted[-1][0]


def _parallel_comp(args):
    try:
        return IdentifierFileCommitRanger(filename=args[COL2IND["filename"]],
                                          repository=args[COL2IND["repository"]],
                                          identifier=args[COL2IND["identifier"]],
                                          commit=args[COL2IND["commit"]],
                                          directory=None)()
    except:  # noqa: E722
        # in case of any exception we should return something
        return "error", "error"


def pipeline(input_csv, output_csv, n_cores=1, cache="/tmp"):
    """
    Find first commit hash of appearing identifier in file.

    :param input_csv: Path to input csv.
    :param output_csv: Path to store result csv.
    :param n_cores: How many cores to use.
    :param cache: Cache location. If empty - no caching
    """
    if cache:
        memory = Memory(cache, verbose=0)
        parallel_comp = memory.cache(func=_parallel_comp)
    else:
        parallel_comp = _parallel_comp
    df = pd.read_csv(input_csv, header=None)
    df.columns = COLUMNS

    args = [tuple(getattr(line, col) for col in COLUMNS) for i, line in df.iterrows()]
    res = Parallel(n_jobs=n_cores)(tqdm((delayed(parallel_comp)(arg) for arg in args),
                                        total=len(args) - 1))

    new_args = [arg + (h, filename) for arg, (h, filename) in zip(args, res)]
    to_df = defaultdict(list)
    for arg in new_args:
        for col in NEW_COLUMNS:
            to_df[col].append(arg[NEW_COL2IND[col]])
    new_df = pd.DataFrame.from_dict(to_df)
    new_df = new_df[NEW_COLUMNS]
    new_df.to_csv(output_csv, index=False, header=False, compression="gzip")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-csv", help="Path to input csv.", required=True)
    parser.add_argument("-o", "--output-csv", help="Path to store result csv.", required=True)
    parser.add_argument("-c", "--cache", default="",
                        help="Cache location. If empty - no caching")
    parser.add_argument("-n", "--n-cores", type=int, default=1,
                        help="How many cores to use.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    start = time.time()
    pipeline(**vars(args))
    print("Success! Total duration of pipeline is", time.time() - start)
