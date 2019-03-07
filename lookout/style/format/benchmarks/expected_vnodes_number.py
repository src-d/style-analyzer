"""Utility to workaround https://github.com/src-d/style-analyzer/issues/557."""
import csv
import logging
import logging.handlers
import os
import subprocess
import tempfile
from typing import Optional

import docker
import docker.errors
from lookout.core.helpers.analyzer_context_manager import AnalyzerContextManager
from lookout.core.helpers.server import find_port
from tqdm import tqdm

from lookout.style.format.analyzer import FormatAnalyzer
from lookout.style.format.benchmarks.quality_report import ensure_repo, handle_input_arg
from lookout.style.format.feature_extractor import FeatureExtractor


docker_client = docker.from_env()
bblfsh_name = "expected_vnodes_bblfshd"


def _stop_bblfshd():
    try:
        docker_client.containers.get(bblfsh_name).remove(force=True)
    except docker.errors.NotFound:
        return


def _restart_bblfshd(first_run: bool=False) -> str:
    log = logging.getLogger(bblfsh_name)
    port = find_port()
    log.info("Restarting bblfshd")
    if not first_run:
        _stop_bblfshd()
    docker_client.containers.run(
        "bblfsh/bblfshd:v2.11.0", detach=True, auto_remove=True, privileged=True,
        name=bblfsh_name, ports={9432: port})
    log.info("Installing javascript driver")
    bblfsh = docker_client.containers.get("expected_vnodes_bblfshd")
    bblfsh.exec_run("bblfshctl driver install javascript docker://bblfsh/javascript-driver:v2.7.1")
    address = "localhost:%d" % port
    log.info("bblfshd available at %s" % address)
    return address


def get_vnodes_number(repository: str, from_commit: str, to_commit: str,
                      context: AnalyzerContextManager, bblfsh: Optional[str]) -> int:
    """
    Calculate the number of expected vnodes number for a repository.

    :param repository: URL of repository.
    :param from_commit: Hash of the base commit.
    :param to_commit: Hash of the head commit.
    :param context: LookoutSDK instance to query analyzer.
    :param bblfsh: Babelfish server address to use. Specify None to use the default value.
    :return: expected vnodes number for a repository.
    """
    expected_vnodes_number = -1

    def _convert_files_to_xy(self, parsed_files):
        nonlocal expected_vnodes_number
        if expected_vnodes_number != -1:
            raise RuntimeError("_files_to_xy should be called only one time.")
        expected_vnodes_number = sum(len(vn) for vn, _, _ in parsed_files)
        raise RuntimeError("Forced FormatAnalyser.train call stop.")

    try:
        _convert_files_to_xy_backup = FeatureExtractor._convert_files_to_xy
        FeatureExtractor._convert_files_to_xy = _convert_files_to_xy
        with tempfile.TemporaryDirectory(prefix="top-repos-quality-repos-") as tmpdirname:
            git_dir = ensure_repo(repository, tmpdirname)
            try:
                context.push(fr=from_commit, to=to_commit, git_dir=git_dir, log_level="info",
                             bblfsh=bblfsh)
            except subprocess.CalledProcessError as e:
                # Force stop expected
                pass
    finally:
        FeatureExtractor._convert_files_to_xy = _convert_files_to_xy_backup
    return expected_vnodes_number


def calc_expected_vnodes_number_entry(input: str, output: str, runs: int) -> None:
    """
    Entry point for `python -m lookout.style.format calc-expected-support` command.

    :param input: сsv file with repositories for quality report. Should contain url, to and from \
                 columns.
    :param output: Path to a output csv file.
    :param runs: Repeat number to ensure the result correctness.
    """
    log = logging.getLogger("expected_vnodes_number")
    handler = logging.handlers.RotatingFileHandler(output + ".errors")
    handler.setLevel(logging.ERROR)
    log.addHandler(handler)

    repositories = list(csv.DictReader(handle_input_arg(input)))
    try:
        bblfsh = _restart_bblfshd(first_run=True)
        for cur_run in range(runs):
            with tempfile.TemporaryDirectory() as tmpdirname:
                database = os.path.join(tmpdirname, "db.sqlite3")
                fs = os.path.join(tmpdirname, "models")
                os.makedirs(fs, exist_ok=fs)
                with AnalyzerContextManager(FormatAnalyzer, db=database, fs=fs,
                                            init=False) as server:
                    for row in tqdm(repositories):
                        try:
                            vnodes_number = get_vnodes_number(
                                row["url"], to_commit=row["to"], from_commit=row["from"],
                                context=server, bblfsh=bblfsh)
                            log.info("%d/%d run. Expected vnodes number for %s is %d.",
                                     cur_run + 1, runs, row["url"], vnodes_number)
                            if row.get("vnodes_number", vnodes_number) != vnodes_number:
                                log.warning("vnodes number is different for %d/%d run. Get %d "
                                            "instead of %d. Set to nan.", cur_run + 1, runs,
                                            vnodes_number, row["vnodes_number"])
                                row["vnodes_number"] = float("nan")
                            else:
                                row["vnodes_number"] = vnodes_number
                        except Exception:
                            log.exception("-" * 20 + "\nFailed to process %s repo", row["url"])
                            continue
                        bblfsh = _restart_bblfshd()
    finally:
        _stop_bblfshd()

    fieldnames = ["url", "to", "from", "vnodes_number"]
    with open(output, "w") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in repositories:
            writer.writerow(row)
