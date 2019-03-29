"""Cloning utilities for git repositories."""

import logging
import os
from typing import Dict, Optional, Sequence, Tuple

from dulwich import porcelain
from joblib import delayed, Parallel


class ParallelWriteToLogs(Parallel):
    """Parallel which logs progress using logger."""

    _log = logging.getLogger("Parallel")

    def _print(self, msg, msg_args):
        """Display the message using `self._log.info`."""
        if not self.verbose:
            return
        self._log.info(msg, *msg_args)


class Cloner:
    """Class to clone a list of git repositories in parallel."""

    _log = logging.getLogger("Cloner")

    def __init__(self, location: str, n_jobs: Optional[int] = -1):
        """
        Initialize a new Cloner instance.

        :param location: Path to the dirrectory where repositories is saved.
        :param n_jobs: Number of processes to use. 'os.cpu_count()' is used by default.
        """
        os.makedirs(location, exist_ok=True)
        self._location = location
        self._n_jobs = n_jobs

    def clone(self, repositories: Sequence[str]) -> Dict[str, str]:
        """
        Run repositories cloning in parallel.

        :param repositories: List of URLs to clone.
        :return: Mapping from the provided URLs to the file system paths of the successfully \
                 downloaded repositories.
        """
        urllib_logger = logging.getLogger("urllib3.connectionpool")
        backup_level = urllib_logger.level
        urllib_logger.setLevel(logging.WARNING)  # Mute urllib3 logging
        try:
            self._log.info("started cloning %d repositories", len(repositories))
            repo_paths = ParallelWriteToLogs(
                n_jobs=self._n_jobs, verbose=10, backend="multiprocessing")(
                delayed(self._clone_repository)(repo, self._location) for repo in repositories)
            repositories_dir = {repo: git_dir for repo, git_dir in repo_paths if git_dir}
            self._log.info("successfully cloned %d/%d repositories",
                           len(repositories_dir), len(repositories))
        finally:
            urllib_logger.setLevel(backup_level)
        return repositories_dir

    @staticmethod
    def get_repo_name(url: str) -> str:
        """
        Convert URL to repository name.

        :param url: Repository URL to get name for.
        :return: Generated repository name.
        """
        return "@".join(url.split("/")[-3:])

    @staticmethod
    def _clone_repository(repository: str, repos_cache: str) -> Tuple[str, Optional[str]]:
        if os.path.exists(repository):
            Cloner._log.info("%s exists", repository)
            return repository, os.path.abspath(repository)
        git_dir = os.path.join(repos_cache, Cloner.get_repo_name(repository))
        if os.path.exists(git_dir):
            Cloner._log.info("%s was found at %s. Skipping", repository, git_dir)
            return repository, os.path.abspath(git_dir)
        try:
            with open(os.devnull, "wb") as devnull:
                porcelain.clone(repository, git_dir, bare=False, errstream=devnull)
            Cloner._log.debug("%s was cloned to %s", repository, git_dir)
            return repository, os.path.abspath(git_dir)
        except Exception:
            Cloner._log.exception("failed to clone %s to %s", repository, git_dir)
            return repository, None
