"""Cloning utilities for git repositories."""

from itertools import repeat
import logging
import multiprocessing
import os
from typing import Dict, Optional, Sequence, Tuple

from dulwich import porcelain


class Cloner:
    """
    Class to clone a list of git repositories in parallel.
    """

    _log = logging.getLogger("Cloner")

    def __init__(self, location: str, processes: Optional[int] = None):
        """
        Create new Cloner instance.

        :param location: Path to the dirrectory where repositories is saved.
        :param processes: Number of processes to use. 'os.cpu_count()' is used by default.
        """
        os.makedirs(location, exist_ok=True)
        self._location = location
        self._processes = processes

    def clone_repositories(self, repositories: Sequence[str]) -> Dict[str, str]:
        """
        Run repositories cloning in parallel.

        :param repositories: List of URLs to clone.
        :return: Mapping from provided URL to the location in the file system for successfully \
                 downloaded repositories.
        """
        self._log.info("start cloning %d repositories", len(repositories))
        with multiprocessing.Pool(self._processes) as pool:
            repo_paths = pool.map(self._clone_repository,
                                  zip(repositories, repeat(self._location)))
        repositories_dir = {repo: git_dir for repo, git_dir in repo_paths if git_dir}
        self._log.info("successfully cloned %d/%d repositories",
                       len(repositories_dir), len(repositories))
        return repositories_dir

    @staticmethod
    def get_repo_name(url: str) -> str:
        """
        Convert URL to repository name.

        :param url: Repository URL to get name for.
        :return: Generated repository name.
        """
        return "-".join(url.split("/")[-2:])

    @staticmethod
    def _clone_repository(args: Tuple[str, str]) -> Tuple[str, Optional[str]]:
        repository, repos_cache = args
        if os.path.exists(repository):
            Cloner._log.info("%s exists", repository)
            return repository, repository
        git_dir = os.path.join(repos_cache, Cloner.get_repo_name(repository))
        if os.path.exists(git_dir):
            Cloner._log.info("%s was found at %s. Skipping", repository, git_dir)
            return repository, git_dir
        try:
            porcelain.clone(repository, git_dir)
            Cloner._log.info("%s was cloned to %s", repository, git_dir)
            return repository, git_dir
        except Exception:
            Cloner._log.exception("Failed to clone %s to %s", repository, git_dir)
            return repository, None
