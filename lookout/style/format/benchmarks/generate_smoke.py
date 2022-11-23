"""A module for repositories generation with different styles in different branches."""
import csv
import logging
import os
from pathlib import Path
import re
import shutil
import tarfile
from typing import Tuple

from dulwich.index import blob_from_path_and_stat, build_index_from_tree
from dulwich.porcelain import branch_create, update_head
from dulwich.repo import Repo


class _NoChangesException(Exception):
    """Exception should be raised if no changes were made to initial code."""

    pass


index_filename = "index.csv"

js_operators = [
    "+", "-", "*", "/", "%", "++", "--",
    "=", "+=", "-=", "/=", "%=",
    "==", "===", "!=", "!==", ">", "<", ">=", "<=",
    "&", "|", "~", "^", ">>", "<<", ">>>",
    "&&", "||", "!",
    "?", ":",
]

js_operators_regex = "|".join(re.escape(op) for op in sorted(js_operators, reverse=True))

js_format_rules = {
    "equal_no_space_style": (" = ", "="),
    "operators_no_space_style": (" (%s) " % js_operators_regex, r"\1"),
    "no_newline_before_function": ("(\n+)function ", "\nfunction "),
    "two_newlines_before_function": ("(\n+)function ", "\n\nfunction "),
    "no_space_after_if": (r"if\s*\(", r"if("),
    "no_spaces_inside_brackets": (r"\{ +([^{}]+) +\}", r"{\1}"),
    "no_spaces_inside_round_brackets": (r"\( +([^()]+) +\)", r"(\1)"),
    "spaces_inside_round_brackets": (r"\(([^()]+)\)", r"( \1 )"),
}


def commit_style(repo: Repo, format_rule_name: str) -> Tuple[str, str]:
    """
    Call bash script which commit all changes to `style_name` branch and checkout master back.

    :param repo: Repo instance to the repository for which style were applied.
    :param format_rule_name: Applied format rule name.
    :return: Two commit hashes: where style was applied and where style was disrupt.
    """
    def commit(repo: Repo, msg: str) -> str:
        """Commit everything."""
        for tree_path, entry in repo.open_index().items():
            full_path = os.path.join(repo.path.encode(), tree_path)
            blob = blob_from_path_and_stat(full_path, os.lstat(full_path))
            if blob.id != entry.sha:
                repo.stage(tree_path)
        return repo.do_commit(msg.encode(), b"Source{d} ML Team <ml@sourced.tech>")

    repopath = repo.path
    base = repo.head()
    branch_create(repopath, format_rule_name, force=True)
    update_head(repopath, format_rule_name)
    style_commit_sha = commit(repo, format_rule_name)
    build_index_from_tree(repo.path, repo.index_path(), repo.object_store, repo[base].tree)
    revert_style_commit_sha = commit(repo, "Revert " + format_rule_name)
    update_head(repopath, b"master")
    return style_commit_sha.decode(), revert_style_commit_sha.decode()


def generate_smoke_entry(inputpath: str, outputpath: str, force: bool = False) -> int:
    """
    Generate repositories with different style in separate branches and its violations.

    :param inputpath: Path to the tar.gz archive containing initial repositories.
    :param outputpath: Path to the directory where the generated dataset should be stored.
    :param force: Override outputpath directory if exists.

    :return: Success status. 0 if Succeeded, 1 otherwise.
    """
    log = logging.getLogger("styles-gen")
    if not inputpath.endswith(".tar.xz"):
        raise ValueError("Input file should be .tar.xz archive.")
    inputpath = Path(inputpath)
    outputpath = Path(outputpath)
    if force and outputpath.exists():
        shutil.rmtree(str(outputpath))
    try:
        outputpath.mkdir()
    except FileExistsError:
        log.error("Directory %s exists. If you want to override it run with --force flag.",
                  outputpath)
        return 1
    with tarfile.open(str(inputpath)) as tar:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tar, str(outputpath))
    repopaths = [x for x in outputpath.iterdir() if x.is_dir()]
    log.info("Repos found: %s", ", ".join(r.stem for r in repopaths))
    with open(str(outputpath / index_filename), "w") as index_file:
        writer = csv.DictWriter(index_file, fieldnames=["repo", "style", "from", "to"])
        writer.writeheader()
        for style_name, (pattern, repl) in js_format_rules.items():
            pattern = re.compile(pattern)
            for repopath in repopaths:
                repo = Repo(str(repopath))
                repo.hooks.clear()  # Speed up dulwich by ~25%
                for filepath in repopath.glob("**/*.js"):
                    code = filepath.read_text()
                    new_code = re.sub(pattern, repl, code)
                    filepath.write_text(new_code)
                from_commit, to_commit = commit_style(repo, style_name)
                writer.writerow({
                    "repo": repopath.name,
                    "style": style_name,
                    "from": from_commit,
                    "to": to_commit,
                })
    return 0
