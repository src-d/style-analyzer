import os
from pathlib import Path
import tempfile
import unittest

from dulwich.index import build_index_from_tree
from dulwich.repo import Repo

from lookout.style.format.benchmarks import generate_smoke


class DescriptionsTests(unittest.TestCase):
    def test_describe_rule_ordinal(self):
        init_test_repo_path = str(Path(__file__).parent / "test_repo.tar.xz")
        init_repo_files = {"test_repo": {"test_file.js": "test\n\n"}}

        generate_smoke.js_format_rules = {"test": ("init", "test")}
        with tempfile.TemporaryDirectory(prefix="benchmark-test") as testdir:
            self.assertEqual(
                0, generate_smoke.generate_smoke_entry(init_test_repo_path, testdir, force=True))
            self.assertEqual(set(os.listdir(testdir)) - {"index.csv"}, set(init_repo_files.keys()))
            for init_repo in init_repo_files:
                repo = Repo(os.path.join(testdir, init_repo))
                repo.hooks.clear()  # Speed up dulwich by ~25%
                self.assertEqual(set(init_repo_files[init_repo]),
                                 set(os.listdir(repo.path)) - {".git"})
                walker = repo.get_graph_walker((b"refs/heads/test",))
                next(walker)  # Skip head commit. We need HEAD~1
                before_head = next(walker)
                build_index_from_tree(repo.path, repo.index_path(), repo.object_store,
                                      repo[before_head].tree)
                for filename in init_repo_files[init_repo]:
                    self.assertEqual(init_repo_files[init_repo][filename],
                                     (Path(testdir) / init_repo / filename).read_text())


if __name__ == "__main__":
    unittest.main()
