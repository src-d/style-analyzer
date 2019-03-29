import json
import tempfile
import unittest
from unittest import mock

from modelforge import slogging

from lookout.style.cloner import Cloner
from lookout.style.format.tests.test_quality_report import Capturing

downloaded_repos = []


def fake_clone(repo, *args, **kwargs):
    if repo == "failed repo":
        raise ValueError("test")
    downloaded_repos.append(repo)


def fake_exists(path: str) -> bool:
    if path == "existing_repo":
        return True
    if path != "downloaded_repo" and path.endswith("downloaded_repo"):
        return True
    return False


class ClonerTests(unittest.TestCase):
    @mock.patch("os.path.exists", side_effect=fake_exists)
    @mock.patch("dulwich.porcelain.clone", side_effect=fake_clone)
    def test_cloner(self, f1, f2):
        slogging.setup("DEBUG", True)
        with Capturing() as output:
            with tempfile.TemporaryDirectory() as tmp_dir:
                cloner = Cloner(tmp_dir, 1)
                cloner.clone(
                    ["repo1", "repo2", "repo3", "existing_repo", "downloaded_repo", "failed repo"])
        expected_log = [
            "started cloning 6 repositories",
            "repo1 was cloned to",
            "repo2 was cloned to",
            "repo3 was cloned to",
            "existing_repo exists",
            "downloaded_repo was found at",
            "failed to clone",
            "successfully cloned 5/6 repositories",
        ]
        output = (json.loads(log_entry)["msg"] for log_entry in output)
        output = [o for o in output if "| elapsed" not in o and "Using backend" not in o]
        self.assertEqual(len(output), len(expected_log))
        for expected_msg, log_msg in zip(expected_log, output):
            self.assertEqual(expected_msg, log_msg[:len(expected_msg)])

        self.assertEqual(set(downloaded_repos), {"repo1", "repo2", "repo3"})

    def test_get_repo_name(self):
        self.assertEqual(Cloner.get_repo_name("https://github.com/src-d/style-analyzer"),
                         "github.com@src-d@style-analyzer")
        self.assertEqual(Cloner.get_repo_name("src-d/style-analyzer"), "src-d@style-analyzer")
        self.assertEqual(Cloner.get_repo_name("style-analyzer"), "style-analyzer")


if __name__ == "__main__":
    unittest.main()
