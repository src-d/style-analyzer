"""Measure quality on several top repositories."""
from argparse import ArgumentParser
import glob
import importlib
import io
import logging
import os
import subprocess
import tempfile
import traceback
from typing import Iterable, Sequence, Type, Union

from lookout.core.analyzer import Analyzer
from lookout.core.cmdline import ArgumentDefaultsHelpFormatterNoNone, create_model_repo_from_args
from lookout.core.data_requests import DataService
from lookout.core.event_listener import EventListener
from lookout.core.manager import AnalyzerManager
from lookout.core.test_helpers.server import fetch, find_port, run as launch_server
from tabulate import tabulate

from lookout.style.format.quality_report import QualityReportAnalyzer


# TODO: add after https://github.com/src-d/style-analyzer/issues/329 fixed
"""https://github.com/meteor/meteor c3309b123a7220ac24cbe73661184ee946bca01f 62fa9927ce34cff064cc3991439553e7c52b5258
https://github.com/webpack/webpack babe736cfa1ef7e8014ed32ba4a4ec38049dce14 3e74cb428af04eedac60ae13d2420d2b5bd3bde1"""  # noqa: E501
# TODO: add after https://github.com/bblfsh/client-python/issues/89 fixed
"""https://github.com/vuejs/vue b7105ae8c9093e36ec89a470caa3b78bda3ef467 db1d0474997e9e60b8b0d39a3b7c2af55cfd0d4a
https://github.com/vuejs/vuex 2e62705d4bce4ebcb8eca23df8c7b849125fc565 1ac16a95c574f6b1386016fb6d4f00cfd2ee1d60"""  # noqa: E501
# TODO: add after https://github.com/bblfsh/bblfshd/issues/219 fixed
"""https://github.com/nodejs/node 6eda924c189e44a36fc97a7cfae41b69483d5bfb 315b1c656cee39c989015cc2b17fe8c864dbc3dd"""  # noqa: E501
# TODO: add after modifying timeout config for `analyzer_push` (https://github.com/src-d/style-analyzer/issues/365)  # noqa: E501
"""https://github.com/atom/atom 108b23210759a8c5b2f51ac99659be5dc31a7371 a3c320dd707b915da2192427bcceea166edbd6d4"""  # noqa: E501
# TODO: add after https://github.com/src-d/style-analyzer/issues/364 fixed
"""https://github.com/segmentio/evergreen ba22d511dad83c072842e47801ef42697d142f7c 1030eca5da38dce4e5047c23a3ea7fc0c246b8ce"""  # noqa: E501
# format: "repository to-commit from-commit"
REPOSITORIES = """https://github.com/30-seconds/30-seconds-of-code 3a122c9cfcbdc091227879a06a32bc67ccd0d35d c8c60895e80b8bc90583502accdaa339b794609c
https://github.com/axios/axios 21ae22dbd3ae3d3a55d9efd4eead3dd7fb6d8e6e 138108ee56bd689305ae505a66b48d5e9c8aa494
https://github.com/GoogleChromeLabs/carlo b8ce2bca042c757b13fc82a3e059980342ddd9a8 26262aad740b7255f17950251ae344b4823572a4
https://github.com/nodejs/citgm 0c4c7ccdd1cad8ce9506e34ca523787ba18cafe2 d21e2a87aaa9e9f50c6175eddc54054a32c64a24
https://github.com/facebook/create-react-app 32106d216e4c31fda30ec475f9f03186d116c893 ffc63d55976f9cbcce7f33dc7c45b3c2190a5924
https://github.com/expressjs/express b4eb1f59d39d801d7365c86b04500f16faeb0b1c 56e90e3c7267782febe35754806ce3f63b527485
https://github.com/freeCodeCamp/freeCodeCamp cf65516cce60645a417e44c4fcea7418ca920572 c353c4c659c3dcb19524ba893f170805c931a44a
https://github.com/jquery/jquery dae5f3ce3d2df27873d01f0d9682f6a91ad66b87 d9a099a58e1bb1f158ea66ec55534770be442907
https://github.com/facebook/react 1034e26fe5e42ba07492a736da7bdf5bf2108bc6 8ced545e3df95afab6fa35bc29f9320bafbcef26
https://github.com/facebook/react-native 1850906e5e557beb2234a1708cfc5fe8e7b4f0bf 764dd511d21aa4dac815c59a7d72497267fde08a
https://github.com/reduxjs/redux 902484ed735d38aec06683c847810a7218d8dba2 b307091af4d7e846d9e5f080fb81caf4a8b4aab1
https://github.com/storybooks/storybook b28217f887af533a17cb1498887d6b4bd41bd643 4c46f273719427788d568c037f285907aabd17f9
https://github.com/laravel/telescope 534030114f47696fe3f3b08ea7ca49467428f2af 6f0a10ec586cfa1a22218b6778bf9c1572b97912
https://github.com/vuejs/vue-cli 2024f4e52097bed96b527d2e519dccba334f434b 85e4f9839ba88d1283b10bb3112582792b8dac6e""".split("\n")  # noqa: E501


class AnalyzerContextManager:
    """Context manager for launching analyzer."""

    def __init__(
            self, port: int, db: str, fs: str, config: str = "",
            analyzer: Union[str, Sequence[str], Iterable[Type[Analyzer]]] = "lookout.style.format"
    ):
        """
        Init analyzer: model_repository, data_service, arguments, etc.

        :param port: port to use for analyzer.
        :param db: database location.
        :param fs: location where to store results of launched analyzer.
        :param config: Path to the configuration file with option defaults. If empty - skip.
        :param analyzer: analyzer(s) to use.
        """
        self.port = port
        self.db = db
        self.fs = fs
        self.config_path = config  # mimic TestAnalyzer - not used so far
        if isinstance(analyzer, (str, type)):
            self.analyzer = [analyzer]
        if isinstance(self.analyzer[0], str):
            self.analyzer = [importlib.import_module(a).analyzer_class for a in self.analyzer]

        class Args:
            pass
        self.args = Args()
        self.args.db = "sqlite:///%s" % self.db
        self.args.fs = self.fs
        self.args.cache_size = "1G"
        self.args.cache_ttl = "6h"
        self.args.db_kwargs = {}
        self.args.workers = 1
        # initialize model repository
        self.model_repository = create_model_repo_from_args(self.args)
        self.model_repository.init()
        # initialize a new instance of DataService
        data_request_address = "0.0.0.0:10301"
        self.data_service = DataService(data_request_address)

    def __enter__(self):
        self.manager = AnalyzerManager(
            analyzers=self.analyzer,
            model_repository=self.model_repository,
            data_service=self.data_service,
        )
        self.listener = EventListener(address="0.0.0.0:%d" % self.port, handlers=self.manager,
                                      n_workers=self.args.workers)
        self.listener.start()
        return self

    def __exit__(self, exc_type=None, exc_val=None, exc_tb=None):
        self.listener.stop()


def train_on_repo(git_dir: str, from_commit: str, to_commit: str, db: str, fs: str, port: str) \
        -> str:
    """
    Prepare environment & train the model for tests.

    :param git_dir: Location of git directory.
    :param from_commit: Hash of commit.
    :param to_commit: Hash of commit.
    :param db: Location of database.
    :param fs: Folder to store results.
    :param port: Running analyzer port on localhost.
    :return: Location of trained rules.
    """
    # analyzer
    logging.warning("TestAnalyzer")
    with AnalyzerContextManager(port=port, db=db, fs=fs):
        # train the rules
        launch_server("push", from_commit, to_commit, port, git_dir=git_dir)
        # find the saved model
        filenames = glob.glob(os.path.join(fs, "**", "*"), recursive=True)
        res = [file for file in filenames if file.endswith(".asdf")]
        return res[0]


def get_repo_name(url: str) -> str:
    """
    Extract name of repository from URL.

    :param url: URL for repository.
    :return: name of repository.
    """
    return url.split("/")[-1]


def measure_quality(repository: str, from_commit: str, to_commit: str, bblfsh: str, config: str) \
        -> str:
    """
    Generate reports for repository and return them as string.

    :param repository: URL of repository.
    :param from_commit: Hash of commit.
    :param to_commit: Hash of commit.
    :param bblfsh: Bblfsh address to use.
    :param config: Path to config file.
                   Skip until https://github.com/src-d/style-analyzer/issues/365 fixed.
    :return: Report.
    """
    port = find_port()
    with tempfile.TemporaryDirectory() as tmpdirname:
        db = tempfile.NamedTemporaryFile(dir=tmpdirname)
        fs = tempfile.TemporaryDirectory(dir=tmpdirname)
        # clone repository
        git_dir = os.path.join(tmpdirname, get_repo_name(repository))  # location for code
        cmd = "git clone --single-branch --branch master %s %s" % (repository, git_dir)
        process = subprocess.Popen(cmd.split())
        output, error = process.communicate()
        assert error is None, "Something went wrong with repository %s " % repository
        # train model
        model_path = train_on_repo(git_dir=git_dir, from_commit=from_commit, to_commit=to_commit,
                                   db=db.name, fs=fs.name, port=port)
        # quality report
        analyzer_name = QualityReportAnalyzer
        captured_reports = []

        def capture_reports(f):
            def decorated(*args, **kwargs):
                captured_reports.append(f(*args, **kwargs))
                return captured_reports[-1]
            return decorated

        QualityReportAnalyzer.generate_model_report =  \
            capture_reports(QualityReportAnalyzer.generate_model_report)
        QualityReportAnalyzer.generate_report =  \
            capture_reports(QualityReportAnalyzer.generate_report)

        config_json = '{"style.format.analyzer.QualityReportAnalyzer": {"model_path": "%s", ' \
                      '"bblfsh_address": "%s"}}' % (model_path, bblfsh)
        with AnalyzerContextManager(port=port, db=db.name, fs=fs.name, analyzer=analyzer_name):
            launch_server("push", fr=from_commit, to=to_commit, port=port,
                          git_dir=git_dir, config_json=config_json)
            # find quality report
            st = [j for j, row in enumerate(captured_reports)
                  if "Classification report:" in row and "Quality report" in row]
            assert len(st) == 1
            st = st[0]
            quality_report = captured_reports[st]
            # find rule stat report
            st = [j for j, row in enumerate(captured_reports)
                  if "Rules summary:" in row and "# Model report for" in row]
            assert len(st) == 1
            st = st[0]
            model_report = captured_reports[st]
    return quality_report, model_report


def calc_weighted_avg(arr: Sequence[Sequence], col: int, weight_col: int = 3) -> float:
    """Calculate average value in `col` weighted by column `weight_col`."""
    numerator, denominator = 0, 0
    for row in arr:
        numerator += row[col] * row[weight_col]
        denominator += row[weight_col]
    if denominator == 0:
        return 1
    return numerator / denominator


def calc_avg(arr: Sequence[Sequence], col: int) -> float:
    """Calculate average value in `col`."""
    numerator, denominator = 0, 0
    for row in arr:
        numerator += row[col]
        denominator += 1
    if denominator == 0:
        return 1
    return numerator / denominator


def get_precision_recall_f1_support(report: str) -> (float, float, float, int):
    """Extract avg / total precision, recall, f1 score, support from report."""
    for line in report.split("\n"):
        if "weighted avg" in line:
            _, prec, recall, f1, support = line.split("` | `")
            prec, recall, f1, support = [float(f.replace("`", "").replace("|", "")) for f in
                                              (prec, recall, f1, support)]
            return prec, recall, f1, int(support)


def get_model_summary(report: str) -> (int, float):
    """Extract model summary - number of rules and avg. len."""
    pattern = " rules, avg.len. "
    for line in report.split("\n"):
        if pattern in line:
            line = line.replace("`", "")
            res = list(map(float, line.split(pattern)))
            assert len(res) == 2
            n_rules, avg_len = res
            n_rules = int(n_rules)
    return n_rules, avg_len


def main(args):
    """Generate report for each repository & calculate average results for repositories."""
    fetch()  # download executable
    # prepare output directory
    dir_to_save = args.output
    os.makedirs(dir_to_save, exist_ok=True)
    assert os.path.isdir(dir_to_save), "Output should be a directory"
    reports = []
    errors_loc = os.path.join(dir_to_save, "errors.txt")
    for repo in REPOSITORIES:
        repo, to_commit, from_commit = repo.split()
        report_loc = os.path.join(dir_to_save, get_repo_name(repo))
        quality_rep_loc = report_loc + ".quality_report.md"
        model_rep_loc = report_loc + ".model_report.md"
        # generate or read report
        try:
            if args.force or not os.path.exists(quality_rep_loc) or \
                    not os.path.exists(model_rep_loc):
                # Skip this step if report was already generated
                quality_report, model_report = measure_quality(
                    repo, to_commit=to_commit, from_commit=from_commit, bblfsh=args.bblfsh_address,
                    config=args.config
                )

                with open(quality_rep_loc, "w", encoding="utf-8") as f:
                    f.write(quality_report)
                with open(model_rep_loc, "w", encoding="utf-8") as f:
                    f.write(model_report)
            else:
                with open(quality_rep_loc, encoding="utf-8") as f:
                    quality_report = f.read()
                with open(model_rep_loc, encoding="utf-8") as f:
                    model_report = f.read()
            report = "\n".join((quality_report, model_report))
            reports.append((repo, report))
        except Exception as e:
            with open(errors_loc, "a") as f:
                f.write("-" * 20 + "\n")
                f.write(repo + "\n")
                traceback.print_exc(file=f)
            continue

    # precision, recall, f1, support, n_rules, avg_len stats
    agg = []
    table = []
    with io.StringIO() as output:
        table.append(("repo", "prec", "recall", "f1", "support", "n_rules", "avg_len"))
        for repo, report in reports:
            precision, recall, f1, support = get_precision_recall_f1_support(report)
            n_rules, avg_len = get_model_summary(report)
            agg.append((precision, recall, f1, support, n_rules, avg_len))
            table.append((get_repo_name(repo), precision, recall, f1, support, n_rules, avg_len))
        # weighted average
        table.append(("Weighted average", "%.2f" % calc_weighted_avg(agg, 0),
                      "%.2f" % calc_weighted_avg(agg, 1), "%.2f" % calc_weighted_avg(agg, 2)))
        # average
        table.append(("Average", "%.2f" % calc_avg(agg, 0), "%.2f" % calc_avg(agg, 1),
                      "%.2f" % calc_avg(agg, 2), "%.2f" % calc_avg(agg, 3),
                      "%.2f" % calc_avg(agg, 4), "%.2f" % calc_avg(agg, 5)))
        print(tabulate(table, tablefmt="pipe", headers="firstrow"), file=output)
        summary = output.getvalue()
    print(summary)
    summary_loc = os.path.join(dir_to_save, "summary.md")
    with open(summary_loc, "w", encoding="utf-8") as f:
        f.write(summary)


def create_parser():  # noqa: D103
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatterNoNone)
    parser.add_argument(
        "-o", "--output", required=True,
        help="Directory where to save results.")
    parser.add_argument(
        "-f", "--force", action="store_true",
        help="If this flag is used - force to overwrite results stored in output directory. "
             "If not - stored results will be used if they exist.")
    parser.add_argument(
        "-b", "--bblfsh-address", default="localhost:9432", help="Bblfsh address to use.")
    parser.add_argument(
        "-c", "--config", default="", help="Path to the configuration file with option defaults.")
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
