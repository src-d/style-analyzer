"""Module for Smoke dataset evaluation."""
import csv
from difflib import SequenceMatcher
from itertools import chain
import json
import logging
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Dict, List, Mapping, Sequence, Tuple, Union

from bblfsh import BblfshClient
from lookout.core.analyzer import ReferencePointer
from lookout.core.api.service_analyzer_pb2 import Comment
from lookout.core.api.service_data_pb2_grpc import DataStub
from lookout.core.data_requests import with_changed_uasts_and_contents
from lookout.core.lib import files_by_language, filter_files, find_deleted_lines, find_new_lines
from lookout.core.test_helpers import server
import numpy
import pandas
from tqdm import tqdm

from lookout.style.format.analyzer import FormatAnalyzer
from lookout.style.format.code_generator import CodeGenerator
from lookout.style.format.feature_extractor import FeatureExtractor
from lookout.style.format.model import FormatModel
from lookout.style.format.tests.test_analyzer_integration import TestAnalyzer
from lookout.style.format.utils import flatten_dict
from lookout.style.format.virtual_node import VirtualNode

log = logging.getLogger("report_summary")

EMPTY = "␣"


def align2(seq1: Sequence, seq2: Sequence, seq2_ghost: Sequence = None
           ) -> Union[Tuple[Sequence, Sequence],
                      Tuple[Sequence, Sequence, Sequence]]:
    """
    Align two sequences using Levenshtein distance.

    For example:
    In[1]: align("aabc", "abbcc")
    Out[1]: ("aab␣c␣",
             "␣abbcc")

    :param seq1: First sequence to align.
    :param seq2: Second sequence to align.
    :param seq2_ghost: All changes to the second sequence are applied to seq2_ghost. \
                       Used by align3 function. Example: \
                       In[1]: align("aabbbc", "abbcc", "xxxxx") \
                       Out[1]: ("aabbbc␣", "␣abb␣cc", "␣xxx␣xx")

    :return: Aligned sequences and seq2_ghost modification if specified.
    """
    matcher = SequenceMatcher(a=seq1, b=seq2)
    res1, res2, res3 = [], [], []
    for action, i, i_end, j, j_end in matcher.get_opcodes():
        if action == "equal" or action == "replace":
            res1.append(seq1[i:i_end])
            res2.append(seq2[j:j_end])
            if seq2_ghost:
                res3.append(seq2_ghost[j:j_end])
            if i_end - i < j_end - j:
                res1.append(EMPTY * (j_end - j - i_end + i))
            elif i_end - i > j_end - j:
                empty = EMPTY * (i_end - i - j_end + j)
                res2.append(empty)
                if seq2_ghost:
                    res3.append(empty)
        if action == "insert":
            res1.append(EMPTY * (j_end - j))
            res2.append(seq2[j:j_end])
            if seq2_ghost:
                res3.append(seq2_ghost[j:j_end])
        if action == "delete":
            res1.append(seq1[i:i_end])
            empty = EMPTY * (i_end - i)
            res2.append(empty)
            if seq2_ghost:
                res3.append(empty)
    if seq2_ghost:
        return "".join(res1), "".join(res2), "".join(res3)
    return "".join(res1), "".join(res2)


def align3(seq1: Sequence, seq2: Sequence, seq3: Sequence) -> Tuple[Sequence, Sequence, Sequence]:
    """
    Align three sequences using Levenshtein distance.

    For example:
    In[1]: align("aabc", "abbcc", "ccdd")
    Out[1]: ("aab␣c␣␣␣",
             "␣abbcc␣␣",
             "␣␣␣␣ccdd")

    The result can be suboptimal because heuristic is used. True calculation requires
    ~ len(seq1) * len(seq2) * len(seq3) time.

    :param seq1: First sequence to align.
    :param seq2: Second sequence to align.
    :param seq3: Third sequence to align.
    :return: Aligned sequences.
    """
    aseq1, aseq2 = align2(seq1, seq2)
    res3, res1, res2 = align2(seq3, aseq1, aseq2)
    return res1, res2, res3


def calc_aligned_metrics(bad_style_code: str, correct_style_code: str, generated_code: str
                         ) -> Tuple[int, int, int, int]:
    """
    Calculate model quality metrics for aligned sequences.

    Metrics description:
    1. Amount of characters misdetected by the model as a style mistake. That is nothing needed to
       be changed but model did.
    2. Amount of characters undetected by model. That is the character has to be changed
       but model did not.
    3. Amount of characters detected by model as a style mistake but fix was wrong. That is
       the character has to be changed and model did but did it wrongly.
    4. Amount of characters detected by model as a style mistake and fix was correct. That is
       the character has to be changed and model did it in a correct way :tada:.

    In scientific words:
    1. False positive.
    2 + 3. False negative. We have two types of false negatives. First one is when the error was
           missed and there is no fix. Second one is when the error was found but wrongly
           fixed.
    4. True positive.

    :param bad_style_code: The file with style violations. It is files from head revision in the \
                           smoke dataset.
    :param correct_style_code: File with correct style. It is files from base revision in  the \
                               smoke dataset.
    :param generated_code: Format Analyser model output. The code with fixed style.

    :return: Tuple with 4 metric values.
    """
    detected_wrong_fix = 0
    detected_correct_fix = 0
    misdetection = 0
    undetected = 0
    for bad_style_c, correct_style_c, generated_c in zip(
            bad_style_code, correct_style_code, generated_code):
        if bad_style_c == correct_style_c == generated_c:
            continue
        assert bad_style_c in {"\t", " ", "\n", "'", '"', "␣"}
        assert correct_style_c in {"\t", " ", "\n", "'", '"', "␣"}
        assert generated_c in {"\t", " ", "\n", "'", '"', "␣"}
        if bad_style_c == correct_style_c and bad_style_c != generated_c:
            misdetection += 1
        elif bad_style_c == generated_c and bad_style_c != correct_style_c:
            undetected += 1
        elif correct_style_c == generated_c and bad_style_c != correct_style_c:
            detected_correct_fix += 1
        else:
            detected_wrong_fix += 1

    # TODO (zurk): Add proper class for benchmark metrics
    # https://github.com/src-d/style-analyzer/issues/333
    return misdetection, undetected, detected_wrong_fix, detected_correct_fix


def calc_metrics(bad_style_code: str, correct_style_code: str, fe: FeatureExtractor,
                 vnodes: Sequence[VirtualNode], y_pred: numpy.ndarray, vnodes_y: numpy.ndarray,
                 url: str, commit: str) -> Dict[str, Any]:
    """
    Calculate metrics for model output.

    Algorithm description:
    1. For a given model predictions `y_pred` we generate a new file.
       Now we have 3 files we should compare:
       1. `bad_style_code`. The file from head revision where style mistakes where applied.
          We inspect this file to find them.
       2. `correct_style_code` The file from base revision. We use this file to train repo format
          model. In the ideal case, we should be able to restore this file.
       3. `predicted_style`. The file we get as format model output.
    2. We compare files on a character level. To do so we has to align them first.
       `align3` function is used for that. There is an example:
    >>> bad_style_code = "import   abcd"
    >>> correct_style_code = "import abcd"
    >>> predicted_code = "import  abcd,"
    >>> print(align3(bad_style_code, correct_style_code, predicted_code))
    >>> Out[1]: ("import   abcd␣",
    >>>          "import ␣␣abcd␣",
    >>>          "import  ␣abcd,")
    4. Now we are able to compare sequences character by character. `calc_aligned_metrics` function
       is used for that. We can have 5 cases here. Let's consider them in the same example:
       ("import   abcd␣",  # aligned bad_style_code
        "import ␣␣abcd␣",  # aligned correct_style_code
        "import  ␣abcd,")  # aligned predicted_code
         ^      ^^    ^
         1      23    4

         1. All characters are equal. Everything is fine.
         2. Characters in bad style and predicted code are equal, but it is different in correct
            code. So, style mistake is undetected.
         3. Characters in correct style and predicted code are equal, but it is different in wrong
            file. So, style mistake is detected and correctly fixed.
         4. Characters in wrong style and correct style code are equal, but it is different in
            predicted code. So, new style mistake is introduced. We call this situation
            misdetection and want to avoid it as much as possible.
         5. All characters are different. There is no such case in the example, but this means that
            style mistake is detected but wrongly fixed.

         Thus, as output we have 4 numbers:
         1. style mistake misdetection
         2. undetected style mistake,
         3. detected style mistake with the wrong fix
         4. detected style mistake with the correct fix

         In scientific words:
         1. False positive.
         2 + 3. False negative. We have two types of false negatives. First one is when the error
                was missed and there is no fix. Second one is when the error was found but wrongly
                fixed.
         4. True positive.

    :param bad_style_code: The file from head revision where style mistakes where applied.
    :param correct_style_code: The file from base revision. In ideal case, we should be able to \
                               restore it.
    :param fe: Feature extraction class that was used to generate corresponding data.
    :param vnodes: Sequence of all the `VirtualNode`-s corresponding to the input code file. \
                   Should be ordered by position.
    :param y_pred: The model predictions for `vnodes_y` `VirtualNode`-s.
    :param vnodes_y: Sequence of the labeled `VirtualNode`-s corresponding to labeled samples. \
                     Should be ordered by start position value.
    :param url: Repository url if applicable. Useful for more informative warning messages.
    :param commit: Commit hash if applicable. Useful for more informative warning messages.

    :return: A dictionary with losses and predicted code for "global" and "local" indentation \
             strategies.
    """
    losses = {}
    for indentation in ("global", "local"):
        generator = CodeGenerator(fe, skip_errors=True, url=url, commit=commit)
        predicted_vnodes = generator.apply_predicted_y(vnodes, vnodes_y, y_pred)
        predicted_code = generator.generate(predicted_vnodes, indentation)
        misdetection, undetected, detected_wrong_fix, detected_correct_fix = \
            calc_aligned_metrics(*align3(bad_style_code, correct_style_code, predicted_code))
        losses[indentation] = {
            "misdetection": misdetection,
            "undetected": undetected,
            "detected_wrong_fix": detected_wrong_fix,
            "detected_correct_fix": detected_correct_fix,
            "predicted_file": predicted_code,
        }
    return losses


class SmokeEvalFormatAnalyzer(FormatAnalyzer):
    """
    Analyzer for Smoke dataset evaluation.
    """

    REPORT_COLNAMES = [
        "repo", "filepath", "style", "global_misdetection", "global_undetected",
        "global_detected_wrong_fix", "global_detected_correct_fix", "local_misdetection",
        "local_undetected", "local_detected_wrong_fix", "local_detected_correct_fix",
        "bad_style_file", "correct_style_file", "local_predicted_file", "global_predicted_file",
    ]

    def __init__(self, model: FormatModel, url: str, config: Mapping[str, Any]) -> None:
        """
        Construct a FormatAnalyzer.

        :param model: FormatModel to use during pull request analysis.
        :param url: Git repository on which the model was trained.
        :param config: Configuration to use to analyze pull requests.
        """
        super().__init__(model, url, config)
        self.config = self._load_analyze_config(self.config)
        self.client = BblfshClient(self.config["bblfsh_address"])

    def _dump_report(self, report: List[dict], outputpath: Path):
        files_dir = outputpath / "files"
        os.makedirs(str(files_dir), exist_ok=True)
        with open(str(outputpath / "report.csv"), "a") as f:
            writer = csv.DictWriter(f, fieldnames=self.REPORT_COLNAMES)
            for report_line in report:
                report_line = flatten_dict(report_line)
                for code_file in ["init_file", "correct_file",
                                  "global_predicted_file", "local_predicted_file"]:
                    code = report_line[code_file]
                    report_line[code_file] = "_".join((
                        report_line["repo"], report_line["style"], code_file,
                        report_line["filepath"].replace("/", "_")))
                with open(str(files_dir / report_line[code_file]), "w") as f:
                    f.write(code)
                writer.writerow(report_line)

    @with_changed_uasts_and_contents
    def analyze(self, ptr_from: ReferencePointer, ptr_to: ReferencePointer,
                data_request_stub: DataStub, **data) -> List[Comment]:
        """
        Analyze a set of changes from one revision to another.

        :param ptr_from: Git repository state pointer to the base revision.
        :param ptr_to: Git repository state pointer to the head revision.
        :param data_request_stub: Connection to the Lookout data retrieval service, not used.
        :param data: Contains "changes" - the list of changes in the pointed state.
        :return: List of comments.
        """
        # TODO (zurk): reuse code from FormatAnalyzer.analyze()
        report = []
        log = self._log
        changes = list(data["changes"])
        base_files_by_lang = files_by_language(c.base for c in changes)
        head_files_by_lang = files_by_language(c.head for c in changes)
        for lang, head_files in head_files_by_lang.items():
            if lang not in self.model:
                log.warning("skipped %d written in %s. Rules for %s do not exist in model",
                            len(head_files), lang, lang)
                continue
            rules = self.model[lang]
            for file in filter_files(head_files, rules.origin_config["line_length_limit"], log):
                log.debug("Analyze %s file", file.path)
                try:
                    base_file = base_files_by_lang[lang][file.path]
                except KeyError:
                    lines = None
                else:
                    lines = sorted(chain.from_iterable((
                        find_new_lines(base_file, file),
                        find_deleted_lines(base_file, file),
                    )))
                fe = FeatureExtractor(language=lang, **rules.origin_config["feature_extractor"])
                res = fe.extract_features([file], [lines])
                if res is None:
                    log.warning("Failed to parse %s", file.path)
                    continue
                X, y, (vnodes_y, vnodes, _, _) = res
                y_pred, _ = rules.predict(X=X, vnodes_y=vnodes_y, vnodes=vnodes,
                                          feature_extractor=fe)
                assert len(y) == len(y_pred)

                correct_style_code = base_file.content.decode("utf-8", "replace")
                bad_style_code = file.content.decode("utf-8", "replace")
                row = {
                    "repo": self.config["repo_name"],
                    "filepath": file.path,
                    "style": self.config["style_name"],
                    "bad_style_file": bad_style_code,
                    "correct_style_file": correct_style_code,
                }
                row.update(calc_metrics(
                    bad_style_code, correct_style_code,
                    fe, vnodes, vnodes_y, y_pred,
                    url=ptr_to.url, commit=ptr_to.commit
                ))
                report.append(row)
        self._dump_report(report, Path(self.config["report_path"]))
        return []


analyzer_class = SmokeEvalFormatAnalyzer


def evaluate_smoke_entry(inputpath: str, reportdir: str, database: str) -> None:
    """
    CLI entry point.
    """
    start_time = time.time()
    report_filename = os.path.join(reportdir, "report.csv")
    log = logging.getLogger("evaluate_smoke")
    port = server.find_port()
    train_config = {analyzer_class.name: {"global": {"cutoff_label_precision": 0}}}
    if database is None:
        db = tempfile.NamedTemporaryFile(dir=inputpath, prefix="db", suffix=".sqlite3")
        database = db.name
        log.info("Database %s created" % database)
    else:
        if os.path.exists(database):
            log.info("Found existing database %s" % database)
        else:
            log.info("Database %s not found and will be created." % database)
    with tempfile.TemporaryDirectory(dir=inputpath) as fs:
        context_manager = TestAnalyzer(
            port=port, db=database, fs=fs,
            analyzer="lookout.style.format.benchmarks.evaluate_smoke")
        with context_manager:
            inputpath = Path(inputpath)
            if not server.exefile.exists():
                server.fetch()
            index_file = inputpath / "index.csv"
            os.makedirs(reportdir, exist_ok=True)
            with open(report_filename, "w") as report:
                csv.DictWriter(report, fieldnames=SmokeEvalFormatAnalyzer.REPORT_COLNAMES
                               ).writeheader()
            with open(str(index_file)) as index:
                reader = csv.DictReader(index)
                for row in tqdm(reader):
                    repopath = inputpath / row["repo"]
                    config_json = {
                        analyzer_class.name: {
                            "repo_name": row["repo"],
                            "style_name": row["style"],
                            "report_path": reportdir
                        }
                    }
                    server.run("push", fr=row["from"], to=row["to"], port=port,
                               git_dir=str(repopath), config_json=json.dumps(train_config))
                    server.run("review", fr=row["from"], to=row["to"], port=port,
                               git_dir=str(repopath),
                               config_json=json.dumps(config_json))
            log.info("Quality report saved to %s", reportdir)

    report = pandas.read_csv(report_filename)
    with pandas.option_context("display.max_columns", 10, "display.expand_frame_repr", False):
        print(report.describe())
    log.info("Time spent: %.3f" % (time.time() - start_time))
