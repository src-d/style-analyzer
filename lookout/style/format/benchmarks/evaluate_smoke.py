"""Module for Smoke dataset evaluation."""
import csv
from difflib import SequenceMatcher
import logging
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Dict, List, Sequence, Tuple, Union

from lookout.core.analyzer import ReferencePointer
from lookout.core.api.service_analyzer_pb2 import Comment
from lookout.core.data_requests import DataService, with_changed_uasts_and_contents
from lookout.core.helpers.analyzer_context_manager import AnalyzerContextManager
import pandas
from tqdm import tqdm

from lookout.style.common import merge_dicts
from lookout.style.format.analyzer import FormatAnalyzer
from lookout.style.format.code_generator import CodeGenerator
from lookout.style.format.feature_extractor import FeatureExtractor
from lookout.style.format.virtual_node import VirtualNode

EMPTY = "␣"


def align2(seq1: Sequence, seq2: Sequence, seq2_ghost: Sequence = None,
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
            if seq2_ghost is not None:
                res3.append(empty)
    if seq2_ghost is not None:
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


def calc_aligned_metrics(bad_style_code: str, correct_style_code: str, generated_code: str,
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
                 vnodes: Sequence[VirtualNode], url: str, commit: str) -> Dict[str, Any]:
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
    :param fe: Feature extraction class that was used to generate corresponding data. Set a value \
            to None if no changes were introduced for `bad_style_code`.
    :param vnodes: Sequence of all the `VirtualNode`-s corresponding to the input code file. \
                   Should be ordered by position. New y values should be applied.
    :param url: Repository url if applicable. Useful for more informative warning messages.
    :param commit: Commit hash if applicable. Useful for more informative warning messages.

    :return: A dictionary with losses and predicted code.
    """
    predicted_code = CodeGenerator(fe, skip_errors=True, url=url, commit=commit).generate(vnodes)
    misdetection, undetected, detected_wrong_fix, detected_correct_fix = \
        calc_aligned_metrics(*align3(bad_style_code, correct_style_code, predicted_code))
    losses = {
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
        "repo", "filepath", "style", "misdetection",
        "undetected", "detected_wrong_fix", "detected_correct_fix",
        "bad_style_file", "correct_style_file", "predicted_file",
    ]

    def _dump_report(self, report: List[dict], outputpath: Path):
        files_dir = outputpath / "files"
        os.makedirs(str(files_dir), exist_ok=True)
        with open(str(outputpath / "report.csv"), "a") as f:
            writer = csv.DictWriter(f, fieldnames=self.REPORT_COLNAMES)
            for report_line in report:
                for code_file in ["bad_style_file", "correct_style_file", "predicted_file"]:
                    code = report_line[code_file]
                    report_line[code_file] = "_".join((
                        report_line["repo"], report_line["style"], code_file,
                        report_line["filepath"].replace("/", "_")))
                with open(str(files_dir / report_line[code_file]), "w") as f:
                    f.write(code)
                writer.writerow(report_line)

    @with_changed_uasts_and_contents(unicode=True)
    def analyze(self, ptr_from: ReferencePointer, ptr_to: ReferencePointer,
                data_service: DataService, **data) -> List[Comment]:
        """
        Analyze a set of changes from one revision to another.

        :param ptr_from: Git repository state pointer to the base revision.
        :param ptr_to: Git repository state pointer to the head revision.
        :param data_service: Connection to the Lookout data retrieval service.
        :param data: Contains "changes" - the list of changes in the pointed state.
        :return: List of comments.
        """
        report = []
        changes = list(data["changes"])
        for file_fix in self.generate_file_fixes(data_service, changes):
            filepath = file_fix.head_file.path
            if file_fix.error:
                continue
            bad_style_code = file_fix.head_file.content
            correct_style_code = file_fix.base_file.content
            row = {
                "repo": self.model.ptr.url.split("/")[-1],
                "style": self.config["style_name"],
                "filepath": filepath,
                "bad_style_file": bad_style_code,
                "correct_style_file": correct_style_code,
            }
            row.update(calc_metrics(
                bad_style_code, correct_style_code, file_fix.feature_extractor,
                file_fix.file_vnodes, url=ptr_to.url, commit=ptr_to.commit))
            report.append(row)
        self._dump_report(report, Path(self.config["report_path"]))
        return []


def evaluate_smoke_entry(inputpath: str, reportdir: str, database: str, bblfsh: str, config: dict,
                         ) -> None:
    """
    CLI entry point.
    """
    start_time = time.time()
    report_filename = os.path.join(reportdir, "report.csv")
    log = logging.getLogger("evaluate_smoke")
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
        with AnalyzerContextManager(SmokeEvalFormatAnalyzer, db=database, fs=fs) as server:
            inputpath = Path(inputpath)
            index_file = inputpath / "index.csv"
            os.makedirs(reportdir, exist_ok=True)
            with open(report_filename, "w") as report:
                csv.DictWriter(report, fieldnames=SmokeEvalFormatAnalyzer.REPORT_COLNAMES,
                               ).writeheader()
            with open(str(index_file)) as index:
                reader = csv.DictReader(index)
                for row in tqdm(reader):
                    repopath = inputpath / row["repo"]
                    config_json = {
                        SmokeEvalFormatAnalyzer.name:
                            merge_dicts(config, {
                                "style_name": row["style"],
                                "report_path": reportdir,
                            })}
                    server.review(fr=row["from"], to=row["to"], git_dir=str(repopath),
                                  log_level="warning", bblfsh=bblfsh, config_json=config_json)
            log.info("Quality report saved to %s", reportdir)

    report = pandas.read_csv(report_filename)
    with pandas.option_context("display.max_columns", 10, "display.expand_frame_repr", False):
        print(report.describe())
    log.info("Time spent: %.3f" % (time.time() - start_time))
