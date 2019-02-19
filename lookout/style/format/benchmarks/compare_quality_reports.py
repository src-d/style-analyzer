import sys
from typing import TextIO, Union

import pandas
from tabulate import tabulate

from lookout.style.format.benchmarks.quality_report import FLOAT_PRECISION


_column_formats = {
    "precision": FLOAT_PRECISION,
    "recall": FLOAT_PRECISION,
    "full_recall": FLOAT_PRECISION,
    "f1": FLOAT_PRECISION,
    "full_f1": FLOAT_PRECISION,
    "ppcr": FLOAT_PRECISION,
    "support": "7g",
    "full_support": "7g",
    "Rules Number": "4g",
    "Average Rule Len": ".1f",
}


def _convert_cell_to_value(cell: str) -> Union[int, float, str]:
    for type_ in (int, float):
        try:
            return type_(cell.strip())
        except ValueError:
            pass
    cell = cell.strip()
    return cell if cell else float("nan")


def _quality_report_table_to_df(report_table_file: TextIO) -> pandas.DataFrame:
    table = []
    separator_set = frozenset(":|-")
    for line in report_table_file:
        if set(line.strip()) <= separator_set:
            continue
        cells = line.strip().split("|")
        if not cells[0]:
            cells = cells[1:]
        if not cells[-1]:
            cells = cells[:-1]
        values = [_convert_cell_to_value(c) for c in cells]
        table.append(values)
    return pandas.DataFrame(table[1:], columns=table[0])


def compare_quality_reports(base: TextIO, new: TextIO, output: TextIO) -> None:
    """
    Print a table with metric difference between the reports.

    :param base: Baseline report file. Usually the latest report from ./report/ directory.
    :param new: New report file. Usually It is a report generated for master or any local \
                change you did and want to validate.
    :param output: The result will be saved to this file.
    """
    base_report = _quality_report_table_to_df(base)
    new_report = _quality_report_table_to_df(new)
    base_report.set_index("repo", inplace=True)
    new_report.set_index("repo", inplace=True)
    delta = new_report - base_report
    delta = delta.reindex(index=list(base_report.index))
    for field in delta.columns:
        delta[field] = new_report[field].map(lambda x: ("%%%s" % _column_formats[field] % x)) + \
                       delta[field].map(lambda x: " (%%+%s)" % _column_formats[field] % x)

    delta = delta.fillna("")
    delta.loc[["weighted average"],
              ["support", "full_support", "Rules Number", "Average Rule Len"]] = ""

    res = "\n# Report comparison\n%s\nvs\n%s\n\n" % (new, base) \
          + tabulate(delta, tablefmt="pipe", headers="keys", stralign="right")
    print(res, file=output)


def compare_quality_reports_entry(base: str, new: str, output: str) -> None:
    """
    Print a table with metric difference between the reports.

    Command line entry point for compare_quality_reports().

    :param base: Baseline report file path. Usually the latest report from ./report/ directory.
    :param new: New report file path. Usually It is a report generated for master or any local \
                change you did and want to validate.
    :param output: The result will be saved to this file or print to stdout if you set `-`.
    """
    with open(base) as base_file, open(new) as new_file:
        if output == "-":
            compare_quality_reports(base_file, new_file, sys.stdout)
        else:
            with open(output, "w") as output_file:
                compare_quality_reports(base_file, new_file, output_file)
