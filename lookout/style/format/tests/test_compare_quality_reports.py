import tempfile
import unittest

from lookout.style.format.benchmarks.compare_quality_reports import compare_quality_reports_entry

report1 = """
| repo               |   precision |   recall |   full_recall |    f1 |   full_f1 |   ppcr |   support |   full_support |   Rules Number |   Average Rule Len |
|:-------------------|------------:|---------:|--------------:|------:|----------:|-------:|----------:|---------------:|---------------:|-------------------:|
| telescope          |       1.000 |    1.000 |         0.166 | 1.000 |     0.284 |  0.166 |       121 |            731 |              2 |                2.0 |
| weighted average   |       0.971 |    0.971 |         0.909 | 0.971 |     0.938 |  0.939 |           |                |                |                    |
""".strip()  # noqa E501

report2 = """
| repo               |   precision |   recall |   full_recall |    f1 |   full_f1 |   ppcr |   support |   full_support |   Rules Number |   Average Rule Len |
|:-------------------|------------:|---------:|--------------:|------:|----------:|-------:|----------:|---------------:|---------------:|-------------------:|
| telescope          |       1.000 |    1.000 |         0.166 | 1.000 |     0.284 |  0.166 |       121 |            731 |              2 |                2.0 |
| weighted average   |       0.971 |    0.971 |         0.909 | 0.971 |     0.938 |  0.939 |           |                |                |                    |
""".strip()  # noqa E501

diff_report = """
|             repo |      precision |         recall |    full_recall |             f1 |        full_f1 |           ppcr |       support |   full_support |   Rules Number |   Average Rule Len |
|-----------------:|---------------:|---------------:|---------------:|---------------:|---------------:|---------------:|--------------:|---------------:|---------------:|-------------------:|
|        telescope | 1.000 (+0.000) | 1.000 (+0.000) | 0.166 (+0.000) | 1.000 (+0.000) | 0.284 (+0.000) | 0.166 (+0.000) | 121 (     +0) |  731 (     +0) |       2 (  +0) |         2.0 (+0.0) |
| weighted average | 0.971 (+0.000) | 0.971 (+0.000) | 0.909 (+0.000) | 0.971 (+0.000) | 0.938 (+0.000) | 0.939 (+0.000) |               |                |                |                    |
""".strip()  # noqa E501


class EvaluateSmokeTests(unittest.TestCase):
    def test_compare_quality_reports_entry(self):
        self.maxDiff = None
        with tempfile.NamedTemporaryFile() as f_res:
            with tempfile.NamedTemporaryFile() as f1:
                with tempfile.NamedTemporaryFile() as f2:
                    with open(f1.name, "w") as f:
                        f.write(report1)
                    with open(f2.name, "w") as f:
                        f.write(report2)
                    compare_quality_reports_entry(f1.name, f2.name, f_res.name)
            with open(f_res.name) as f:
                res = f.read()
        self.assertEqual("\n".join(res.splitlines()[-4:]), diff_report)


if __name__ == "__main__":
    unittest.main()
