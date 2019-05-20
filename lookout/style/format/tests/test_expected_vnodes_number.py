import os
import sys
from tempfile import NamedTemporaryFile
import unittest

from modelforge import slogging

from lookout.style.format.benchmarks.expected_vnodes_number import \
    calc_expected_vnodes_number_entry


class ExpectedVnodesTest(unittest.TestCase):
    def setUp(self):
        slogging.setup("INFO", False)

    @unittest.skipIf(sys.version_info.minor == 6, "We test python 3.6 inside docker container, "
                                                  "so there is no another docker inside.")
    def test_calc_expected_vnodes_number_entry(self):
        quality_report_repos_filepath = os.path.abspath(
            os.path.join(os.path.split(__file__)[0],
                         "../benchmarks/data/quality_report_repos.csv"))
        with NamedTemporaryFile(prefix="output-expected-vnodes-", suffix=".csv") as f_result:
            with NamedTemporaryFile(prefix="expected-vnodes-", suffix=".csv") as f_new:
                with open(quality_report_repos_filepath) as f_orig:
                    with open(f_new.name, "w") as f_new_file:
                        for i, line in enumerate(f_orig):
                            f_new_file.write(line)
                            if i > 0:
                                break

                calc_expected_vnodes_number_entry(f_new.name, f_result.name, 2)
                with open(f_result.name) as f_result_file:
                    result = f_result_file.read()

        self.assertEqual(result.strip(),
                         """url,to,from,vnodes_number
https://github.com/laravel/telescope,534030114f47696fe3f3b08ea7ca49467428f2af,6f0a10ec586cfa1a22218b6778bf9c1572b97912,3041
""".strip())


if __name__ == "__main__":
    unittest.main()
