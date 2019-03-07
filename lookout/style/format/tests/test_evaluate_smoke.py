import os
import tempfile
import unittest

from modelforge import slogging
import pandas

from lookout.style.format.benchmarks import generate_smoke
from lookout.style.format.benchmarks.evaluate_smoke import (
    align2, align3, calc_aligned_metrics,
    evaluate_smoke_entry)
from lookout.style.format.tests.test_analyzer import get_config


class EvaluateSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        slogging.setup("DEBUG", False)
        cls.seqs3 = [
            ("", "", ""),
            ("abc", "", ""),
            ("abc", "abc", ""),
            ("ab", "abc", "ab"),
            ("ab", "abc", "abc"),
            ("abd", "abc", ""),
            ("abd", "abc", "abg"),
            ("abd", "abc", "ab"),
            ("abcd", "abc", ""),
            ("abcd", "abc", "abcd"),
            ("abcd", "abc", "axbcd"),
            ("aabb", "bbcc", "ccdd"),  # case with nonoptimal alignment
            ("bbcc", "ccdd", "aabb"),  # case with nonoptimal alignment
            ("abbc", "aabcc", "abc"),
            ("abcabcbabac", "abcbabababc", "abcbacabca"),
        ]
        cls.seqs3_answers = [
            (("", "", ""), ("", "", ""), ("", "", "")),
            (("abc", "␣␣␣", "␣␣␣"), ("␣␣␣", "␣␣␣", "abc"), ("␣␣␣", "abc", "␣␣␣")),
            (("abc", "abc", "␣␣␣"), ("abc", "␣␣␣", "abc"), ("␣␣␣", "abc", "abc")),
            (("ab␣", "abc", "ab␣"), ("abc", "ab␣", "ab␣"), ("ab␣", "ab␣", "abc")),
            (("ab␣", "abc", "abc"), ("abc", "abc", "ab␣"), ("abc", "ab␣", "abc")),
            (("abd", "abc", "␣␣␣"), ("abc", "␣␣␣", "abd"), ("␣␣␣", "abd", "abc")),
            (("abd", "abc", "abg"), ("abc", "abg", "abd"), ("abg", "abd", "abc")),
            (("abd", "abc", "ab␣"), ("abc", "ab␣", "abd"), ("ab␣", "abd", "abc")),
            (("abcd", "abc␣", "␣␣␣␣"), ("abc␣", "␣␣␣␣", "abcd"), ("␣␣␣␣", "abcd", "abc␣")),
            (("abcd", "abc␣", "abcd"), ("abc␣", "abcd", "abcd"), ("abcd", "abcd", "abc␣")),
            (("a␣bcd", "a␣bc␣", "axbcd"), ("a␣bc␣", "axbcd", "a␣bcd"),
             ("axbcd", "a␣bcd", "a␣bc␣")),
            (("aabb␣␣", "␣␣bbcc", "ccdd␣␣"), ("␣␣bbcc␣␣", "␣␣␣␣ccdd", "aabb␣␣␣␣"),
             ("␣␣ccdd", "␣␣aabb", "bbcc␣␣")),
            (("␣␣bbcc␣␣", "␣␣␣␣ccdd", "aabb␣␣␣␣"), ("␣␣ccdd", "␣␣aabb", "bbcc␣␣"),
             ("aabb␣␣", "␣␣bbcc", "ccdd␣␣")),
            (("␣abbc␣", "aab␣cc", "␣ab␣c␣"), ("aab␣cc", "␣ab␣c␣", "␣abbc␣"),
             ("␣ab␣c␣", "␣abbc␣", "aab␣cc")),
            (("abcabcbaba␣␣␣c␣␣␣␣", "␣␣␣abcbabababc␣␣␣␣", "␣␣␣abcba␣␣␣␣␣cabca"),
             ("␣␣␣abcbabababc␣", "␣␣␣abcbac␣␣abca", "abcabcbaba␣␣␣c␣"),
             ("␣␣␣abcba␣␣cabca", "abcabcbabac␣␣␣␣", "␣␣␣abcbabababc␣")),
        ]

    def test_align2(self):
        seqs = [
            ("", ""),
            ("abc", ""),
            ("abc", "abc"),
            ("ab", "abc"),
            ("abd", "abc"),
            ("abcd", "abc"),
            ("aabb", "bbcc"),
            ("abbc", "aabcc"),
            ("abcabcbabac", "abcbabababc"),
        ]
        answers = [
            ("", ""),
            ("abc", "␣␣␣"),
            ("abc", "abc"),
            ("ab␣", "abc"),
            ("abd", "abc"),
            ("abcd", "abc␣"),
            ("aabb␣␣", "␣␣bbcc"),
            ("␣abbc␣", "aab␣cc"),
            ("abcabcbaba␣␣␣c", "␣␣␣abcbabababc"),
        ]

        for seq, answer in zip(seqs, answers):
            self.assertEqual(align2(seq[0], seq[1]), answer)
            self.assertEqual(align2(seq[1], seq[0]), answer[::-1])

    def test_align3(self):
        for seq, answer in zip(self.seqs3, self.seqs3_answers):
            self.assertEqual(align3(seq[0], seq[1], seq[2]), answer[0])
            self.assertEqual(align3(seq[1], seq[2], seq[0]), answer[1])
            self.assertEqual(align3(seq[2], seq[0], seq[1]), answer[2])

    def test_calc_aligned_metrics(self):
        # misdetection, undetected, detected_wrong_fix, detected_correct_fix
        answers = [
            (0, 0, 0, 0),
            (0, 0, 0, 3),
            (3, 0, 0, 0),
            (0, 1, 0, 0),
            (0, 0, 0, 1),
            (2, 0, 1, 0),
            (0, 0, 1, 0),
            (0, 0, 1, 0),
            (3, 0, 0, 1),
            (0, 1, 0, 0),
            (1, 1, 0, 0),
            (2, 2, 2, 0),
            (4, 4, 0, 0),
            (0, 2, 0, 1),
            (6, 3, 0, 3),
        ]

        for aligned_seqs, answer in zip(self.seqs3_answers, answers):
            self.assertEqual(calc_aligned_metrics(*aligned_seqs[0]), answer)

    def test_evaluate_smoke_entry_integration(self):
        input_path = os.path.join(os.path.dirname(generate_smoke.__file__), "data",
                                  "js_smoke_init.tar.xz")
        generate_smoke.js_format_rules = {"equal_no_space_style": (" = ", "=")}
        with tempfile.TemporaryDirectory(prefix="test-smoke-eval-") as outputpath:
            generate_smoke.generate_smoke_entry(input_path, outputpath, force=True)
            with open(os.path.join(outputpath, "index.csv")) as index:
                index_content = index.read().splitlines()
            self.assertEqual(len(index_content), 5)
            self.assertEqual(set(os.listdir(outputpath)),
                             {"index.csv", "nodejs", "jsquery", "freeCodeCamp", "react"})
            with open(os.path.join(outputpath, "index.csv"), "w") as index:
                index.write("\n".join(index_content[::3]))
            report_dir = os.path.join(outputpath, "report")
            evaluate_smoke_entry(outputpath, report_dir, None, "0.0.0.0:9432", get_config())
            report = pandas.read_csv(os.path.join(report_dir, "report.csv"))
            self.assertEqual(len(report), 4)
            self.assertEqual(len(report.columns), 10)


if __name__ == "__main__":
    unittest.main()
