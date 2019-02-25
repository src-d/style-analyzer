import argparse
import sys
import unittest
from unittest.mock import patch

import lookout.style.format.cmdline_tools as main


class MainTests(unittest.TestCase):
    def test_handlers(self):
        raw_imports = """
        from lookout.style.format.benchmarks.compare_quality_reports import compare_quality_reports_entry
        from lookout.style.format.benchmarks.evaluate_smoke import evaluate_smoke_entry
        from lookout.style.format.benchmarks.generate_smoke import generate_smoke_entry
        from lookout.style.format.benchmarks.general_report import print_reports
        from lookout.style.format.benchmarks.quality_report_noisy import quality_report_noisy
        from lookout.style.format.cmdline_tools import dump_rule_entry
        from lookout.style.format.benchmarks.expected_vnodes_number import calc_expected_vnodes_number_entry
        from lookout.style.format.benchmarks.quality_report import generate_quality_report
        """  # noqa E501
        imports = {}
        for line in raw_imports.splitlines():
            line = line.strip()
            if line:
                parts = line.split(" ")
                imports[parts[-1]] = parts[1]
        action2handler = {
            "eval": "print_reports",
            "quality-report": "generate_quality_report",
            "quality-report-noisy": "quality_report_noisy",
            "gen-smoke-dataset": "generate_smoke_entry",
            "eval-smoke-dataset": "evaluate_smoke_entry",
            "rule": "dump_rule_entry",
            "calc-expected-vnodes-number": "calc_expected_vnodes_number_entry",
            "compare-quality": "compare_quality_reports_entry",
        }
        parser = main.create_parser()
        subcommands = set([x.dest for x in parser._subparsers._actions[4]._choices_actions])
        set_action2handler = set(action2handler)
        self.assertFalse(len(subcommands - set_action2handler),
                         "You forgot to add to this test {} subcommand(s) check".format(
                             subcommands - set_action2handler))

        self.assertFalse(len(set_action2handler - subcommands),
                         "You cover unexpected subcommand(s) {}".format(
                             set_action2handler - subcommands))

        called_actions = []
        args_save = sys.argv
        error_save = argparse.ArgumentParser.error
        try:
            argparse.ArgumentParser.error = lambda self, message: None

            for action, handler in action2handler.items():
                def handler_append(*args, **kwargs):
                    called_actions.append(action)
                with patch(imports[handler] + "." + handler, handler_append):
                    sys.argv = [main.__file__, action]
                    main.main()
        finally:
            sys.argv = args_save
            argparse.ArgumentParser.error = error_save

        set_called_actions = set(called_actions)
        set_actions = set(action2handler)
        self.assertEqual(set_called_actions, set_actions)
        self.assertEqual(len(set_called_actions), len(called_actions))


if __name__ == "__main__":
    unittest.main()
