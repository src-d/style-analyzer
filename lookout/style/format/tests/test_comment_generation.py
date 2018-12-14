import functools
import os
import unittest

from lookout.style.format import descriptions, FormatAnalyzer
from lookout.style.format.analyzer import FixData
from lookout.style.format.virtual_node import Position, VirtualNode


class CodeGeneratorTests(unittest.TestCase):
    def test_template(self):
        class FakeRules:
            rules = {34: "<rule # 34>"}

        class FakeModel:
            def __getitem__(self, item):
                return FakeRules()

        class FakeHeadFile:
            content = b"<first code line>\n<second code line>\n<third code line>"

        def fake_partitial(func, *_, **__):
            if func == descriptions.describe_rule:
                def fake_describe_rule(rule, *_, **__):
                    return rule
                return fake_describe_rule

            def fake_get_change_description(*_, **__):
                return "<change description>"
            return fake_get_change_description
        comment_template_flie = os.path.join(os.path.dirname(__file__), "..", "templates",
                                             "comment.jinja2")
        config = {
            "report_code_lines": True,
            "report_triggered_rules": True,
            "comment_template": comment_template_flie,
        }
        analyzer = FormatAnalyzer(config=config, model=FakeModel(), url="http://github.com/x/y")
        language = "<language>"
        line_number = 2
        suggested_code = "<new code line>"
        partial_backup = functools.partial
        fix_data = FixData(
            base_file=None, head_file=FakeHeadFile, confidence=100, line_number=line_number,
            error="Failed to parse", language=language, feature_extractor=None,
            winner_rules=[34], suggested_code=suggested_code, all_vnodes=[],
            fixed_vnodes=[VirtualNode(start=Position(10, 2, 1), end=Position(12, 3, 1),
                                      value="!", y=(1,))])
        try:

            functools.partial = fake_partitial
            text = analyzer.render_comment_text(fix_data)
            res = """format: style mismatch:
```<language>
1|<first code line>
2|<second code line>
3|<third code line>
```
```suggestion
<new code line>
```

<change description>
Triggered rule # 34
```
<rule # 34>
```
"""
            self.assertEqual(text, res)
        finally:
            functools.partial = partial_backup


if __name__ == "__main__":
    unittest.main()
