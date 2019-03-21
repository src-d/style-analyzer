import functools
import os
import unittest

from lookout.style.format import descriptions, FormatAnalyzer
from lookout.style.format.analyzer import FileFix, LineFix
from lookout.style.format.virtual_node import Position, VirtualNode


class CodeGeneratorTests(unittest.TestCase):
    def test_template(self):
        class FakeRules:
            rules = {34: "<rule # 34>"}

        class FakeModel:
            def __getitem__(self, item):
                return FakeRules()

        class FakeHeadFile:
            content = "<first code line>\n<second code line>\n<third code line>"

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
            "analyze": {
                "language_defaults": {
                    "report_code_lines": True,
                    "report_triggered_rules": True,
                    "comment_template": comment_template_flie,
                },
            },
        }
        analyzer = FormatAnalyzer(config=config, model=FakeModel(), url="http://github.com/x/y")
        language = "javascript"
        line_number = 2
        suggested_code = "<new code line>"
        partial_backup = functools.partial
        vnode = VirtualNode(start=Position(10, 2, 1), end=Position(12, 3, 1), value="!",
                            y=(1,))
        vnode.applied_rule = FakeRules.rules[34]
        line_fix = LineFix(
            line_number=line_number, suggested_code=suggested_code,
            fixed_vnodes=[vnode], confidence=100)
        file_fix = FileFix(error="", line_fixes=[line_fix], language=language, base_file=None,
                           feature_extractor=None, file_vnodes=[], head_file=FakeHeadFile,
                           y=None, y_pred_pure=None)

        try:
            functools.partial = fake_partitial
            text = analyzer.render_comment_text(file_fix, 0)
            res = """format: style mismatch:
```javascript
1|<first code line>
2|<second code line>
3|<third code line>
```
```suggestion
<new code line>
```

`<change description>` <change description>
Triggered rule
```
<rule # 34>
```
"""
            self.assertEqual(text, res)
        finally:
            functools.partial = partial_backup


if __name__ == "__main__":
    unittest.main()
