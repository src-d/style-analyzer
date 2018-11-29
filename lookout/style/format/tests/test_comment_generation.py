import functools
import os
import unittest

from lookout.style.format import descriptions, FormatAnalyzer
from lookout.style.format.virtual_node import Position, VirtualNode


class CodeGeneratorTests(unittest.TestCase):
    def test_template(self):
        class FakeRules():
            rules = {34: "<rule # 34>"}

        class FakeModel:
            def __getitem__(self, item):
                return FakeRules()

        def fake_partitial(func, *_, **__):
            if func == descriptions.describe_rule:
                def fake_describe_rule(rule, *_, **__):
                    return rule
                return fake_describe_rule

            def fake_get_change_description(*_, **__):
                return "<change description>"
            return fake_get_change_description
        comment_template_flie = os.path.join(os.path.dirname(__file__), "..", "templates",
                                             "comment_default.jinja2")
        config = {
            "report_code_lines": True,
            "report_triggered_rules": True,
            "comment_template": comment_template_flie,
        }
        analyzer = FormatAnalyzer(config=config, model=FakeModel(), url="http://github.com/x/y")
        language = "<language>"
        line_number = 2
        code_lines = ["<first code line>", "<second code line>", "<third code line>"]
        new_code_line = "<new code line>"
        partial_backup = functools.partial
        try:

            functools.partial = fake_partitial
            text = analyzer.render_comment_text(
                language=language,  # programming language of the code
                line_number=line_number,  # line number for the comment
                code_lines=code_lines,  # original file code lines
                new_code_line=new_code_line,  # code line suggested by our model
                winners=[34],
                vnodes=[VirtualNode(start=Position(10, 2, 1), end=Position(12, 3, 1),
                                    value="!", y=1)],
                fixed_labels=[2],
                confidence=99,
                feature_extractor=None,
            )
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
