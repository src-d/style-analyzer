import os
import unittest

from lookout.style.format import FormatAnalyzer
from lookout.style.format.virtual_node import VirtualNode, Position


class CodeGeneratorTests(unittest.TestCase):
    def test_template(self):
        comment_template_flie = os.path.join(os.path.dirname(__file__), "..", "templates",
                                             "comment_default.jinja2")
        config = {
            "report_code_lines": True,
            "report_triggered_rules": True,
            "comment_template": comment_template_flie,
        }
        analyzer = FormatAnalyzer(config=config, model=None, url="http://github.com/x/y")

        lang = "<language>"
        line_number = 2
        code_lines = ["<first code line>", "<second code line>", "<third code line>"]
        new_code_line = "<new code line>"

        text = analyzer.render_comment_text(
            language=lang,  # programming language of the code
            line_number=line_number,  # line number for the comment
            code_lines=code_lines,  # original file code lines
            new_code_line=new_code_line,  # code line suggested by our model
            winners=[34],
            vnodes=[VirtualNode(start=Position(10, 2, 1), end=Position(12, 3, 1), value="!", y=1)],
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

<change description here>
Triggered rule # 34
```
<rule description here>
```
"""
        self.assertEqual(text, res)


if __name__ == "__main__":
    unittest.main()
