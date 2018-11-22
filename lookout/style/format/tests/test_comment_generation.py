import os
import unittest

from jinja2 import Template


class CodeGeneratorTests(unittest.TestCase):
    def test_template(self):
        comment_template_flie = os.path.join(os.path.dirname(__file__), "../templates",
                                             "comment_default.jinja2")
        with open(comment_template_flie) as f:
            comment_template = Template(f.read(), trim_blocks=True, lstrip_blocks=True)
        config = {
            "report_code_lines": True,
            "report_triggered_rules": True}
        lang = "<language>"
        line_number = 2
        code_lines = ["<first code line>", "<second code line>", "<third code line>"]
        new_code_line = "<new code line>"
        change_descriptions = [(34, "<change description here>", "<rule description here>")]

        text = comment_template.render(
            config=config,  # configuration of the analyzer
            lang=lang,  # programming language of the code
            line_number=line_number,  # line number for the comment
            code_lines=code_lines,  # original file code lines
            new_code_line=new_code_line,  # code line suggested by our model
            change_descriptions=change_descriptions,  # change descriptions proposed by model
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

        config = {
            "report_code_lines": False,
            "report_triggered_rules": False}
        change_descriptions = [(34, "<change description here>", "<rule description here>"),
                               (35, "<change2 description here>", "<rule2 description here>")]
        text = comment_template.render(
            config=config,  # configuration of the analyzer
            lang=lang,  # programming language of the code
            line_number=line_number,  # line number for the comment
            code_lines=code_lines,  # original file code lines
            new_code_line=new_code_line,  # code line suggested by our model
            change_descriptions=change_descriptions,  # change descriptions proposed by model
        )
        res = """format: style mismatch:
```suggestion
<new code line>
```

<change description here>
<change2 description here>
"""
        self.assertEqual(text, res)


if __name__ == "__main__":
    unittest.main()
