"""Typos corrector. Uses symspell, FastText over a dataset of identifiers, etc."""
from lookout.style.typos.analyzer import IdTyposAnalyzer
from lookout.style.typos.cmdline_tools import main


analyzer_class = IdTyposAnalyzer
run_cmdline_tool = main
