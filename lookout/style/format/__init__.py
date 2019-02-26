"""Package to train on existing git repos and analyze new pull requests to correct formatting."""
import os

from lookout.style.format.analyzer import FormatAnalyzer
from lookout.style.format.cmdline_tools import main


TEMPLATES_ROOT = os.path.join(os.path.dirname(__file__), "templates")
analyzer_class = FormatAnalyzer
run_cmdline_tool = main
