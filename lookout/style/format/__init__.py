"""Package to train on existing git repos and analyze new pull requests to correct formatting."""
from lookout.style.format.__main__ import main
from lookout.style.format.analyzer import FormatAnalyzer


analyzer_class = FormatAnalyzer
run_cmdline_tool = main
