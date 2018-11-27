"""Package to train on existing git repos and analyze new pull requests to correct formatting."""
from lookout.style.format.analyzer import FormatAnalyzer
from lookout.style.format.cmdline_tools import main


analyzer_class = FormatAnalyzer
run_cmdline_tool = main
