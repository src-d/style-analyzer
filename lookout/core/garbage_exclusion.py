"""Pattern to filter out garbage files."""
from importlib import import_module
from pathlib import Path
from typing import Iterable


def _gather_patterns() -> Iterable[str]:
    """Return available patterns to filter out garbage files from the langs package."""
    for path in (Path(__file__).parent / "langs").iterdir():
        if path.is_dir():
            if (path / "garbage.py").is_file():
                yield import_module("lookout.core.langs.%s.garbage" % path.name).GARBAGE_PATTERN


GARBAGE_PATTERN = "|".join(_gather_patterns())
