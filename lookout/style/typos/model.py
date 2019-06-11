"""Identifier typos analyzer."""
from typing import Set

from lookout.core.analyzer import AnalyzerModel
from modelforge import merge_strings, split_strings
from sourced.ml.core.models.license import DEFAULT_LICENSE


class IdTyposModel(AnalyzerModel):
    """
    A model to store approved identifiers from the repo.
    """

    LICENSE = DEFAULT_LICENSE

    def __init__(self, **kwargs):
        """Construct a FormatModel."""
        super().__init__(**kwargs)
        self._identifiers = set()

    @property
    def identifiers(self) -> Set[str]:
        """Get stored identifiers."""
        return self._identifiers

    def dump(self) -> str:
        """Serialize this model and return the result as a string."""
        return super().dump() + \
            "\nNumber of known and approved identifiers is %s" % len(self._identifiers)

    def _generate_tree(self) -> dict:
        tree = super()._generate_tree()
        tree.update(identifiers=merge_strings(sorted(self._identifiers)))
        return tree

    def _load_tree(self, tree: dict) -> None:
        super()._load_tree(tree)
        self._identifiers = set(split_strings(tree["identifiers"]))
