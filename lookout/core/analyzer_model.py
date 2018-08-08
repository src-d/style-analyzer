from typing import Type

from modelforge import Model

from lookout.core.analyzer import Analyzer


class AnalyzerModel(Model):
    """
    All models used in `Analyzer`-s must derive from this base class.
    """
    def __init__(self, **kwargs):
        """
        Defines:
        `name` - name of the model. Corresponds to the bound analyzer's class name and version.
        `url` - Git repository on which the model was trained.
        `commit` - revision of the Git repository on which the model was trained.
        :param kwargs: passed to the upstream __init__.
        """
        super().__init__(**kwargs)
        self.name = "<unknown name>"
        self.url = "<unknown url>"
        self.commit = "<unknown commit>"

    def construct(self, analyzer: Type[Analyzer], url: str, commit: str):
        """
        Initializing method.

        :param analyzer: Bound type of the `Analyzer`. Not instance!
        :param url: Git repository on which the model was trained.
        :param commit: revision of the Git repository on which the model was trained.
        :return: self
        """
        assert isinstance(self, analyzer.model_type)
        self.name = analyzer.__name__
        self.version = [analyzer.version]
        self.url = url
        self.commit = commit
        return self

    def dump(self) -> str:
        """
        Implements the upstream abstract method.

        :return: summary text of the model.
        """
        return "%s/%s %s %s" % (self.name, self.version, self.url, self.commit)
