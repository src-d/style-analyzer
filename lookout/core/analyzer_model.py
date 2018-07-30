from typing import Type

from modelforge import Model

from lookout.core.analyzer import Analyzer


class AnalyzerModel(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "<unknown name>"
        self.url = "<unknown url>"
        self.commit = "<unknown commit>"

    def construct(self, analyzer: Type[Analyzer], url: str, commit: str):
        assert isinstance(self, analyzer.model_type)
        self.name = analyzer.__name__
        self.version = [analyzer.version]
        self.url = url
        self.commit = commit
        return self

    def dump(self) -> str:
        return "%s/%s %s %s" % (self.name, self.version, self.url, self.commit)
