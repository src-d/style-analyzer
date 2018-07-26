from modelforge import Model


class Analyzer:
    def __init__(self, model: Model, config: dict):
        raise NotImplementedError

    def analyze(self, url, commit):
        raise NotImplementedError

    @classmethod
    def train(cls, url, commit, config):
        raise NotImplementedError
