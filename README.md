# style-analyzer [![PyPI](https://img.shields.io/pypi/v/lookout-style.svg)](https://pypi.python.org/pypi/lookout-style) [![Build Status](https://travis-ci.org/src-d/style-analyzer.svg)](https://travis-ci.org/src-d/style-analyzer) [![Docker Build Status](https://img.shields.io/docker/build/srcd/style-analyzer.svg)](https://hub.docker.com/r/srcd/style-analyzer) [![codecov](https://codecov.io/github/src-d/style-analyzer/coverage.svg)](https://codecov.io/gh/src-d/style-analyzer) [![Read the Docs](https://img.shields.io/readthedocs/style-analyzer.svg)](https://readthedocs.org/projects/style-analyzer/)


Code style analyzer.

### How to write and run an Analyzer using Python SDK

[lookout/core/doc/getting_started.md](lookout/core/doc/getting_started.md)

### How to run the format analyzer

1. Download [`lookout`](https://github.com/src-d/lookout/releases) binary.
2. Install the deps `sudo pip3 install -e .`
3. Write the configuration file `lookout.yaml`:

```yaml
server: 0.0.0.0:2000
db: sqlite:////tmp/lookout.sqlite
fs: /tmp
```

4. Run the analyzer `python3 -m lookout run lookout.style.format -c lookout.yaml`
5. File a fake pull request `./lookout review -v ipv4://localhost:2000`

### API Documentation

API documentation is available on [Read The Docs](https://style-analyzer.readthedocs.io/en/latest/).

### License
AGPL-3.0, see [LICENSE.md](LICENSE.md).
