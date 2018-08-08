# style-analyzer [![PyPI](https://img.shields.io/pypi/v/lookout-style.svg)](https://pypi.python.org/pypi/lookout-style) [![Build Status](https://travis-ci.org/src-d/style-analyzer.svg)](https://travis-ci.org/src-d/style-analyzer) [![Docker Build Status](https://img.shields.io/docker/build/srcd/style-analyzer.svg)](https://hub.docker.com/r/srcd/style-analyzer) [![codecov](https://codecov.io/github/src-d/style-analyzer/coverage.svg)](https://codecov.io/gh/src-d/style-analyzer)

Style analyzer experiments

### Quick start

1. Download [`lookout`](https://github.com/src-d/lookout/releases) binary.
2. Run an instance of PostgreSQL `docker run -d --name postgres -e POSTGRES_PASSWORD=postgres -p5432:5432 postgres`
3. Install the deps `sudo pip3 install -e .`
4. Initialize the DB `python3 -m lookout init --db "postgresql://postgres:postgres@localhost:5432/postgres" --fs /tmp`
5. Run the analyzer `python3 -m lookout run lookout.style.format -s 0.0.0.0:2000 --db "postgresql://postgres:postgres@localhost:5432/postgres" --fs /tmp`
6. File a fake pull request `./lookout review -v ipv4://localhost:2000`

To erase the database of models, run `docker exec -it postgres psql -U postgres -d postgres -c "TRUNCATE models;"`

### License
AGPL-3.0, see [LICENSE.md](LICENSE.md).
