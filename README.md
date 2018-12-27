![Format man](doc/logo.png)
# style-analyzer
Fix code style faults using 🤖


[![Read the Docs](https://img.shields.io/readthedocs/style-analyzer.svg)](https://readthedocs.org/projects/style-analyzer/)
[![Travis build status](https://travis-ci.org/src-d/style-analyzer.svg?branch=master)](https://travis-ci.org/src-d/style-analyzer)
[![Code coverage](https://codecov.io/github/src-d/style-analyzer/coverage.svg)](https://codecov.io/github/src-d/style-analyzer)
[![Docker build status](https://img.shields.io/docker/build/srcd/style-analyzer.svg)](https://hub.docker.com/r/srcd/style-analyzer)
[![PyPi package status](https://img.shields.io/pypi/v/lookout-style.svg)](https://pypi.python.org/pypi/lookout-style)
![stability: beta](https://svg-badge.appspot.com/badge/stability/beta?color=ff8000)
[![AGPL 3.0 license](https://img.shields.io/badge/license-AGPL%203.0-blue.svg)](https://opensource.org/licenses/AGPL-3.0)

[Overview](#overview) • [How To Use](#how-to-use) • [Science](#science) • [Contributions](#contributions) • [License](#license)

## Overview

This is a collection of analyzers for [Lookout](https://github.com/src-d/lookout) - the open source framework for code review intelligence.
You can run them directly on your Git repositories, but most likely you don't want that and instead
just use the upcoming code review product from [source{d}](https://sourced.tech).
Overall, this project is a mix of research ideas and their applications to solving real problems.
Consider it as an experiment at this stage.

Currently, there is the "format" analyzer working and the one "typos" incubating. All the current and the future
ones are based on machine learning and never contain any hidden domain knowledge such as static
code analysis rules or human-written pattern matchers.

* [`lookout.style.format`](lookout/style/format) - mine "white box" code formatting rules with machine learning and validate new code against them.
* [`lookout.style.typos`](lookout/style/typos) - find typos in identifier names, using the dataset of 60 million identifiers already present in open source repositories on GitHub.

"format" analyzer supports only JavaScript for now, though it is not nailed to that language and
is based on the language-agnostic [Babelfish](https://doc.bblf.sh/) parser. Everything is written in Python.

## How To Use

The following steps are required to try the "format" analyzer.

1. Download [`lookout-sdk`](https://github.com/src-d/lookout/releases) binary.
2. Start a [babelfish server](https://doc.bblf.sh/using-babelfish/getting-started.html) with the v1.2.0 javascript driver installed (`docker exec -it bblfshd bblfshctl driver install --update javascript bblfsh/javascript-driver:v1.2.0`)
3. Install the deps `pip3 install -e .`
4. Write the configuration file, e.g. `config.yml`:

```yaml
server: 0.0.0.0:9930
db: sqlite:////tmp/lookout.sqlite
fs: /tmp
```

5. Run the analyzer `python3 -m lookout run lookout.style.format -c config.yml`
6. File a fake pull request `./lookout-sdk review`

Your git repository should contain a sufficient number of JavaScript files so that it is possible
to infer sane, statistically significant rules.

## Science

The implemented analyzers are driven by bleeding edge research. One day we will write papers about them,
but first we want to focus on making them work. Below are brief descriptions of how the analyzers
are designed.

#### format

The core of the format analyzer is a language model: we learn without labeled data, just by modeling the existing format in a repository given the current code at a given point in a file. We then check whether the proposed changes follow those learnt formatting conventions.
The training algorithm is summarized below.

1. Represent a file as a linear sequence of "virtual" nodes. Some nodes correspond to the UAST nodes, and some are inserted to mirror the real tokens in the code which are not present in the UAST (e.g. white spaces, keywords, quotes or braces).
2. Identify the nodes which we use as labels - that is, identify Y-s in the (X, Y) training samples. We have around 50 classes at the moment. Some of the classes are sequences of nodes, e.g. four space indentation increase. We also predict NOOP-s: the empty gaps between non-Y nodes.
3. Extract features from the nodes surrounding the Y nodes. We take a fixed-size window and record the internal types, roles, positions and unique identifiers (for tokens which are not present in the UAST) for the left and right siblings and the parent hierarchy (2-3 levels). The features for the left and for the right siblings are different so that we avoid the information "leakage". For example, the difference in offsets between the left and the right neighbor defines the exact length of the predicted token in between.
4. We train the random forest model on the collected (X, Y) dataset. We fine-tune it with bayesian optimization.
5. We extract the rules - the branches of the trees. We prune them in several steps: first we exclude the rules which do not improve the accuracy, second we remove the rule parts which are redundant.
6. We put 95% rule confidence threshold - that is, precision on the validation - and discard the rest. This is the part which we expect to change in the future with the development of artificial random noise evaluation datasets.
7. The rules which are left is our model - the training result.

The application algorithm is much simpler - we take the rules and apply them. However, there are several quirks:
1. In case several rules are triggered, the rule with the highest confidence wins.
2. There are paired tokens which we predict such as quotes. It is possible that there are two rules which contradict each other - the left and the right quotes are predicted to be different. We pick the most confident prediction and change the second quote accordingly.
3. We check that the prediction does not break the code. For example, it can insert a newline in the middle of the expression which can change the AST. We run Babelfish on each changed line to see if the AST remains the same.
4. There is a huge chunk of code to represent the triggered rule in a human-readable format.

#### typos

We take the dataset with identifiers extracted from [Public Git Archive](https://github.com/src-d/datasets/tree/master/PublicGitArchive).
We split them (blog post is pending early November). There are frequencies present for each "atom",
so we consider top frequent ones as ground truth. For each checked "atom", we take it's embedding
computed with [fasttext](https://github.com/facebookresearch/fastText), refine it with a deep
fully-connected neural network, generate candidates with [symspell](https://github.com/wolfgarbe/SymSpell)
and rank them with [XGBoost](https://github.com/dmlc/xgboost).

## Contributions

Contributions are very welcome and desired! Please follow the [code of conduct](doc/code_of_conduct.md)
and read the [contribution guidelines](doc/contributing.md). If you want to add a new cool style
fixer backed by machine learning, it is always a good idea to discuss it on
[Slack](https://sourced.tech/community/#talk).

## License

AGPL-3.0, see [LICENSE.md](LICENSE.md).
