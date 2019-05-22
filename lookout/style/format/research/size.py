import glob
import json
from pathlib import Path
import re
import sys

from adjustText import adjust_text
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy
import seaborn as sns


pattern = re.compile(r'''<details>
    <summary>Machine-readable report</summary>
```json(.*)```
</details>''', re.M | re.DOTALL)


names = []
precisions = []
ppcrs = []
rules_lens = []
rules_numbers = []
supports = []
basedir = Path("/home/mog/newhugo/reports-100/")

names = []
precisions = []
supports = []
for path_test in basedir.glob("*.test_report.md"):
    name = path_test.name.replace(".test_report.md", "")
    names.append(name)
    path_train = basedir / ("%s.train_report.md" % name)
    with open(path_test) as fh_test, open(path_train) as fh_train:
        content_test = fh_test.read()
        content_train = fh_train.read()
    matches_test = pattern.search(content_test)
    matches_train = pattern.search(content_train)
    if not matches_test or not matches_train:
        sys.exit(1)
    metrics_test = json.loads(matches_test.group(1))
    metrics_train = json.loads(matches_train.group(1))
    precisions.append(
        float(metrics_test["cl_report"]["micro avg"]["precision"]))
    supports.append(int(metrics_train["cl_report"]["micro avg"]["support"]))
xs = numpy.array(precisions)
ys = numpy.array(supports)


@ticker.FuncFormatter
def major_formatter(x, pos):
    return "%dk" % (x / 1000)


fig, ax = plt.subplots(figsize=(6, 3))
ax.yaxis.set_major_formatter(major_formatter)
plot = sns.regplot(xs, ys, logx=False, ci=None, truncate=True, marker="+")
ax.set_ylim([0, ax.get_ylim()[1]])
ax.set_xlabel("Precision")
ax.set_ylabel("Samples in the training set")
texts = []

for name, x, y in zip(names, xs, ys):
    texts.append(ax.text(x, y, name, horizontalalignment='left', size='medium', color='black'))
adjust_text(texts)
plt.savefig("size.pdf", pad_inches=0, bbox_inches="tight")
