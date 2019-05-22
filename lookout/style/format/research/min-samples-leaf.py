import glob
import json
import re
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy
import seaborn as sns


pattern_quality = re.compile(r'```json(.*)```', re.M | re.DOTALL)
pattern_model = re.compile(r'''<details>
    <summary>Machine-readable report</summary>
```json(.*)```
</details>''', re.M | re.DOTALL)


all_precisions, all_ppcrs, all_rules_lens, all_rules_numbers = [], [], [], []
for n in [1, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]:
    precisions = []
    ppcrs = []
    rules_lens = []
    rules_numbers = []
    for path_quality in glob.glob(
            "/home/mog/newhugo/reports-%d/*.test_report.md" % n):
        path_model = path_quality.replace(".test_report.md",
                                          ".model_report.md")
        with open(path_quality) as fh_quality, open(path_model) as fh_model:
            content_quality = fh_quality.read()
            content_model = fh_model.read()
        matches_quality = pattern_quality.search(content_quality)
        matches_model = pattern_model.search(content_model)
        if not matches_quality or not matches_model:
            sys.exit(1)
        metrics_quality = json.loads(matches_quality.group(1))
        metrics_model = json.loads(matches_model.group(1))
        precisions.append(
            float(metrics_quality["cl_report"]["micro avg"]["precision"]))
        ppcrs.append(float(metrics_quality["ppcr"]))
        rules_lens.append(float(metrics_model["javascript"]["avg_rule_len"]))
        rules_numbers.append(int(metrics_model["javascript"]["num_rules"]))
    precisions_arr = numpy.array(precisions)
    ppcrs_arr = numpy.array(ppcrs)
    rules_lens_arr = numpy.array(rules_lens)
    rules_numbers_arr = numpy.array(rules_numbers)
    all_precisions.append(precisions)
    all_ppcrs.append(ppcrs)
    all_rules_lens.append(rules_lens)
    all_rules_numbers.append(rules_numbers)
    # print("%2d & %.2f±%.2f & %.2f±%.2f & %.2f±%.2f & %d±%.2f \\\\" % (
    #     n,
    #     precisions_arr.mean(), precisions_arr.std(),
    #     ppcrs_arr.mean(), ppcrs_arr.std(),
    #     rules_lens_arr.mean(), rules_lens_arr.std(),
    #     rules_numbers_arr.mean(), rules_numbers_arr.std(),
    # ))

x = numpy.linspace(0, 200, 11, dtype=numpy.int)

all_precisions_arr = numpy.array(all_precisions)
all_ppcrs_arr = numpy.array(all_ppcrs)
all_rules_lens_arr = numpy.array(all_rules_lens)
all_rules_numbers_arr = numpy.array(all_rules_numbers)


matplotlib.rcParams.update({'font.size': 14})


f, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True, tight_layout=True)
f.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.xlabel("Minimal number of samples per leaf")
plt.xticks(x)
ax1.plot(x, all_precisions_arr.mean(axis=1), "-")
ax1.plot(x, all_ppcrs_arr.mean(axis=1), "--")
ax1.legend(['Precision', 'PredR'])
ln2 = ax2.plot(x, all_rules_lens_arr.mean(axis=1), "-.")
ax2.set_ylabel("Rule length")
ax3 = ax2.twinx()
ax3.set_ylabel("Number of rules")
ln3 = ax3.plot(x, all_rules_numbers_arr.mean(axis=1), "C1:")
f.add_subplot(212, frameon=False)
plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
plt.legend(ln2 + ln3, ["Length", "Number"])
plt.savefig("min-samples-leaf.pdf", pad_inches=0, bbox_inches="tight")
