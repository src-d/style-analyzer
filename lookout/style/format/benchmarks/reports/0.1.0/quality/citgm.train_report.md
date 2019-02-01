# Train report for javascript / file:///tmp/top-repos-quality-repos-hb3o_tt9/citgm HEAD 0c4c7ccdd1cad8ce9506e34ca523787ba18cafe2

### Classification report

PPCR: 0.900

| Class | Precision | Recall | Full Recall | F1-score | Full F1-score | Support | Full Support | PPCR |
|------:|:----------|:-------|:------------|:---------|:---------|:--------|:-------------|:-----|
| `∅` | 0.985| 0.993| 0.980| 0.989| 0.982| 11690| 11845| 0.987 |
| `␣` | 0.967| 0.972| 0.888| 0.969| 0.926| 4322| 4730| 0.914 |
| `'` | 1.000| 1.000| 1.000| 1.000| 1.000| 2133| 2133| 1.000 |
| `⏎␣⁺␣⁺` | 0.978| 0.950| 0.796| 0.964| 0.878| 646| 771| 0.838 |
| `⏎␣⁻␣⁻` | 1.000| 0.967| 0.702| 0.983| 0.825| 480| 661| 0.726 |
| `⏎` | 0.962| 0.703| 0.171| 0.813| 0.290| 327| 1345| 0.243 |
| `⏎⏎` | 0.947| 0.961| 0.281| 0.954| 0.433| 129| 442| 0.292 |
| `␣'` | 0.000| 0.000| 0.000| 0.000| 0.000| 0| 0| 0.000 |
| `macro avg` | 0.855| 0.818| 0.602| 0.834| 0.667| 19727| 21927| 0.900 |
| `weighted avg` | 0.982| 0.982| 0.884| 0.982| 0.910| 19727| 21927| 0.900 |
| `micro avg` | 0.982| 0.982| 0.884| 0.982| 0.930| 19727| 21927| 0.900 |

### Confusion matrix

|refusal|  ∅| ␣| '| ⏎| ⏎␣⁺␣⁺| ␣'| ⏎␣⁻␣⁻| ⏎⏎| 
|:---|:---|:---|:---|:---|:---|:---|:---|
|0 |0 |0 |0 |0 |0 |0 |0 |
|155 |11608 |76 |0 |0 |6 |0 |0 |
|408 |108 |4202 |0 |7 |5 |0 |0 |
|0 |0 |0 |2133 |0 |0 |0 |0 |
|1018 |33 |56 |0 |230 |1 |0 |7 |
|125 |18 |12 |0 |2 |614 |0 |0 |
|181 |13 |1 |0 |0 |2 |464 |0 |
|313 |5 |0 |0 |0 |0 |0 |124 |

### Files with most errors

| filename | number of errors|
|:----:|:-----|
| test/test-lookup.js | 32 |
| lib/match-conditions.js | 28 |
| lib/citgm.js | 27 |
| bin/citgm-all.js | 26 |
| lib/lookup.js | 24 |
| lib/common-args.js | 20 |
| test/reporter/test-reporter-util.js | 19 |
| lib/out.js | 17 |
| bin/citgm.js | 16 |
| test/test-match-conditions.js | 15 |

<details>
    <summary>Machine-readable report</summary>
```json
{
  "cl_report": {"\u0027": {"f1-score": 1.0, "precision": 1.0, "recall": 1.0, "support": 2133}, "macro avg": {"f1-score": 0.8339886740098946, "precision": 0.8547799447911962, "recall": 0.8183694776764701, "support": 19727}, "micro avg": {"f1-score": 0.9821564353424241, "precision": 0.9821564353424241, "recall": 0.9821564353424241, "support": 19727}, "weighted avg": {"f1-score": 0.9817636485024301, "precision": 0.98208812850414, "recall": 0.9821564353424241, "support": 19727}, "\u2205": {"f1-score": 0.9889669861554845, "precision": 0.9849809079338142, "recall": 0.9929854576561163, "support": 11690}, "\u23ce": {"f1-score": 0.812720848056537, "precision": 0.9623430962343096, "recall": 0.7033639143730887, "support": 327}, "\u23ce\u23ce": {"f1-score": 0.9538461538461538, "precision": 0.9465648854961832, "recall": 0.9612403100775194, "support": 129}, "\u23ce\u2423\u207a\u2423\u207a": {"f1-score": 0.9638932496075353, "precision": 0.9777070063694268, "recall": 0.9504643962848297, "support": 646}, "\u23ce\u2423\u207b\u2423\u207b": {"f1-score": 0.983050847457627, "precision": 1.0, "recall": 0.9666666666666667, "support": 480}, "\u2423": {"f1-score": 0.9694313069558196, "precision": 0.9666436622958362, "recall": 0.97223507635354, "support": 4322}, "\u2423\u0027": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 0}},
  "cl_report_full": {"\u0027": {"f1-score": 1.0, "precision": 1.0, "recall": 1.0, "support": 2133}, "macro avg": {"f1-score": 0.6667761246231887, "precision": 0.8547799447911962, "recall": 0.6022806780529824, "support": 21927}, "micro avg": {"f1-score": 0.9302828059730158, "precision": 0.9821564353424241, "recall": 0.8836138094586583, "support": 21927}, "weighted avg": {"f1-score": 0.9100051708469428, "precision": 0.9805203002175683, "recall": 0.8836138094586583, "support": 21927}, "\u2205": {"f1-score": 0.9824798984341939, "precision": 0.9849809079338142, "recall": 0.9799915576192486, "support": 11845}, "\u23ce": {"f1-score": 0.29040404040404044, "precision": 0.9623430962343096, "recall": 0.17100371747211895, "support": 1345}, "\u23ce\u23ce": {"f1-score": 0.43280977312390917, "precision": 0.9465648854961832, "recall": 0.28054298642533937, "support": 442}, "\u23ce\u2423\u207a\u2423\u207a": {"f1-score": 0.8777698355968548, "precision": 0.9777070063694268, "recall": 0.7963683527885862, "support": 771}, "\u23ce\u2423\u207b\u2423\u207b": {"f1-score": 0.8248888888888889, "precision": 1.0, "recall": 0.7019667170953101, "support": 661}, "\u2423": {"f1-score": 0.9258565605376227, "precision": 0.9666436622958362, "recall": 0.8883720930232558, "support": 4730}, "\u2423\u0027": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 0}},
  "ppcr": 0.899667077119533
}
```
</details>
