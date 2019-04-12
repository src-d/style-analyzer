# Train report for javascript / file:///tmp/top-repos-quality-repos-sbxt95f2/carlo HEAD b8ce2bca042c757b13fc82a3e059980342ddd9a8

### Classification report

PPCR: 0.584

| Class | Precision | Recall | Full Recall | F1-score | Full F1-score | Support | Full Support | PPCR |
|------:|:----------|:-------|:------------|:---------|:---------|:--------|:-------------|:-----|
| `∅` | 0.977| 0.997| 0.889| 0.987| 0.931| 3859| 4328| 0.892 |
| `␣` | 0.956| 0.886| 0.216| 0.920| 0.353| 516| 2114| 0.244 |
| `'` | 1.000| 1.000| 0.489| 1.000| 0.657| 312| 638| 0.489 |
| `⏎` | 0.000| 0.000| 0.000| 0.000| 0.000| 19| 407| 0.047 |
| `⏎␣⁻␣⁻` | 0.000| 0.000| 0.000| 0.000| 0.000| 13| 210| 0.062 |
| `⏎␣⁺␣⁺` | 0.000| 0.000| 0.000| 0.000| 0.000| 8| 253| 0.032 |
| `⏎⏎` | 0.000| 0.000| 0.000| 0.000| 0.000| 1| 148| 0.007 |
| `"` | 0.000| 0.000| 0.000| 0.000| 0.000| 0| 0| 0.000 |
| `micro avg` | 0.977| 0.977| 0.570| 0.977| 0.720| 4728| 8098| 0.584 |
| `macro avg` | 0.367| 0.360| 0.199| 0.363| 0.243| 4728| 8098| 0.584 |
| `weighted avg` | 0.968| 0.977| 0.570| 0.972| 0.642| 4728| 8098| 0.584 |

### Confusion matrix

|refusal|  ∅| ␣| '| ⏎| ⏎␣⁺␣⁺| ⏎␣⁻␣⁻| "| ⏎⏎| 
|:---|:---|:---|:---|:---|:---|:---|:---|
|0 |0 |0 |0 |0 |0 |0 |0 |
|469 |3849 |10 |0 |0 |0 |0 |0 |
|1598 |59 |457 |0 |0 |0 |0 |0 |
|326 |0 |0 |312 |0 |0 |0 |0 |
|388 |10 |9 |0 |0 |0 |0 |0 |
|245 |6 |2 |0 |0 |0 |0 |0 |
|197 |13 |0 |0 |0 |0 |0 |0 |
|147 |1 |0 |0 |0 |0 |0 |0 |

### Files with most errors

| filename | number of errors|
|:----:|:-----|
| lib/find-chrome.js | 36 |
| lib/carlo.js | 27 |
| lib/color.js | 22 |
| lib/rpc.js | 9 |
| examples/terminal/main.js | 7 |
| examples/systeminfo/main.js | 6 |
| lib/intercepted_request.js | 3 |

<details>
    <summary>Machine-readable report</summary>
```json
{
  "cl_report": {"\"": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 0}, "\u0027": {"f1-score": 1.0, "precision": 1.0, "recall": 1.0, "support": 312}, "macro avg": {"f1-score": 0.3633524889235374, "precision": 0.36668333011043563, "recall": 0.36038344622758434, "support": 4728}, "micro avg": {"f1-score": 0.9767343485617598, "precision": 0.9767343485617598, "recall": 0.9767343485617598, "support": 4728}, "weighted avg": {"f1-score": 0.9721811260582024, "precision": 0.9680871336730517, "recall": 0.9767343485617598, "support": 4728}, "\u2205": {"f1-score": 0.9873028087726049, "precision": 0.9773996952767903, "recall": 0.9974086550919927, "support": 3859}, "\u23ce": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 19}, "\u23ce\u23ce": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 1}, "\u23ce\u2423\u207a\u2423\u207a": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 8}, "\u23ce\u2423\u207b\u2423\u207b": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 13}, "\u2423": {"f1-score": 0.9195171026156942, "precision": 0.9560669456066946, "recall": 0.8856589147286822, "support": 516}},
  "cl_report_full": {"\"": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 0}, "\u0027": {"f1-score": 0.6568421052631579, "precision": 1.0, "recall": 0.4890282131661442, "support": 638}, "macro avg": {"f1-score": 0.2425937928855012, "precision": 0.36668333011043563, "recall": 0.19931642481430206, "support": 8098}, "micro avg": {"f1-score": 0.7200997972867613, "precision": 0.9767343485617598, "recall": 0.5702642627809336, "support": 8098}, "weighted avg": {"f1-score": 0.6415302276010383, "precision": 0.8507423319548656, "recall": 0.5702642627809336, "support": 8098}, "\u2205": {"f1-score": 0.9312847810307283, "precision": 0.9773996952767903, "recall": 0.8893253234750462, "support": 4328}, "\u23ce": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 407}, "\u23ce\u23ce": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 148}, "\u23ce\u2423\u207a\u2423\u207a": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 253}, "\u23ce\u2423\u207b\u2423\u207b": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 210}, "\u2423": {"f1-score": 0.3526234567901234, "precision": 0.9560669456066946, "recall": 0.2161778618732261, "support": 2114}},
  "ppcr": 0.5838478636700419
}
```
</details>
