# Train report for javascript / file:///tmp/top-repos-quality-repos-6jaweu5a/freeCodeCamp HEAD cf65516cce60645a417e44c4fcea7418ca920572

### Classification report

PPCR: 0.863

| Class | Precision | Recall | Full Recall | F1-score | Full F1-score | Support | Full Support | PPCR |
|------:|:----------|:-------|:------------|:---------|:---------|:--------|:-------------|:-----|
| `∅` | 0.960| 0.993| 0.962| 0.976| 0.961| 58403| 60317| 0.968 |
| `␣` | 0.961| 0.970| 0.817| 0.966| 0.883| 25172| 29883| 0.842 |
| `'` | 0.981| 1.000| 0.999| 0.991| 0.990| 11032| 11046| 0.999 |
| `⏎` | 0.931| 0.833| 0.360| 0.879| 0.520| 4153| 9596| 0.433 |
| `⏎␣⁻␣⁻` | 0.924| 0.716| 0.543| 0.806| 0.684| 3063| 4036| 0.759 |
| `⏎␣⁺␣⁺` | 0.921| 0.578| 0.310| 0.710| 0.464| 2387| 4452| 0.536 |
| `⏎⏎` | 0.936| 0.843| 0.269| 0.887| 0.418| 728| 2279| 0.319 |
| `"` | 0.000| 0.000| 0.000| 0.000| 0.000| 208| 208| 1.000 |
| `⏎␣⁻␣⁻␣⁻␣⁻` | 0.000| 0.000| 0.000| 0.000| 0.000| 172| 227| 0.758 |
| `micro avg` | 0.960| 0.960| 0.828| 0.960| 0.889| 105318| 122044| 0.863 |
| `weighted avg` | 0.956| 0.960| 0.828| 0.956| 0.869| 105318| 122044| 0.863 |
| `macro avg` | 0.735| 0.659| 0.473| 0.691| 0.547| 105318| 122044| 0.863 |

### Confusion matrix

|refusal|  ∅| ␣| '| ⏎| ⏎␣⁺␣⁺| ⏎␣⁻␣⁻| ⏎⏎| "| ⏎␣⁻␣⁻␣⁻␣⁻| 
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
|0 |0 |0 |0 |0 |0 |0 |0 |0 |0 |
|1914 |58003 |376 |0 |0 |2 |21 |1 |0 |0 |
|4711 |433 |24418 |0 |169 |113 |35 |4 |0 |0 |
|14 |0 |0 |11032 |0 |0 |0 |0 |0 |0 |
|5443 |430 |224 |0 |3459 |3 |0 |37 |0 |0 |
|2065 |847 |160 |0 |0 |1380 |0 |0 |0 |0 |
|973 |660 |195 |0 |16 |0 |2192 |0 |0 |0 |
|1551 |16 |27 |0 |71 |0 |0 |614 |0 |0 |
|0 |0 |0 |208 |0 |0 |0 |0 |0 |0 |
|55 |46 |0 |0 |1 |0 |125 |0 |0 |0 |

### Files with most errors

| filename | number of errors|
|:----:|:-----|
| api-server/public/js/calculator.js | 327 |
| api-server/common/models/user.js | 282 |
| api-server/server/boot/challenge.js | 104 |
| api-server/server/utils/map.js | 85 |
| curriculum/test/test-challenges.js | 70 |
| api-server/server/boot/certificate.js | 65 |
| api-server/server/utils/user-stats.test.js | 65 |
| api-server/server/boot/settings.js | 57 |
| client/src/components/settings/Portfolio.js | 55 |
| api-server/common/utils/ajax-stream.js | 55 |

<details>
    <summary>Machine-readable report</summary>
```json
{
  "cl_report": {"\"": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 208}, "\u0027": {"f1-score": 0.99066091954023, "precision": 0.9814946619217082, "recall": 1.0, "support": 11032}, "macro avg": {"f1-score": 0.690648019505774, "precision": 0.7349289368546545, "recall": 0.6592517119462183, "support": 105318}, "micro avg": {"f1-score": 0.959930876013597, "precision": 0.959930876013597, "recall": 0.959930876013597, "support": 105318}, "weighted avg": {"f1-score": 0.9562577148242907, "precision": 0.9557240390264581, "recall": 0.959930876013597, "support": 105318}, "\u2205": {"f1-score": 0.9761692387956714, "precision": 0.9597584181351866, "recall": 0.9931510367618102, "support": 58403}, "\u23ce": {"f1-score": 0.8791460160121999, "precision": 0.9308396124865447, "recall": 0.8328918853840597, "support": 4153}, "\u23ce\u23ce": {"f1-score": 0.8872832369942195, "precision": 0.9359756097560976, "recall": 0.8434065934065934, "support": 728}, "\u23ce\u2423\u207a\u2423\u207a": {"f1-score": 0.7104247104247104, "precision": 0.9212283044058746, "recall": 0.5781315458734814, "support": 2387}, "\u23ce\u2423\u207b\u2423\u207b": {"f1-score": 0.8064753495217071, "precision": 0.9237252423093131, "recall": 0.7156382631407118, "support": 3063}, "\u23ce\u2423\u207b\u2423\u207b\u2423\u207b\u2423\u207b": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 172}, "\u2423": {"f1-score": 0.9656727042632287, "precision": 0.9613385826771653, "recall": 0.9700460829493087, "support": 25172}},
  "cl_report_full": {"\"": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 208}, "\u0027": {"f1-score": 0.9900385892488557, "precision": 0.9814946619217082, "recall": 0.9987325728770595, "support": 11046}, "macro avg": {"f1-score": 0.5466778343422406, "precision": 0.7349289368546545, "recall": 0.47338364877860745, "support": 122044}, "micro avg": {"f1-score": 0.889313077823031, "precision": 0.959930876013597, "recall": 0.8283733735374127, "support": 122044}, "weighted avg": {"f1-score": 0.86892175595515, "precision": 0.9533767185082375, "recall": 0.8283733735374127, "support": 122044}, "\u2205": {"f1-score": 0.9606963031668213, "precision": 0.9597584181351866, "recall": 0.9616360230117545, "support": 60317}, "\u23ce": {"f1-score": 0.5196814903846154, "precision": 0.9308396124865447, "recall": 0.36046269278866194, "support": 9596}, "\u23ce\u23ce": {"f1-score": 0.4183986371379897, "precision": 0.9359756097560976, "recall": 0.26941641070645017, "support": 2279}, "\u23ce\u2423\u207a\u2423\u207a": {"f1-score": 0.4638655462184874, "precision": 0.9212283044058746, "recall": 0.30997304582210244, "support": 4452}, "\u23ce\u2423\u207b\u2423\u207b": {"f1-score": 0.6840380714620066, "precision": 0.9237252423093131, "recall": 0.5431119920713577, "support": 4036}, "\u23ce\u2423\u207b\u2423\u207b\u2423\u207b\u2423\u207b": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 227}, "\u2423": {"f1-score": 0.8833818714613895, "precision": 0.9613385826771653, "recall": 0.8171201017300807, "support": 29883}},
  "ppcr": 0.862951066828357
}
```
</details>
