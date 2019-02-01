# Train report for javascript / file:///tmp/top-repos-quality-repos-u22ofaub/telescope HEAD 534030114f47696fe3f3b08ea7ca49467428f2af

### Classification report

PPCR: 0.166

| Class | Precision | Recall | Full Recall | F1-score | Full F1-score | Support | Full Support | PPCR |
|------:|:----------|:-------|:------------|:---------|:---------|:--------|:-------------|:-----|
| `'` | 1.000| 1.000| 1.000| 1.000| 1.000| 121| 121| 1.000 |
| `∅` | 0.000| 0.000| 0.000| 0.000| 0.000| 0| 411| 0.000 |
| `␣` | 0.000| 0.000| 0.000| 0.000| 0.000| 0| 110| 0.000 |
| `␣'` | 0.000| 0.000| 0.000| 0.000| 0.000| 0| 0| 0.000 |
| `⏎` | 0.000| 0.000| 0.000| 0.000| 0.000| 0| 89| 0.000 |
| `macro avg` | 0.200| 0.200| 0.200| 0.200| 0.200| 121| 731| 0.166 |
| `weighted avg` | 1.000| 1.000| 0.166| 1.000| 0.166| 121| 731| 0.166 |
| `micro avg` | 1.000| 1.000| 0.166| 1.000| 0.284| 121| 731| 0.166 |

### Confusion matrix

|refusal|  ∅| '| ␣| ␣'| ⏎| 
|:---|:---|:---|:---|:---|
|0 |0 |0 |0 |0 |
|411 |0 |0 |0 |0 |
|0 |0 |121 |0 |0 |
|110 |0 |0 |0 |0 |
|89 |0 |0 |0 |0 |

### Files with most errors

| filename | number of errors|
|:----:|:-----|

<details>
    <summary>Machine-readable report</summary>
```json
{
  "cl_report": {"\u0027": {"f1-score": 1.0, "precision": 1.0, "recall": 1.0, "support": 121}, "macro avg": {"f1-score": 0.2, "precision": 0.2, "recall": 0.2, "support": 121}, "micro avg": {"f1-score": 1.0, "precision": 1.0, "recall": 1.0, "support": 121}, "weighted avg": {"f1-score": 1.0, "precision": 1.0, "recall": 1.0, "support": 121}, "\u2205": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 0}, "\u23ce": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 0}, "\u2423": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 0}, "\u2423\u0027": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 0}},
  "cl_report_full": {"\u0027": {"f1-score": 1.0, "precision": 1.0, "recall": 1.0, "support": 121}, "macro avg": {"f1-score": 0.2, "precision": 0.2, "recall": 0.2, "support": 731}, "micro avg": {"f1-score": 0.28403755868544606, "precision": 1.0, "recall": 0.16552667578659372, "support": 731}, "weighted avg": {"f1-score": 0.16552667578659372, "precision": 0.16552667578659372, "recall": 0.16552667578659372, "support": 731}, "\u2205": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 411}, "\u23ce": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 89}, "\u2423": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 110}, "\u2423\u0027": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 0}},
  "ppcr": 0.16552667578659372
}
```
</details>
