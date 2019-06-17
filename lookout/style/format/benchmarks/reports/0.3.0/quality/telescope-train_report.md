# Train report for javascript / file:///tmp/top-repos-quality-repos-j5mj6xka/telescope HEAD 534030114f47696fe3f3b08ea7ca49467428f2af

### Classification report

PPCR: 0.210

| Class | Precision | Recall | Full Recall | F1-score | Full F1-score | Support | Full Support | PPCR |
|------:|:----------|:-------|:------------|:---------|:---------|:--------|:-------------|:-----|
| `'` | 1.000| 1.000| 0.812| 1.000| 0.896| 203| 250| 0.812 |
| `∅` | 0.000| 0.000| 0.000| 0.000| 0.000| 0| 428| 0.000 |
| `␣` | 0.000| 0.000| 0.000| 0.000| 0.000| 0| 194| 0.000 |
| `⏎` | 0.000| 0.000| 0.000| 0.000| 0.000| 0| 93| 0.000 |
| `micro avg` | 1.000| 1.000| 0.210| 1.000| 0.348| 203| 965| 0.210 |
| `weighted avg` | 1.000| 1.000| 0.210| 1.000| 0.232| 203| 965| 0.210 |
| `macro avg` | 0.250| 0.250| 0.203| 0.250| 0.224| 203| 965| 0.210 |

### Confusion matrix

|refusal|  ∅| '| ␣| ⏎| 
|:---|:---|:---|:---|:---|
|0 |0 |0 |0 |0 |
|428 |0 |0 |0 |0 |
|47 |0 |203 |0 |0 |
|194 |0 |0 |0 |0 |
|93 |0 |0 |0 |0 |

### Files with most errors

| filename | number of errors|
|:----:|:-----|

<details>
    <summary>Machine-readable report</summary>
```json
{
  "cl_report": {"\u0027": {"f1-score": 1.0, "precision": 1.0, "recall": 1.0, "support": 203}, "macro avg": {"f1-score": 0.25, "precision": 0.25, "recall": 0.25, "support": 203}, "micro avg": {"f1-score": 1.0, "precision": 1.0, "recall": 1.0, "support": 203}, "weighted avg": {"f1-score": 1.0, "precision": 1.0, "recall": 1.0, "support": 203}, "\u2205": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 0}, "\u23ce": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 0}, "\u2423": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 0}},
  "cl_report_full": {"\u0027": {"f1-score": 0.8962472406181016, "precision": 1.0, "recall": 0.812, "support": 250}, "macro avg": {"f1-score": 0.2240618101545254, "precision": 0.25, "recall": 0.203, "support": 965}, "micro avg": {"f1-score": 0.3476027397260274, "precision": 1.0, "recall": 0.21036269430051813, "support": 965}, "weighted avg": {"f1-score": 0.23218840430520768, "precision": 0.25906735751295334, "recall": 0.21036269430051813, "support": 965}, "\u2205": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 428}, "\u23ce": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 93}, "\u2423": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 194}},
  "ppcr": 0.21036269430051813
}
```
</details>
