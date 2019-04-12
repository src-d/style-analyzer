# Test report for javascript / file:///tmp/top-repos-quality-repos-xh6434di/telescope HEAD 534030114f47696fe3f3b08ea7ca49467428f2af

### Classification report

PPCR: 0.553

| Class | Precision | Recall | Full Recall | F1-score | Full F1-score | Support | Full Support | PPCR |
|------:|:----------|:-------|:------------|:---------|:---------|:--------|:-------------|:-----|
| `∅` | 0.800| 1.000| 0.628| 0.889| 0.704| 120| 191| 0.628 |
| `␣` | 0.000| 0.000| 0.000| 0.000| 0.000| 24| 63| 0.381 |
| `'` | 1.000| 1.000| 0.846| 1.000| 0.917| 22| 26| 0.846 |
| `⏎` | 0.000| 0.000| 0.000| 0.000| 0.000| 6| 31| 0.194 |
| `micro avg` | 0.826| 0.826| 0.457| 0.826| 0.588| 172| 311| 0.553 |
| `macro avg` | 0.450| 0.500| 0.369| 0.472| 0.405| 172| 311| 0.553 |
| `weighted avg` | 0.686| 0.826| 0.457| 0.748| 0.509| 172| 311| 0.553 |

### Confusion matrix

|refusal|  ∅| '| ␣| ⏎| 
|:---|:---|:---|:---|:---|
|0 |0 |0 |0 |0 |
|71 |120 |0 |0 |0 |
|4 |0 |22 |0 |0 |
|39 |24 |0 |0 |0 |
|25 |6 |0 |0 |0 |

### Files with most errors

| filename | number of errors|
|:----:|:-----|

<details>
    <summary>Machine-readable report</summary>
```json
{
  "cl_report": {"\u0027": {"f1-score": 1.0, "precision": 1.0, "recall": 1.0, "support": 22}, "macro avg": {"f1-score": 0.4722222222222222, "precision": 0.45, "recall": 0.5, "support": 172}, "micro avg": {"f1-score": 0.8255813953488372, "precision": 0.8255813953488372, "recall": 0.8255813953488372, "support": 172}, "weighted avg": {"f1-score": 0.7480620155038761, "precision": 0.686046511627907, "recall": 0.8255813953488372, "support": 172}, "\u2205": {"f1-score": 0.888888888888889, "precision": 0.8, "recall": 1.0, "support": 120}, "\u23ce": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 6}, "\u2423": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 24}},
  "cl_report_full": {"\u0027": {"f1-score": 0.9166666666666666, "precision": 1.0, "recall": 0.8461538461538461, "support": 26}, "macro avg": {"f1-score": 0.40511974584555227, "precision": 0.45, "recall": 0.36860652436568664, "support": 311}, "micro avg": {"f1-score": 0.587991718426501, "precision": 0.8255813953488372, "recall": 0.4565916398713826, "support": 311}, "weighted avg": {"f1-score": 0.5088793756463085, "precision": 0.57491961414791, "recall": 0.4565916398713826, "support": 311}, "\u2205": {"f1-score": 0.7038123167155425, "precision": 0.8, "recall": 0.6282722513089005, "support": 191}, "\u23ce": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 31}, "\u2423": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 63}},
  "ppcr": 0.5530546623794212
}
```
</details>
