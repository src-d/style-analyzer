# Test report for javascript / file:///tmp/top-repos-quality-repos-u22ofaub/telescope HEAD 534030114f47696fe3f3b08ea7ca49467428f2af

### Classification report

PPCR: 0.570

| Class | Precision | Recall | Full Recall | F1-score | Full F1-score | Support | Full Support | PPCR |
|------:|:----------|:-------|:------------|:---------|:---------|:--------|:-------------|:-----|
| `∅` | 0.790| 1.000| 0.642| 0.883| 0.709| 124| 193| 0.642 |
| `␣` | 0.000| 0.000| 0.000| 0.000| 0.000| 26| 48| 0.542 |
| `'` | 1.000| 1.000| 0.765| 1.000| 0.867| 13| 17| 0.765 |
| `⏎` | 0.000| 0.000| 0.000| 0.000| 0.000| 7| 31| 0.226 |
| `␣'` | 0.000| 0.000| 0.000| 0.000| 0.000| 0| 9| 0.000 |
| `macro avg` | 0.358| 0.400| 0.281| 0.377| 0.315| 170| 298| 0.570 |
| `weighted avg` | 0.653| 0.806| 0.460| 0.720| 0.508| 170| 298| 0.570 |
| `micro avg` | 0.806| 0.806| 0.460| 0.806| 0.585| 170| 298| 0.570 |

### Confusion matrix

|refusal|  ∅| '| ␣| ␣'| ⏎| 
|:---|:---|:---|:---|:---|:---|
|0 |0 |0 |0 |0 |0 |
|69 |124 |0 |0 |0 |0 |
|4 |0 |13 |0 |0 |0 |
|22 |26 |0 |0 |0 |0 |
|9 |0 |0 |0 |0 |0 |
|24 |7 |0 |0 |0 |0 |

### Files with most errors

| filename | number of errors|
|:----:|:-----|

<details>
    <summary>Machine-readable report</summary>
```json
{
  "cl_report": {"\u0027": {"f1-score": 1.0, "precision": 1.0, "recall": 1.0, "support": 13}, "macro avg": {"f1-score": 0.37651245551601425, "precision": 0.3579617834394905, "recall": 0.4, "support": 170}, "micro avg": {"f1-score": 0.8058823529411765, "precision": 0.8058823529411765, "recall": 0.8058823529411765, "support": 170}, "weighted avg": {"f1-score": 0.7202218965878167, "precision": 0.6525665043087299, "recall": 0.8058823529411765, "support": 170}, "\u2205": {"f1-score": 0.8825622775800712, "precision": 0.7898089171974523, "recall": 1.0, "support": 124}, "\u23ce": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 7}, "\u2423": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 26}, "\u2423\u0027": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 0}},
  "cl_report_full": {"\u0027": {"f1-score": 0.8666666666666666, "precision": 1.0, "recall": 0.7647058823529411, "support": 17}, "macro avg": {"f1-score": 0.3150476190476191, "precision": 0.3579617834394905, "recall": 0.2814385857970131, "support": 298}, "micro avg": {"f1-score": 0.5854700854700854, "precision": 0.8058823529411765, "recall": 0.4597315436241611, "support": 298}, "weighted avg": {"f1-score": 0.508347714924896, "precision": 0.56856752019835, "recall": 0.4597315436241611, "support": 298}, "\u2205": {"f1-score": 0.7085714285714286, "precision": 0.7898089171974523, "recall": 0.6424870466321243, "support": 193}, "\u23ce": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 31}, "\u2423": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 48}, "\u2423\u0027": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 9}},
  "ppcr": 0.5704697986577181
}
```
</details>
