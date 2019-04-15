# Test report for javascript / file:///tmp/top-repos-quality-repos-sbxt95f2/carlo HEAD b8ce2bca042c757b13fc82a3e059980342ddd9a8

### Classification report

PPCR: 0.804

| Class | Precision | Recall | Full Recall | F1-score | Full F1-score | Support | Full Support | PPCR |
|------:|:----------|:-------|:------------|:---------|:---------|:--------|:-------------|:-----|
| `∅` | 0.985| 0.976| 0.922| 0.981| 0.953| 1106| 1171| 0.944 |
| `␣` | 0.849| 0.972| 0.782| 0.906| 0.814| 398| 495| 0.804 |
| `'` | 1.000| 0.845| 0.817| 0.916| 0.899| 58| 60| 0.967 |
| `⏎` | 0.000| 0.000| 0.000| 0.000| 0.000| 19| 91| 0.209 |
| `⏎⏎` | 0.000| 0.000| 0.000| 0.000| 0.000| 10| 43| 0.233 |
| `⏎␣⁺␣⁺` | 0.000| 0.000| 0.000| 0.000| 0.000| 10| 70| 0.143 |
| `"` | 0.000| 0.000| 0.000| 0.000| 0.000| 0| 0| 0.000 |
| `⏎␣⁻␣⁻` | 0.000| 0.000| 0.000| 0.000| 0.000| 0| 62| 0.000 |
| `micro avg` | 0.947| 0.947| 0.761| 0.947| 0.844| 1601| 1992| 0.804 |
| `macro avg` | 0.354| 0.349| 0.315| 0.350| 0.333| 1601| 1992| 0.804 |
| `weighted avg` | 0.928| 0.947| 0.761| 0.936| 0.789| 1601| 1992| 0.804 |

### Confusion matrix

|refusal|  ∅| ␣| '| ⏎| ⏎␣⁺␣⁺| ⏎␣⁻␣⁻| "| ⏎⏎| 
|:---|:---|:---|:---|:---|:---|:---|:---|
|0 |0 |0 |0 |0 |0 |0 |0 |
|65 |1080 |26 |0 |0 |0 |0 |0 |
|97 |11 |387 |0 |0 |0 |0 |0 |
|2 |4 |5 |49 |0 |0 |0 |0 |
|72 |1 |18 |0 |0 |0 |0 |0 |
|60 |0 |10 |0 |0 |0 |0 |0 |
|62 |0 |0 |0 |0 |0 |0 |0 |
|33 |0 |10 |0 |0 |0 |0 |0 |

### Files with most errors

| filename | number of errors|
|:----:|:-----|

<details>
    <summary>Machine-readable report</summary>
```json
{
  "cl_report": {"\"": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 0}, "\u0027": {"f1-score": 0.9158878504672897, "precision": 1.0, "recall": 0.8448275862068966, "support": 58}, "macro avg": {"f1-score": 0.3503921832495888, "precision": 0.3542607087975413, "recall": 0.3492101572274918, "support": 1601}, "micro avg": {"f1-score": 0.9469081823860087, "precision": 0.9469081823860087, "recall": 0.9469081823860087, "support": 1601}, "weighted avg": {"f1-score": 0.9361291412331992, "precision": 0.9279389946208706, "recall": 0.9469081823860087, "support": 1601}, "\u2205": {"f1-score": 0.9809264305177112, "precision": 0.9854014598540146, "recall": 0.976491862567812, "support": 1106}, "\u23ce": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 19}, "\u23ce\u23ce": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 10}, "\u23ce\u2423\u207a\u2423\u207a": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 10}, "\u23ce\u2423\u207b\u2423\u207b": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 0}, "\u2423": {"f1-score": 0.9063231850117096, "precision": 0.8486842105263158, "recall": 0.9723618090452262, "support": 398}},
  "cl_report_full": {"\"": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 0}, "\u0027": {"f1-score": 0.8990825688073394, "precision": 1.0, "recall": 0.8166666666666667, "support": 60}, "macro avg": {"f1-score": 0.33322046920726844, "precision": 0.3542607087975413, "recall": 0.3150966863338768, "support": 1992}, "micro avg": {"f1-score": 0.8438630670748678, "precision": 0.9469081823860087, "recall": 0.7610441767068273, "support": 1992}, "weighted avg": {"f1-score": 0.7894305503459097, "precision": 0.8202830289656512, "recall": 0.7610441767068273, "support": 1992}, "\u2205": {"f1-score": 0.9528010586678429, "precision": 0.9854014598540146, "recall": 0.9222886421861657, "support": 1171}, "\u23ce": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 91}, "\u23ce\u23ce": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 43}, "\u23ce\u2423\u207a\u2423\u207a": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 70}, "\u23ce\u2423\u207b\u2423\u207b": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 62}, "\u2423": {"f1-score": 0.8138801261829653, "precision": 0.8486842105263158, "recall": 0.7818181818181819, "support": 495}},
  "ppcr": 0.803714859437751
}
```
</details>
