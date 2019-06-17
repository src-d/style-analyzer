# Test report for javascript / file:///tmp/top-repos-quality-repos-z21a9_wv/carlo HEAD b8ce2bca042c757b13fc82a3e059980342ddd9a8

### Classification report

PPCR: 0.856

| Class | Precision | Recall | Full Recall | F1-score | Full F1-score | Support | Full Support | PPCR |
|------:|:----------|:-------|:------------|:---------|:---------|:--------|:-------------|:-----|
| `∅` | 0.928| 0.964| 0.914| 0.946| 0.921| 1110| 1171| 0.948 |
| `␣` | 0.771| 0.846| 0.798| 0.807| 0.785| 467| 495| 0.943 |
| `'` | 1.000| 0.727| 0.667| 0.842| 0.800| 55| 60| 0.917 |
| `⏎` | 0.000| 0.000| 0.000| 0.000| 0.000| 28| 91| 0.308 |
| `⏎␣⁺␣⁺` | 0.000| 0.000| 0.000| 0.000| 0.000| 26| 70| 0.371 |
| `⏎⏎` | 0.000| 0.000| 0.000| 0.000| 0.000| 17| 43| 0.395 |
| `⏎␣⁻␣⁻` | 0.000| 0.000| 0.000| 0.000| 0.000| 2| 62| 0.032 |
| `"` | 0.000| 0.000| 0.000| 0.000| 0.000| 0| 0| 0.000 |
| `micro avg` | 0.883| 0.883| 0.756| 0.883| 0.814| 1705| 1992| 0.856 |
| `weighted avg` | 0.848| 0.883| 0.756| 0.864| 0.760| 1705| 1992| 0.856 |
| `macro avg` | 0.337| 0.317| 0.297| 0.324| 0.313| 1705| 1992| 0.856 |

### Confusion matrix

|refusal|  ∅| ␣| '| ⏎| ⏎␣⁺␣⁺| ⏎␣⁻␣⁻| "| ⏎⏎| 
|:---|:---|:---|:---|:---|:---|:---|:---|
|0 |0 |0 |0 |0 |0 |0 |0 |
|61 |1070 |40 |0 |0 |0 |0 |0 |
|28 |72 |395 |0 |0 |0 |0 |0 |
|5 |4 |11 |40 |0 |0 |0 |0 |
|63 |4 |24 |0 |0 |0 |0 |0 |
|44 |1 |25 |0 |0 |0 |0 |0 |
|60 |2 |0 |0 |0 |0 |0 |0 |
|26 |0 |17 |0 |0 |0 |0 |0 |

### Files with most errors

| filename | number of errors|
|:----:|:-----|

<details>
    <summary>Machine-readable report</summary>
```json
{
  "cl_report": {"\"": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 0}, "\u0027": {"f1-score": 0.8421052631578948, "precision": 1.0, "recall": 0.7272727272727273, "support": 55}, "macro avg": {"f1-score": 0.3243373121287912, "precision": 0.33743728148037727, "recall": 0.3171326377964494, "support": 1705}, "micro avg": {"f1-score": 0.8826979472140762, "precision": 0.8826979472140762, "recall": 0.8826979472140762, "support": 1705}, "weighted avg": {"f1-score": 0.8638287912506165, "precision": 0.8477293879300588, "recall": 0.8826979472140762, "support": 1705}, "\u2205": {"f1-score": 0.9456473707467963, "precision": 0.9280138768430182, "recall": 0.963963963963964, "support": 1110}, "\u23ce": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 28}, "\u23ce\u23ce": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 17}, "\u23ce\u2423\u207a\u2423\u207a": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 26}, "\u23ce\u2423\u207b\u2423\u207b": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 2}, "\u2423": {"f1-score": 0.8069458631256384, "precision": 0.771484375, "recall": 0.8458244111349036, "support": 467}},
  "cl_report_full": {"\"": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 0}, "\u0027": {"f1-score": 0.8, "precision": 1.0, "recall": 0.6666666666666666, "support": 60}, "macro avg": {"f1-score": 0.3131668253379527, "precision": 0.33743728148037727, "recall": 0.2972994246478448, "support": 1992}, "micro avg": {"f1-score": 0.8141736543143089, "precision": 0.8826979472140762, "recall": 0.7555220883534136, "support": 1992}, "weighted avg": {"f1-score": 0.7603509606969601, "precision": 0.7673639635583204, "recall": 0.7555220883534136, "support": 1992}, "\u2205": {"f1-score": 0.9208261617900172, "precision": 0.9280138768430182, "recall": 0.9137489325362937, "support": 1171}, "\u23ce": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 91}, "\u23ce\u23ce": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 43}, "\u23ce\u2423\u207a\u2423\u207a": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 70}, "\u23ce\u2423\u207b\u2423\u207b": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 62}, "\u2423": {"f1-score": 0.7845084409136047, "precision": 0.771484375, "recall": 0.797979797979798, "support": 495}},
  "ppcr": 0.8559236947791165
}
```
</details>
