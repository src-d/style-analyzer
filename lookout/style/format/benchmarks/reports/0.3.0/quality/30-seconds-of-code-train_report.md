# Train report for javascript / file:///tmp/top-repos-quality-repos-iiso_e9t/30-seconds-of-code HEAD 3a122c9cfcbdc091227879a06a32bc67ccd0d35d

### Classification report

PPCR: 0.938

| Class | Precision | Recall | Full Recall | F1-score | Full F1-score | Support | Full Support | PPCR |
|------:|:----------|:-------|:------------|:---------|:---------|:--------|:-------------|:-----|
| `∅` | 0.945| 0.987| 0.967| 0.966| 0.956| 37102| 37869| 0.980 |
| `␣` | 0.941| 0.895| 0.749| 0.917| 0.834| 14573| 17395| 0.838 |
| `'` | 1.000| 1.000| 0.991| 1.000| 0.995| 6192| 6248| 0.991 |
| `⏎` | 0.911| 0.837| 0.799| 0.873| 0.852| 3052| 3197| 0.955 |
| `⏎␣⁺␣⁺` | 0.926| 0.791| 0.746| 0.853| 0.826| 1780| 1888| 0.943 |
| `⏎␣⁻␣⁻` | 0.999| 0.802| 0.777| 0.890| 0.874| 1724| 1778| 0.970 |
| `⏎⏎` | 0.000| 0.000| 0.000| 0.000| 0.000| 114| 457| 0.249 |
| `micro avg` | 0.948| 0.948| 0.889| 0.948| 0.918| 64537| 68832| 0.938 |
| `weighted avg` | 0.947| 0.948| 0.889| 0.947| 0.912| 64537| 68832| 0.938 |
| `macro avg` | 0.817| 0.759| 0.719| 0.785| 0.763| 64537| 68832| 0.938 |

### Confusion matrix

|refusal|  ∅| ␣| '| ⏎| ⏎␣⁺␣⁺| ⏎␣⁻␣⁻| ⏎⏎| 
|:---|:---|:---|:---|:---|:---|:---|:---|
|0 |0 |0 |0 |0 |0 |0 |0 |
|767 |36638 |374 |0 |0 |90 |0 |0 |
|2822 |1497 |13036 |0 |17 |23 |0 |0 |
|56 |0 |0 |6192 |0 |0 |0 |0 |
|145 |184 |312 |0 |2556 |0 |0 |0 |
|108 |281 |90 |0 |0 |1408 |1 |0 |
|54 |173 |40 |0 |129 |0 |1382 |0 |
|343 |11 |0 |0 |103 |0 |0 |0 |

### Files with most errors

| filename | number of errors|
|:----:|:-----|
| scripts/web.js | 237 |
| scripts/util.js | 101 |
| scripts/tdd.js | 58 |
| scripts/extract.js | 45 |
| scripts/analyze.js | 40 |
| scripts/rollup.js | 40 |
| scripts/localize.js | 39 |
| scripts/tag.js | 36 |
| test/levenshteinDistance/levenshteinDistance.js | 33 |
| test/hexToRGB/hexToRGB.js | 33 |

<details>
    <summary>Machine-readable report</summary>
```json
{
  "cl_report": {"\u0027": {"f1-score": 1.0, "precision": 1.0, "recall": 1.0, "support": 6192}, "macro avg": {"f1-score": 0.7854725402518042, "precision": 0.8174247280005782, "recall": 0.7588777001107717, "support": 64537}, "micro avg": {"f1-score": 0.9484791669894789, "precision": 0.9484791669894789, "recall": 0.9484791669894789, "support": 64537}, "weighted avg": {"f1-score": 0.9467518616064704, "precision": 0.9468549958932182, "recall": 0.9484791669894789, "support": 64537}, "\u2205": {"f1-score": 0.9656063041931318, "precision": 0.944667904290429, "recall": 0.9874939356368929, "support": 37102}, "\u23ce": {"f1-score": 0.8728017756530647, "precision": 0.9112299465240642, "recall": 0.8374836173001311, "support": 3052}, "\u23ce\u23ce": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 114}, "\u23ce\u2423\u207a\u2423\u207a": {"f1-score": 0.8530748258103605, "precision": 0.925706771860618, "recall": 0.7910112359550562, "support": 1780}, "\u23ce\u2423\u207b\u2423\u207b": {"f1-score": 0.8896041197296427, "precision": 0.9992769342010123, "recall": 0.8016241299303944, "support": 1724}, "\u2423": {"f1-score": 0.9172207563764293, "precision": 0.9410915391279238, "recall": 0.8945309819529267, "support": 14573}},
  "cl_report_full": {"\u0027": {"f1-score": 0.9954983922829581, "precision": 1.0, "recall": 0.9910371318822023, "support": 6248}, "macro avg": {"f1-score": 0.7625712051733623, "precision": 0.8174247280005782, "recall": 0.7186401664693188, "support": 68832}, "micro avg": {"f1-score": 0.9179344525339471, "precision": 0.9484791669894789, "recall": 0.8892956764295676, "support": 68832}, "weighted avg": {"f1-score": 0.911957410835585, "precision": 0.9418521487683661, "recall": 0.8892956764295676, "support": 68832}, "\u2205": {"f1-score": 0.9559443205093081, "precision": 0.944667904290429, "recall": 0.9674932002429428, "support": 37869}, "\u23ce": {"f1-score": 0.8517160946351217, "precision": 0.9112299465240642, "recall": 0.7994995308101345, "support": 3197}, "\u23ce\u23ce": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 457}, "\u23ce\u2423\u207a\u2423\u207a": {"f1-score": 0.8260486946318568, "precision": 0.925706771860618, "recall": 0.7457627118644068, "support": 1888}, "\u23ce\u2423\u207b\u2423\u207b": {"f1-score": 0.8744068332806075, "precision": 0.9992769342010123, "recall": 0.7772778402699663, "support": 1778}, "\u2423": {"f1-score": 0.834384100873684, "precision": 0.9410915391279238, "recall": 0.7494107502155792, "support": 17395}},
  "ppcr": 0.9376016968851697
}
```
</details>
