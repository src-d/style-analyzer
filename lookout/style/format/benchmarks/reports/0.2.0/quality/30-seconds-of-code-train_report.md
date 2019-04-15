# Train report for javascript / file:///tmp/top-repos-quality-repos-o7mv5p7e/30-seconds-of-code HEAD 3a122c9cfcbdc091227879a06a32bc67ccd0d35d

### Classification report

PPCR: 0.938

| Class | Precision | Recall | Full Recall | F1-score | Full F1-score | Support | Full Support | PPCR |
|------:|:----------|:-------|:------------|:---------|:---------|:--------|:-------------|:-----|
| `∅` | 0.944| 0.987| 0.968| 0.965| 0.956| 37111| 37869| 0.980 |
| `␣` | 0.941| 0.894| 0.749| 0.917| 0.834| 14586| 17394| 0.839 |
| `'` | 1.000| 1.000| 0.991| 1.000| 0.995| 6192| 6248| 0.991 |
| `⏎` | 0.911| 0.837| 0.799| 0.873| 0.852| 3052| 3197| 0.955 |
| `⏎␣⁺␣⁺` | 0.926| 0.791| 0.746| 0.853| 0.826| 1781| 1888| 0.943 |
| `⏎␣⁻␣⁻` | 0.999| 0.802| 0.777| 0.890| 0.874| 1724| 1778| 0.970 |
| `⏎⏎` | 0.000| 0.000| 0.000| 0.000| 0.000| 114| 457| 0.249 |
| `micro avg` | 0.948| 0.948| 0.889| 0.948| 0.918| 64560| 68831| 0.938 |
| `macro avg` | 0.817| 0.759| 0.719| 0.785| 0.763| 64560| 68831| 0.938 |
| `weighted avg` | 0.947| 0.948| 0.889| 0.947| 0.912| 64560| 68831| 0.938 |

### Confusion matrix

|refusal|  ∅| ␣| '| ⏎| ⏎␣⁺␣⁺| ⏎␣⁻␣⁻| ⏎⏎| 
|:---|:---|:---|:---|:---|:---|:---|:---|
|0 |0 |0 |0 |0 |0 |0 |0 |
|758 |36647 |374 |0 |0 |90 |0 |0 |
|2808 |1511 |13035 |0 |17 |23 |0 |0 |
|56 |0 |0 |6192 |0 |0 |0 |0 |
|145 |184 |312 |0 |2556 |0 |0 |0 |
|107 |282 |90 |0 |0 |1408 |1 |0 |
|54 |173 |40 |0 |129 |0 |1382 |0 |
|343 |11 |0 |0 |103 |0 |0 |0 |

### Files with most errors

| filename | number of errors|
|:----:|:-----|
| scripts/web.js | 238 |
| scripts/util.js | 101 |
| scripts/tdd.js | 58 |
| scripts/extract.js | 46 |
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
  "cl_report": {"\u0027": {"f1-score": 1.0, "precision": 1.0, "recall": 1.0, "support": 6192}, "macro avg": {"f1-score": 0.7853442024489883, "precision": 0.8173737919617302, "recall": 0.7586909958988294, "support": 64560}, "micro avg": {"f1-score": 0.9482651796778191, "precision": 0.9482651796778191, "recall": 0.9482651796778191, "support": 64560}, "weighted avg": {"f1-score": 0.9465316482975635, "precision": 0.9466497299503976, "recall": 0.9482651796778191, "support": 64560}, "\u2205": {"f1-score": 0.9654236752328139, "precision": 0.9443156050298908, "recall": 0.9874969685537981, "support": 37111}, "\u23ce": {"f1-score": 0.8728017756530647, "precision": 0.9112299465240642, "recall": 0.8374836173001311, "support": 3052}, "\u23ce\u23ce": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 114}, "\u23ce\u2423\u207a\u2423\u207a": {"f1-score": 0.8528164748637189, "precision": 0.925706771860618, "recall": 0.7905670971364402, "support": 1781}, "\u23ce\u2423\u207b\u2423\u207b": {"f1-score": 0.8896041197296427, "precision": 0.9992769342010123, "recall": 0.8016241299303944, "support": 1724}, "\u2423": {"f1-score": 0.9167633716636776, "precision": 0.9410872861165259, "recall": 0.8936651583710408, "support": 14586}},
  "cl_report_full": {"\u0027": {"f1-score": 0.9954983922829581, "precision": 1.0, "recall": 0.9910371318822023, "support": 6248}, "macro avg": {"f1-score": 0.7625604820373706, "precision": 0.8173737919617302, "recall": 0.7186720600068727, "support": 68831}, "micro avg": {"f1-score": 0.9179030069494943, "precision": 0.9482651796778191, "recall": 0.8894248231174907, "support": 68831}, "weighted avg": {"f1-score": 0.9119203942304989, "precision": 0.9416572593004237, "recall": 0.8894248231174907, "support": 68831}, "\u2205": {"f1-score": 0.9558798596710879, "precision": 0.9443156050298908, "recall": 0.9677308616546516, "support": 37869}, "\u23ce": {"f1-score": 0.8517160946351217, "precision": 0.9112299465240642, "recall": 0.7994995308101345, "support": 3197}, "\u23ce\u23ce": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 457}, "\u23ce\u2423\u207a\u2423\u207a": {"f1-score": 0.8260486946318568, "precision": 0.925706771860618, "recall": 0.7457627118644068, "support": 1888}, "\u23ce\u2423\u207b\u2423\u207b": {"f1-score": 0.8744068332806075, "precision": 0.9992769342010123, "recall": 0.7772778402699663, "support": 1778}, "\u2423": {"f1-score": 0.8343734997599617, "precision": 0.9410872861165259, "recall": 0.7493963435667471, "support": 17394}},
  "ppcr": 0.9379494704420973
}
```
</details>
