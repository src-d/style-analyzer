# Train report for javascript / file:///tmp/top-repos-quality-repos-apbl1pxu/citgm HEAD 0c4c7ccdd1cad8ce9506e34ca523787ba18cafe2

### Classification report

PPCR: 0.857

| Class | Precision | Recall | Full Recall | F1-score | Full F1-score | Support | Full Support | PPCR |
|------:|:----------|:-------|:------------|:---------|:---------|:--------|:-------------|:-----|
| `∅` | 0.947| 0.997| 0.987| 0.971| 0.966| 11695| 11813| 0.990 |
| `␣` | 0.975| 0.931| 0.680| 0.952| 0.801| 4074| 5577| 0.731 |
| `'` | 1.000| 1.000| 0.970| 1.000| 0.985| 3157| 3256| 0.970 |
| `⏎␣⁺␣⁺` | 0.977| 0.899| 0.887| 0.936| 0.930| 762| 772| 0.987 |
| `⏎␣⁻␣⁻` | 1.000| 0.744| 0.642| 0.853| 0.782| 624| 723| 0.863 |
| `⏎` | 0.000| 0.000| 0.000| 0.000| 0.000| 116| 1361| 0.085 |
| `⏎⏎` | 0.000| 0.000| 0.000| 0.000| 0.000| 95| 442| 0.215 |
| `micro avg` | 0.962| 0.962| 0.825| 0.962| 0.888| 20523| 23944| 0.857 |
| `weighted avg` | 0.953| 0.962| 0.825| 0.957| 0.851| 20523| 23944| 0.857 |
| `macro avg` | 0.700| 0.653| 0.595| 0.673| 0.638| 20523| 23944| 0.857 |

### Confusion matrix

|refusal|  ∅| ␣| '| ⏎| ⏎␣⁺␣⁺| ⏎␣⁻␣⁻| ⏎⏎| 
|:---|:---|:---|:---|:---|:---|:---|:---|
|0 |0 |0 |0 |0 |0 |0 |0 |
|118 |11655 |36 |0 |0 |4 |0 |0 |
|1503 |273 |3792 |0 |0 |9 |0 |0 |
|99 |0 |0 |3157 |0 |0 |0 |0 |
|1245 |90 |25 |0 |0 |1 |0 |0 |
|10 |39 |38 |0 |0 |685 |0 |0 |
|99 |158 |0 |0 |0 |2 |464 |0 |
|347 |95 |0 |0 |0 |0 |0 |0 |

### Files with most errors

| filename | number of errors|
|:----:|:-----|
| bin/citgm-all.js | 62 |
| lib/lookup.js | 57 |
| test/test-lookup.js | 49 |
| lib/match-conditions.js | 39 |
| lib/citgm.js | 38 |
| test/test-check-tags.js | 35 |
| bin/citgm.js | 34 |
| lib/npm/test.js | 30 |
| lib/out.js | 26 |
| lib/reporter/tap.js | 26 |

<details>
    <summary>Machine-readable report</summary>
```json
{
  "cl_report": {"\u0027": {"f1-score": 1.0, "precision": 1.0, "recall": 1.0, "support": 3157}, "macro avg": {"f1-score": 0.6732266555682006, "precision": 0.6997890513578712, "recall": 0.6528428813427622, "support": 20523}, "micro avg": {"f1-score": 0.9624811187448229, "precision": 0.9624811187448229, "recall": 0.9624811187448229, "support": 20523}, "weighted avg": {"f1-score": 0.9568933619046472, "precision": 0.9534997305152593, "recall": 0.9624811187448229, "support": 20523}, "\u2205": {"f1-score": 0.9710476983961674, "precision": 0.946791226645004, "recall": 0.9965797349294571, "support": 11695}, "\u23ce": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 116}, "\u23ce\u23ce": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 95}, "\u23ce\u2423\u207a\u2423\u207a": {"f1-score": 0.9364319890635681, "precision": 0.9771754636233951, "recall": 0.8989501312335958, "support": 762}, "\u23ce\u2423\u207b\u2423\u207b": {"f1-score": 0.8529411764705882, "precision": 1.0, "recall": 0.7435897435897436, "support": 624}, "\u2423": {"f1-score": 0.952165725047081, "precision": 0.9745566692367, "recall": 0.930780559646539, "support": 4074}},
  "cl_report_full": {"\u0027": {"f1-score": 0.9845626072041166, "precision": 1.0, "recall": 0.9695945945945946, "support": 3256}, "macro avg": {"f1-score": 0.6376788307084179, "precision": 0.6997890513578712, "recall": 0.595033007016445, "support": 23944}, "micro avg": {"f1-score": 0.8884341196842602, "precision": 0.9624811187448229, "recall": 0.8249665887069829, "support": 23944}, "weighted avg": {"f1-score": 0.8507821057233983, "precision": 0.891786116046098, "recall": 0.8249665887069829, "support": 23944}, "\u2205": {"f1-score": 0.9662977241636613, "precision": 0.946791226645004, "recall": 0.9866249047659358, "support": 11813}, "\u23ce": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 1361}, "\u23ce\u23ce": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 442}, "\u23ce\u2423\u207a\u2423\u207a": {"f1-score": 0.9300746775288528, "precision": 0.9771754636233951, "recall": 0.8873056994818653, "support": 772}, "\u23ce\u2423\u207b\u2423\u207b": {"f1-score": 0.7818028643639428, "precision": 1.0, "recall": 0.6417704011065007, "support": 723}, "\u2423": {"f1-score": 0.8010139416983523, "precision": 0.9745566692367, "recall": 0.6799354491662184, "support": 5577}},
  "ppcr": 0.8571249582358837
}
```
</details>
