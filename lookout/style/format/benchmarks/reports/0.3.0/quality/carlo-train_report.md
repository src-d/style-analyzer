# Train report for javascript / file:///tmp/top-repos-quality-repos-z21a9_wv/carlo HEAD b8ce2bca042c757b13fc82a3e059980342ddd9a8

### Classification report

PPCR: 0.667

| Class | Precision | Recall | Full Recall | F1-score | Full F1-score | Support | Full Support | PPCR |
|------:|:----------|:-------|:------------|:---------|:---------|:--------|:-------------|:-----|
| `∅` | 0.924| 0.995| 0.900| 0.958| 0.912| 3916| 4328| 0.905 |
| `␣` | 0.960| 0.748| 0.392| 0.841| 0.557| 1110| 2117| 0.524 |
| `'` | 1.000| 1.000| 0.494| 1.000| 0.661| 315| 638| 0.494 |
| `⏎` | 0.000| 0.000| 0.000| 0.000| 0.000| 24| 402| 0.060 |
| `⏎␣⁻␣⁻` | 0.000| 0.000| 0.000| 0.000| 0.000| 24| 209| 0.115 |
| `⏎␣⁺␣⁺` | 0.000| 0.000| 0.000| 0.000| 0.000| 6| 253| 0.024 |
| `⏎⏎` | 0.000| 0.000| 0.000| 0.000| 0.000| 3| 148| 0.020 |
| `"` | 0.000| 0.000| 0.000| 0.000| 0.000| 0| 0| 0.000 |
| `micro avg` | 0.934| 0.934| 0.623| 0.934| 0.747| 5398| 8095| 0.667 |
| `weighted avg` | 0.926| 0.934| 0.623| 0.926| 0.685| 5398| 8095| 0.667 |
| `macro avg` | 0.360| 0.343| 0.223| 0.350| 0.266| 5398| 8095| 0.667 |

### Confusion matrix

|refusal|  ∅| ␣| '| ⏎| ⏎␣⁺␣⁺| ⏎␣⁻␣⁻| "| ⏎⏎| 
|:---|:---|:---|:---|:---|:---|:---|:---|
|0 |0 |0 |0 |0 |0 |0 |0 |
|412 |3897 |19 |0 |0 |0 |0 |0 |
|1007 |280 |830 |0 |0 |0 |0 |0 |
|323 |0 |0 |315 |0 |0 |0 |0 |
|378 |15 |9 |0 |0 |0 |0 |0 |
|247 |3 |3 |0 |0 |0 |0 |0 |
|185 |20 |4 |0 |0 |0 |0 |0 |
|145 |3 |0 |0 |0 |0 |0 |0 |

### Files with most errors

| filename | number of errors|
|:----:|:-----|
| lib/color.js | 148 |
| lib/find-chrome.js | 59 |
| lib/rpc.js | 59 |
| lib/carlo.js | 46 |
| lib/intercepted_request.js | 18 |
| examples/terminal/main.js | 15 |
| examples/systeminfo/main.js | 10 |
| index.js | 1 |

<details>
    <summary>Machine-readable report</summary>
```json
{
  "cl_report": {"\"": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 0}, "\u0027": {"f1-score": 1.0, "precision": 1.0, "recall": 1.0, "support": 315}, "macro avg": {"f1-score": 0.3498383095803519, "precision": 0.36042939425583176, "recall": 0.34286198225804965, "support": 5398}, "micro avg": {"f1-score": 0.9340496480177843, "precision": 0.9340496480177843, "recall": 0.9340496480177843, "support": 5398}, "weighted avg": {"f1-score": 0.9263197115671717, "precision": 0.9259113811598798, "recall": 0.9340496480177843, "support": 5398}, "\u2205": {"f1-score": 0.958200147528891, "precision": 0.9238975817923186, "recall": 0.9951481103166496, "support": 3916}, "\u23ce": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 24}, "\u23ce\u23ce": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 3}, "\u23ce\u2423\u207a\u2423\u207a": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 6}, "\u23ce\u2423\u207b\u2423\u207b": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 24}, "\u2423": {"f1-score": 0.8405063291139241, "precision": 0.9595375722543352, "recall": 0.7477477477477478, "support": 1110}},
  "cl_report_full": {"\"": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 0}, "\u0027": {"f1-score": 0.6610703043022036, "precision": 1.0, "recall": 0.493730407523511, "support": 638}, "macro avg": {"f1-score": 0.266218661817469, "precision": 0.36042939425583176, "recall": 0.22327631823289665, "support": 8095}, "micro avg": {"f1-score": 0.7473504780256429, "precision": 0.9340496480177843, "recall": 0.6228536133415689, "support": 8095}, "weighted avg": {"f1-score": 0.6852873001753023, "precision": 0.8237146108041485, "recall": 0.6228536133415689, "support": 8095}, "\u2205": {"f1-score": 0.9120056166627662, "precision": 0.9238975817923186, "recall": 0.9004158964879853, "support": 4328}, "\u23ce": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 402}, "\u23ce\u23ce": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 148}, "\u23ce\u2423\u207a\u2423\u207a": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 253}, "\u23ce\u2423\u207b\u2423\u207b": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 209}, "\u2423": {"f1-score": 0.5566733735747821, "precision": 0.9595375722543352, "recall": 0.3920642418516769, "support": 2117}},
  "ppcr": 0.6668313773934528
}
```
</details>
