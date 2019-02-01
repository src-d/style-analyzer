# Train report for javascript / file:///tmp/top-repos-quality-repos-7xvmh62d/carlo HEAD b8ce2bca042c757b13fc82a3e059980342ddd9a8

### Classification report

PPCR: 0.951

| Class | Precision | Recall | Full Recall | F1-score | Full F1-score | Support | Full Support | PPCR |
|------:|:----------|:-------|:------------|:---------|:---------|:--------|:-------------|:-----|
| `∅` | 0.948| 0.995| 0.985| 0.971| 0.966| 4092| 4133| 0.990 |
| `␣` | 0.953| 0.850| 0.850| 0.899| 0.899| 1271| 1271| 1.000 |
| `'` | 1.000| 1.000| 0.428| 1.000| 0.599| 74| 173| 0.428 |
| `⏎␣⁺␣⁺` | 0.000| 0.000| 0.000| 0.000| 0.000| 24| 24| 1.000 |
| `⏎␣⁻␣⁻` | 0.000| 0.000| 0.000| 0.000| 0.000| 21| 21| 1.000 |
| `⏎` | 0.000| 0.000| 0.000| 0.000| 0.000| 17| 17| 1.000 |
| `"` | 0.000| 0.000| 0.000| 0.000| 0.000| 0| 0| 0.000 |
| `⏎⏎` | 0.000| 0.000| 0.000| 0.000| 0.000| 0| 0| 0.000 |
| `␣'` | 0.000| 0.000| 0.000| 0.000| 0.000| 0| 143| 0.000 |
| `macro avg` | 0.322| 0.316| 0.251| 0.319| 0.274| 5499| 5782| 0.951 |
| `weighted avg` | 0.939| 0.950| 0.903| 0.944| 0.906| 5499| 5782| 0.951 |
| `micro avg` | 0.950| 0.950| 0.903| 0.950| 0.926| 5499| 5782| 0.951 |

### Confusion matrix

|refusal|  ∅| ␣| '| ⏎| ⏎␣⁺␣⁺| ⏎␣⁻␣⁻| "| ⏎⏎| ␣'| 
|:---|:---|:---|:---|:---|:---|:---|:---|
|0 |0 |0 |0 |0 |0 |0 |0 |
|41 |4070 |22 |0 |0 |0 |0 |0 |
|0 |191 |1080 |0 |0 |0 |0 |0 |
|99 |0 |0 |74 |0 |0 |0 |0 |
|0 |8 |9 |0 |0 |0 |0 |0 |
|0 |4 |20 |0 |0 |0 |0 |0 |
|0 |19 |2 |0 |0 |0 |0 |0 |
|143 |0 |0 |0 |0 |0 |0 |0 |

### Files with most errors

| filename | number of errors|
|:----:|:-----|
| lib/color.js | 114 |
| lib/find-chrome.js | 50 |
| lib/carlo.js | 42 |
| lib/rpc.js | 42 |
| examples/terminal/main.js | 12 |
| lib/intercepted_request.js | 8 |
| examples/systeminfo/main.js | 7 |

<details>
    <summary>Machine-readable report</summary>
```json
{
  "cl_report": {"\"": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 0}, "\u0027": {"f1-score": 1.0, "precision": 1.0, "recall": 1.0, "support": 74}, "macro avg": {"f1-score": 0.3188221602672391, "precision": 0.322388599757197, "recall": 0.3160386980213888, "support": 5499}, "micro avg": {"f1-score": 0.949990907437716, "precision": 0.949990907437716, "recall": 0.949990907437716, "support": 5499}, "weighted avg": {"f1-score": 0.9436091975916511, "precision": 0.9394234223529966, "recall": 0.949990907437716, "support": 5499}, "\u2205": {"f1-score": 0.9708969465648855, "precision": 0.9482758620689655, "recall": 0.9946236559139785, "support": 4092}, "\u23ce": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 17}, "\u23ce\u23ce": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 0}, "\u23ce\u2423\u207a\u2423\u207a": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 24}, "\u23ce\u2423\u207b\u2423\u207b": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 21}, "\u2423": {"f1-score": 0.8985024958402662, "precision": 0.9532215357458076, "recall": 0.8497246262785209, "support": 1271}, "\u2423\u0027": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 0}},
  "cl_report_full": {"\"": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 0}, "\u0027": {"f1-score": 0.5991902834008098, "precision": 1.0, "recall": 0.4277456647398844, "support": 173}, "macro avg": {"f1-score": 0.27376276511844466, "precision": 0.322388599757197, "recall": 0.25135856958300584, "support": 5782}, "micro avg": {"f1-score": 0.9261590284549241, "precision": 0.949990907437716, "recall": 0.9034936008301626, "support": 5782}, "weighted avg": {"f1-score": 0.9060612087078411, "precision": 0.9172896419688613, "recall": 0.9034936008301626, "support": 5782}, "\u2205": {"f1-score": 0.9661721068249258, "precision": 0.9482758620689655, "recall": 0.9847568352286474, "support": 4133}, "\u23ce": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 17}, "\u23ce\u23ce": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 0}, "\u23ce\u2423\u207a\u2423\u207a": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 24}, "\u23ce\u2423\u207b\u2423\u207b": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 21}, "\u2423": {"f1-score": 0.8985024958402662, "precision": 0.9532215357458076, "recall": 0.8497246262785209, "support": 1271}, "\u2423\u0027": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 143}},
  "ppcr": 0.9510549982704947
}
```
</details>
