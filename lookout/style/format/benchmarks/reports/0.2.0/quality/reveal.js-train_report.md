# Train report for javascript / file:///tmp/top-repos-quality-repos-6adtmc4j/reveal.js HEAD 0b3e7839ebf4ed8b6c180aca0abafa28c67aee6d

### Classification report

PPCR: 0.685

| Class | Precision | Recall | Full Recall | F1-score | Full F1-score | Support | Full Support | PPCR |
|------:|:----------|:-------|:------------|:---------|:---------|:--------|:-------------|:-----|
| `∅` | 0.986| 0.983| 0.889| 0.984| 0.935| 5810| 6426| 0.904 |
| `␣` | 0.956| 0.983| 0.656| 0.969| 0.778| 3490| 5225| 0.668 |
| `⏎` | 0.992| 0.894| 0.473| 0.941| 0.640| 436| 825| 0.528 |
| `'` | 0.980| 1.000| 0.377| 0.990| 0.545| 385| 1021| 0.377 |
| `⏎⇥⁺` | 0.000| 0.000| 0.000| 0.000| 0.000| 15| 341| 0.044 |
| `⏎⇥⁻` | 0.000| 0.000| 0.000| 0.000| 0.000| 14| 318| 0.044 |
| `"` | 0.000| 0.000| 0.000| 0.000| 0.000| 8| 37| 0.216 |
| `⏎⏎` | 0.000| 0.000| 0.000| 0.000| 0.000| 7| 440| 0.016 |
| `⏎⏎⇥⁺` | 0.000| 0.000| 0.000| 0.000| 0.000| 0| 88| 0.000 |
| `⏎⏎⇥⁻` | 0.000| 0.000| 0.000| 0.000| 0.000| 0| 108| 0.000 |
| `micro avg` | 0.975| 0.975| 0.669| 0.975| 0.793| 10165| 14829| 0.685 |
| `macro avg` | 0.391| 0.386| 0.239| 0.388| 0.290| 10165| 14829| 0.685 |
| `weighted avg` | 0.971| 0.975| 0.669| 0.973| 0.752| 10165| 14829| 0.685 |

### Confusion matrix

|refusal|  ∅| ␣| ⏎| '| ⏎⏎| "| ⏎⇥⁻| ⏎⇥⁺| ⏎⏎⇥⁺| ⏎⏎⇥⁻| 
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
|0 |0 |0 |0 |0 |0 |0 |0 |0 |0 |0 |
|616 |5711 |99 |0 |0 |0 |0 |0 |0 |0 |0 |
|1735 |61 |3429 |0 |0 |0 |0 |0 |0 |0 |0 |
|389 |7 |39 |390 |0 |0 |0 |0 |0 |0 |0 |
|636 |0 |0 |0 |385 |0 |0 |0 |0 |0 |0 |
|433 |0 |4 |3 |0 |0 |0 |0 |0 |0 |0 |
|29 |0 |0 |0 |8 |0 |0 |0 |0 |0 |0 |
|304 |14 |0 |0 |0 |0 |0 |0 |0 |0 |0 |
|326 |0 |15 |0 |0 |0 |0 |0 |0 |0 |0 |
|88 |0 |0 |0 |0 |0 |0 |0 |0 |0 |0 |
|108 |0 |0 |0 |0 |0 |0 |0 |0 |0 |0 |

### Files with most errors

| filename | number of errors|
|:----:|:-----|
| js/reveal.js | 114 |
| plugin/markdown/markdown.js | 22 |
| plugin/notes-server/index.js | 19 |
| plugin/multiplex/index.js | 17 |
| plugin/remotes/remotes.js | 15 |
| plugin/notes/notes.js | 14 |
| plugin/notes-server/client.js | 13 |
| plugin/zoom-js/zoom.js | 13 |
| Gruntfile.js | 9 |
| plugin/multiplex/master.js | 7 |

<details>
    <summary>Machine-readable report</summary>
```json
{
  "cl_report": {"\"": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 8}, "\u0027": {"f1-score": 0.9897172236503856, "precision": 0.9796437659033079, "recall": 1.0, "support": 385}, "macro avg": {"f1-score": 0.38842020851340997, "precision": 0.3914073791441936, "recall": 0.38599773158962786, "support": 10165}, "micro avg": {"f1-score": 0.9754058042302016, "precision": 0.9754058042302016, "recall": 0.9754058042302016, "support": 10165}, "weighted avg": {"f1-score": 0.9732529787965598, "precision": 0.9714507606552621, "recall": 0.9754058042302016, "support": 10165}, "\u2205": {"f1-score": 0.9844005860553305, "precision": 0.985844985327119, "recall": 0.982960413080895, "support": 5810}, "\u23ce": {"f1-score": 0.9408926417370325, "precision": 0.9923664122137404, "recall": 0.8944954128440367, "support": 436}, "\u23ce\u21e5\u207a": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 15}, "\u23ce\u21e5\u207b": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 14}, "\u23ce\u23ce": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 7}, "\u23ce\u23ce\u21e5\u207a": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 0}, "\u23ce\u23ce\u21e5\u207b": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 0}, "\u2423": {"f1-score": 0.969191633691351, "precision": 0.9562186279977691, "recall": 0.9825214899713467, "support": 3490}},
  "cl_report_full": {"\"": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 37}, "\u0027": {"f1-score": 0.5445544554455446, "precision": 0.9796437659033079, "recall": 0.3770812928501469, "support": 1021}, "macro avg": {"f1-score": 0.28980675074406675, "precision": 0.3914073791441936, "recall": 0.23948097792473638, "support": 14829}, "micro avg": {"f1-score": 0.7933904136992878, "precision": 0.9754058042302016, "recall": 0.6686222941533482, "support": 14829}, "weighted avg": {"f1-score": 0.7524462225998702, "precision": 0.8867894518891378, "recall": 0.6686222941533482, "support": 14829}, "\u2205": {"f1-score": 0.9347737130698093, "precision": 0.985844985327119, "recall": 0.8887332710862123, "support": 6426}, "\u23ce": {"f1-score": 0.6403940886699507, "precision": 0.9923664122137404, "recall": 0.4727272727272727, "support": 825}, "\u23ce\u21e5\u207a": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 341}, "\u23ce\u21e5\u207b": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 318}, "\u23ce\u23ce": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 440}, "\u23ce\u23ce\u21e5\u207a": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 88}, "\u23ce\u23ce\u21e5\u207b": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 108}, "\u2423": {"f1-score": 0.7783452502553627, "precision": 0.9562186279977691, "recall": 0.6562679425837321, "support": 5225}},
  "ppcr": 0.6854811517971542
}
```
</details>
