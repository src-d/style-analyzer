# Train report for javascript / file:///tmp/top-repos-quality-repos-dhds81z3/reveal.js HEAD 0b3e7839ebf4ed8b6c180aca0abafa28c67aee6d

### Classification report

PPCR: 0.679

| Class | Precision | Recall | Full Recall | F1-score | Full F1-score | Support | Full Support | PPCR |
|------:|:----------|:-------|:------------|:---------|:---------|:--------|:-------------|:-----|
| `∅` | 0.986| 0.967| 0.877| 0.976| 0.928| 5830| 6426| 0.907 |
| `␣` | 0.929| 0.982| 0.633| 0.955| 0.753| 3369| 5225| 0.645 |
| `⏎` | 0.987| 0.892| 0.473| 0.938| 0.639| 437| 825| 0.530 |
| `'` | 0.980| 1.000| 0.377| 0.990| 0.545| 385| 1021| 0.377 |
| `⏎⇥⁺` | 0.000| 0.000| 0.000| 0.000| 0.000| 16| 341| 0.047 |
| `⏎⇥⁻` | 0.000| 0.000| 0.000| 0.000| 0.000| 14| 318| 0.044 |
| `⏎⏎` | 0.000| 0.000| 0.000| 0.000| 0.000| 9| 440| 0.020 |
| `"` | 0.000| 0.000| 0.000| 0.000| 0.000| 8| 37| 0.216 |
| `⏎⏎⇥⁺` | 0.000| 0.000| 0.000| 0.000| 0.000| 0| 88| 0.000 |
| `⏎⏎⇥⁻` | 0.000| 0.000| 0.000| 0.000| 0.000| 0| 108| 0.000 |
| `micro avg` | 0.965| 0.965| 0.655| 0.965| 0.781| 10068| 14829| 0.679 |
| `weighted avg` | 0.962| 0.965| 0.655| 0.963| 0.741| 10068| 14829| 0.679 |
| `macro avg` | 0.388| 0.384| 0.236| 0.386| 0.287| 10068| 14829| 0.679 |

### Confusion matrix

|refusal|  ∅| ␣| ⏎| '| ⏎⏎| "| ⏎⇥⁻| ⏎⇥⁺| ⏎⏎⇥⁺| ⏎⏎⇥⁻| 
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
|0 |0 |0 |0 |0 |0 |0 |0 |0 |0 |0 |
|596 |5637 |193 |0 |0 |0 |0 |0 |0 |0 |0 |
|1856 |61 |3308 |0 |0 |0 |0 |0 |0 |0 |0 |
|388 |7 |40 |390 |0 |0 |0 |0 |0 |0 |0 |
|636 |0 |0 |0 |385 |0 |0 |0 |0 |0 |0 |
|431 |0 |4 |5 |0 |0 |0 |0 |0 |0 |0 |
|29 |0 |0 |0 |8 |0 |0 |0 |0 |0 |0 |
|304 |14 |0 |0 |0 |0 |0 |0 |0 |0 |0 |
|325 |0 |16 |0 |0 |0 |0 |0 |0 |0 |0 |
|88 |0 |0 |0 |0 |0 |0 |0 |0 |0 |0 |
|108 |0 |0 |0 |0 |0 |0 |0 |0 |0 |0 |

### Files with most errors

| filename | number of errors|
|:----:|:-----|
| js/reveal.js | 170 |
| plugin/markdown/markdown.js | 37 |
| plugin/zoom-js/zoom.js | 25 |
| plugin/remotes/remotes.js | 22 |
| plugin/notes-server/index.js | 20 |
| plugin/multiplex/index.js | 19 |
| plugin/notes/notes.js | 18 |
| plugin/notes-server/client.js | 13 |
| Gruntfile.js | 9 |
| plugin/multiplex/master.js | 7 |

<details>
    <summary>Machine-readable report</summary>
```json
{
  "cl_report": {"\"": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 8}, "\u0027": {"f1-score": 0.9897172236503856, "precision": 0.9796437659033079, "recall": 1.0, "support": 385}, "macro avg": {"f1-score": 0.38580953929214895, "precision": 0.3881599908467238, "recall": 0.3841237618381924, "support": 10068}, "micro avg": {"f1-score": 0.965435041716329, "precision": 0.965435041716329, "recall": 0.965435041716329, "support": 10068}, "weighted avg": {"f1-score": 0.9632754143488722, "precision": 0.9619269745101651, "recall": 0.965435041716329, "support": 10068}, "\u2205": {"f1-score": 0.976188414581349, "precision": 0.9856618289910823, "recall": 0.9668953687821612, "support": 5830}, "\u23ce": {"f1-score": 0.9375000000000001, "precision": 0.9873417721518988, "recall": 0.8924485125858124, "support": 437}, "\u23ce\u21e5\u207a": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 16}, "\u23ce\u21e5\u207b": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 14}, "\u23ce\u23ce": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 9}, "\u23ce\u23ce\u21e5\u207a": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 0}, "\u23ce\u23ce\u21e5\u207b": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 0}, "\u2423": {"f1-score": 0.9546897546897548, "precision": 0.9289525414209492, "recall": 0.9818937370139508, "support": 3369}},
  "cl_report_full": {"\"": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 37}, "\u0027": {"f1-score": 0.5445544554455446, "precision": 0.9796437659033079, "recall": 0.3770812928501469, "support": 1021}, "macro avg": {"f1-score": 0.28651981239500446, "precision": 0.3881599908467238, "recall": 0.23601361671124516, "support": 14829}, "micro avg": {"f1-score": 0.780816965899506, "precision": 0.965435041716329, "recall": 0.6554723851911795, "support": 14829}, "weighted avg": {"f1-score": 0.7406505286298304, "precision": 0.8768233319194652, "recall": 0.6554723851911795, "support": 14829}, "\u2205": {"f1-score": 0.9282832441333883, "precision": 0.9856618289910823, "recall": 0.8772175536881419, "support": 6426}, "\u23ce": {"f1-score": 0.639344262295082, "precision": 0.9873417721518988, "recall": 0.4727272727272727, "support": 825}, "\u23ce\u21e5\u207a": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 341}, "\u23ce\u21e5\u207b": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 318}, "\u23ce\u23ce": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 440}, "\u23ce\u23ce\u21e5\u207a": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 88}, "\u23ce\u23ce\u21e5\u207b": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 108}, "\u2423": {"f1-score": 0.7530161620760301, "precision": 0.9289525414209492, "recall": 0.63311004784689, "support": 5225}},
  "ppcr": 0.6789399150313574
}
```
</details>
