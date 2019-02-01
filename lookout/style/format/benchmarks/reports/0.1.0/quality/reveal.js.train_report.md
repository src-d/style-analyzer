# Train report for javascript / file:///tmp/top-repos-quality-repos-_5k_hmxx/reveal.js HEAD 0b3e7839ebf4ed8b6c180aca0abafa28c67aee6d

### Classification report

PPCR: 0.997

| Class | Precision | Recall | Full Recall | F1-score | Full F1-score | Support | Full Support | PPCR |
|------:|:----------|:-------|:------------|:---------|:---------|:--------|:-------------|:-----|
| `∅` | 0.983| 0.982| 0.978| 0.982| 0.980| 5514| 5534| 0.996 |
| `␣` | 0.955| 0.973| 0.973| 0.964| 0.964| 3126| 3126| 1.000 |
| `⏎` | 0.992| 0.919| 0.917| 0.954| 0.953| 422| 423| 0.998 |
| `'` | 1.000| 1.000| 1.000| 1.000| 1.000| 79| 79| 1.000 |
| `⏎⇥⁺` | 0.000| 0.000| 0.000| 0.000| 0.000| 13| 13| 1.000 |
| `⏎⏎` | 0.000| 0.000| 0.000| 0.000| 0.000| 9| 11| 0.818 |
| `⏎⇥⁻` | 0.000| 0.000| 0.000| 0.000| 0.000| 0| 0| 0.000 |
| `␣'` | 0.000| 0.000| 0.000| 0.000| 0.000| 0| 0| 0.000 |
| `"` | 0.000| 0.000| 0.000| 0.000| 0.000| 0| 1| 0.000 |
| `'␣` | 0.000| 0.000| 0.000| 0.000| 0.000| 0| 0| 0.000 |
| `␣"` | 0.000| 0.000| 0.000| 0.000| 0.000| 0| 0| 0.000 |
| `⏎⏎⇥⁺` | 0.000| 0.000| 0.000| 0.000| 0.000| 0| 0| 0.000 |
| `⏎⏎⇥⁻` | 0.000| 0.000| 0.000| 0.000| 0.000| 0| 0| 0.000 |
| `"␣` | 0.000| 0.000| 0.000| 0.000| 0.000| 0| 7| 0.000 |
| `macro avg` | 0.281| 0.277| 0.276| 0.279| 0.278| 9163| 9194| 0.997 |
| `weighted avg` | 0.972| 0.974| 0.970| 0.973| 0.970| 9163| 9194| 0.997 |
| `micro avg` | 0.974| 0.974| 0.970| 0.974| 0.972| 9163| 9194| 0.997 |

### Confusion matrix

|refusal|  ∅| ␣| ⏎| ⏎⏎| ⏎⇥⁺| ⏎⇥⁻| ␣'| "| '| '␣| ␣"| ⏎⏎⇥⁺| ⏎⏎⇥⁻| "␣| 
|:---|:---|:---|:---|:---|:---|:---|:---|:---|
|0 |0 |0 |0 |0 |0 |0 |0 |0 |
|20 |5412 |102 |0 |0 |0 |0 |0 |0 |
|0 |83 |3043 |0 |0 |0 |0 |0 |0 |
|1 |12 |22 |388 |0 |0 |0 |0 |0 |
|2 |1 |5 |3 |0 |0 |0 |0 |0 |
|0 |0 |13 |0 |0 |0 |0 |0 |0 |
|1 |0 |0 |0 |0 |0 |0 |0 |0 |
|0 |0 |0 |0 |0 |0 |0 |79 |0 |
|7 |0 |0 |0 |0 |0 |0 |0 |0 |

### Files with most errors

| filename | number of errors|
|:----:|:-----|
| js/reveal.js | 104 |
| plugin/markdown/markdown.js | 27 |
| plugin/remotes/remotes.js | 22 |
| plugin/notes/notes.js | 17 |
| plugin/notes-server/client.js | 14 |
| plugin/zoom-js/zoom.js | 13 |
| plugin/multiplex/index.js | 11 |
| plugin/multiplex/master.js | 10 |
| Gruntfile.js | 9 |
| plugin/notes-server/index.js | 5 |

<details>
    <summary>Machine-readable report</summary>
```json
{
  "cl_report": {"\"": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 0}, "\"\u2423": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 0}, "\u0027": {"f1-score": 1.0, "precision": 1.0, "recall": 1.0, "support": 79}, "\u0027\u2423": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 0}, "macro avg": {"f1-score": 0.2786338169222689, "precision": 0.2807367274562848, "recall": 0.27674152916506445, "support": 9163}, "micro avg": {"f1-score": 0.9736985703372258, "precision": 0.9736985703372258, "recall": 0.9736985703372258, "support": 9163}, "weighted avg": {"f1-score": 0.9725300044522612, "precision": 0.9715473129360714, "recall": 0.9736985703372258, "support": 9163}, "\u2205": {"f1-score": 0.9820359281437127, "precision": 0.9825708061002179, "recall": 0.9815016322089227, "support": 5514}, "\u23ce": {"f1-score": 0.9544895448954489, "precision": 0.9923273657289002, "recall": 0.919431279620853, "support": 422}, "\u23ce\u21e5\u207a": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 13}, "\u23ce\u21e5\u207b": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 0}, "\u23ce\u23ce": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 9}, "\u23ce\u23ce\u21e5\u207a": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 0}, "\u23ce\u23ce\u21e5\u207b": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 0}, "\u2423": {"f1-score": 0.9643479638726034, "precision": 0.9554160125588697, "recall": 0.9734484964811261, "support": 3126}, "\u2423\"": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 0}, "\u2423\u0027": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 0}},
  "cl_report_full": {"\"": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 1}, "\"\u2423": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 7}, "\u0027": {"f1-score": 1.0, "precision": 1.0, "recall": 1.0, "support": 79}, "\u0027\u2423": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 0}, "macro avg": {"f1-score": 0.2784230083551575, "precision": 0.2807367274562848, "recall": 0.2763329030724235, "support": 9194}, "micro avg": {"f1-score": 0.9720542572315739, "precision": 0.9736985703372258, "recall": 0.9704154883619752, "support": 9194}, "weighted avg": {"f1-score": 0.9703663421705465, "precision": 0.9705168340135912, "recall": 0.9704154883619752, "support": 9194}, "\u2205": {"f1-score": 0.9802571997826481, "precision": 0.9825708061002179, "recall": 0.9779544633176726, "support": 5534}, "\u23ce": {"f1-score": 0.9533169533169532, "precision": 0.9923273657289002, "recall": 0.91725768321513, "support": 423}, "\u23ce\u21e5\u207a": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 13}, "\u23ce\u21e5\u207b": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 0}, "\u23ce\u23ce": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 11}, "\u23ce\u23ce\u21e5\u207a": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 0}, "\u23ce\u23ce\u21e5\u207b": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 0}, "\u2423": {"f1-score": 0.9643479638726034, "precision": 0.9554160125588697, "recall": 0.9734484964811261, "support": 3126}, "\u2423\"": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 0}, "\u2423\u0027": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 0}},
  "ppcr": 0.9966282358059604
}
```
</details>
