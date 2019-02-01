# Train report for javascript / file:///tmp/top-repos-quality-repos-1qjo6yjz/30-seconds-of-code HEAD 3a122c9cfcbdc091227879a06a32bc67ccd0d35d

### Classification report

PPCR: 0.954

| Class | Precision | Recall | Full Recall | F1-score | Full F1-score | Support | Full Support | PPCR |
|------:|:----------|:-------|:------------|:---------|:---------|:--------|:-------------|:-----|
| `∅` | 0.947| 0.988| 0.975| 0.967| 0.961| 37957| 38475| 0.987 |
| `␣` | 0.943| 0.898| 0.770| 0.920| 0.848| 14373| 16763| 0.857 |
| `'` | 1.000| 1.000| 1.000| 1.000| 1.000| 5216| 5216| 1.000 |
| `⏎` | 0.910| 0.831| 0.810| 0.869| 0.857| 3105| 3184| 0.975 |
| `⏎␣⁺␣⁺` | 0.974| 0.801| 0.743| 0.879| 0.843| 1771| 1909| 0.928 |
| `⏎␣⁻␣⁻` | 0.999| 0.797| 0.795| 0.887| 0.886| 1746| 1749| 0.998 |
| `⏎⏎` | 0.851| 0.752| 0.752| 0.798| 0.798| 455| 455| 1.000 |
| `␣'` | 0.000| 0.000| 0.000| 0.000| 0.000| 0| 0| 0.000 |
| `macro avg` | 0.828| 0.758| 0.731| 0.790| 0.774| 64623| 67751| 0.954 |
| `weighted avg` | 0.950| 0.950| 0.906| 0.949| 0.925| 64623| 67751| 0.954 |
| `micro avg` | 0.950| 0.950| 0.906| 0.950| 0.927| 64623| 67751| 0.954 |

### Confusion matrix

|refusal|  ∅| ␣| '| ⏎| ⏎␣⁺␣⁺| ⏎␣⁻␣⁻| ␣'| ⏎⏎| 
|:---|:---|:---|:---|:---|:---|:---|:---|
|0 |0 |0 |0 |0 |0 |0 |0 |
|518 |37520 |415 |0 |0 |22 |0 |0 |
|2390 |1429 |12909 |0 |19 |16 |0 |0 |
|0 |0 |0 |5216 |0 |0 |0 |0 |
|79 |185 |285 |0 |2580 |0 |0 |55 |
|138 |280 |72 |0 |0 |1418 |1 |0 |
|3 |211 |7 |0 |132 |0 |1391 |5 |
|0 |10 |0 |0 |103 |0 |0 |342 |

### Files with most errors

| filename | number of errors|
|:----:|:-----|
| scripts/web.js | 217 |
| scripts/util.js | 102 |
| scripts/tdd.js | 56 |
| scripts/extract.js | 49 |
| test/uniqueElementsBy/uniqueElementsBy.test.js | 47 |
| test/uniqueElementsByRight/uniqueElementsByRight.test.js | 47 |
| scripts/rollup.js | 41 |
| test/filterNonUniqueBy/filterNonUniqueBy.test.js | 40 |
| scripts/localize.js | 40 |
| scripts/tag.js | 39 |

<details>
    <summary>Machine-readable report</summary>
```json
{
  "cl_report": {"\u0027": {"f1-score": 1.0, "precision": 1.0, "recall": 1.0, "support": 5216}, "macro avg": {"f1-score": 0.7899412427911112, "precision": 0.8280037394676891, "recall": 0.7583189066363996, "support": 64623}, "micro avg": {"f1-score": 0.949754731287622, "precision": 0.949754731287622, "recall": 0.949754731287622, "support": 64623}, "weighted avg": {"f1-score": 0.948794256565494, "precision": 0.9499076585407242, "recall": 0.949754731287622, "support": 64623}, "\u2205": {"f1-score": 0.9671100113413754, "precision": 0.946638072410748, "recall": 0.9884869721000079, "support": 37957}, "\u23ce": {"f1-score": 0.8688331368917326, "precision": 0.9103740296400847, "recall": 0.8309178743961353, "support": 3105}, "\u23ce\u23ce": {"f1-score": 0.7981330221703618, "precision": 0.8507462686567164, "recall": 0.7516483516483516, "support": 455}, "\u23ce\u2423\u207a\u2423\u207a": {"f1-score": 0.8788348311124885, "precision": 0.9739010989010989, "recall": 0.8006775832862789, "support": 1771}, "\u23ce\u2423\u207b\u2423\u207b": {"f1-score": 0.8865519439133205, "precision": 0.9992816091954023, "recall": 0.7966781214203894, "support": 1746}, "\u2423": {"f1-score": 0.9200669968996116, "precision": 0.9430888369374635, "recall": 0.8981423502400334, "support": 14373}, "\u2423\u0027": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 0}},
  "cl_report_full": {"\u0027": {"f1-score": 1.0, "precision": 1.0, "recall": 1.0, "support": 5216}, "macro avg": {"f1-score": 0.774076221692801, "precision": 0.8280037394676891, "recall": 0.7306657894470128, "support": 67751}, "micro avg": {"f1-score": 0.9273120099113118, "precision": 0.949754731287622, "recall": 0.905905447890068, "support": 67751}, "weighted avg": {"f1-score": 0.9245998279597755, "precision": 0.9496470783380914, "recall": 0.905905447890068, "support": 67751}, "\u2205": {"f1-score": 0.960696453719114, "precision": 0.946638072410748, "recall": 0.9751786874593892, "support": 38475}, "\u23ce": {"f1-score": 0.8574277168494516, "precision": 0.9103740296400847, "recall": 0.8103015075376885, "support": 3184}, "\u23ce\u23ce": {"f1-score": 0.7981330221703618, "precision": 0.8507462686567164, "recall": 0.7516483516483516, "support": 455}, "\u23ce\u2423\u207a\u2423\u207a": {"f1-score": 0.8427934621099553, "precision": 0.9739010989010989, "recall": 0.7427972760607648, "support": 1909}, "\u23ce\u2423\u207b\u2423\u207b": {"f1-score": 0.8857051894301178, "precision": 0.9992816091954023, "recall": 0.7953116066323613, "support": 1749}, "\u2423": {"f1-score": 0.8478539292634069, "precision": 0.9430888369374635, "recall": 0.770088886237547, "support": 16763}, "\u2423\u0027": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 0}},
  "ppcr": 0.9538309397647268
}
```
</details>
