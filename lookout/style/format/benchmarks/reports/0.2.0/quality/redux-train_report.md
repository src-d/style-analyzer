# Train report for javascript / file:///tmp/top-repos-quality-repos-lui79obx/redux HEAD 902484ed735d38aec06683c847810a7218d8dba2

### Classification report

PPCR: 0.590

| Class | Precision | Recall | Full Recall | F1-score | Full F1-score | Support | Full Support | PPCR |
|------:|:----------|:-------|:------------|:---------|:---------|:--------|:-------------|:-----|
| `∅` | 0.966| 1.000| 0.893| 0.983| 0.928| 17217| 19278| 0.893 |
| `'` | 1.000| 1.000| 0.967| 1.000| 0.983| 2853| 2951| 0.967 |
| `␣` | 0.992| 0.940| 0.227| 0.965| 0.369| 2707| 11219| 0.241 |
| `⏎` | 0.913| 0.951| 0.137| 0.932| 0.239| 309| 2142| 0.144 |
| `⏎␣⁻␣⁻` | 0.000| 0.000| 0.000| 0.000| 0.000| 243| 1476| 0.165 |
| `⏎␣⁺␣⁺` | 0.000| 0.000| 0.000| 0.000| 0.000| 181| 1635| 0.111 |
| `⏎⏎` | 0.000| 0.000| 0.000| 0.000| 0.000| 28| 1065| 0.026 |
| `⏎␣⁻␣⁻␣⁻␣⁻` | 0.000| 0.000| 0.000| 0.000| 0.000| 16| 77| 0.208 |
| `"` | 0.000| 0.000| 0.000| 0.000| 0.000| 0| 50| 0.000 |
| `micro avg` | 0.972| 0.972| 0.574| 0.972| 0.722| 23554| 39893| 0.590 |
| `macro avg` | 0.430| 0.432| 0.247| 0.431| 0.280| 23554| 39893| 0.590 |
| `weighted avg` | 0.953| 0.972| 0.574| 0.963| 0.638| 23554| 39893| 0.590 |

### Confusion matrix

|refusal|  ∅| ␣| '| ⏎| ⏎␣⁺␣⁺| ⏎␣⁻␣⁻| ⏎⏎| "| ⏎␣⁻␣⁻␣⁻␣⁻| 
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
|0 |0 |0 |0 |0 |0 |0 |0 |0 |0 |
|2061 |17214 |3 |0 |0 |0 |0 |0 |0 |0 |
|8512 |162 |2545 |0 |0 |0 |0 |0 |0 |0 |
|98 |0 |0 |2853 |0 |0 |0 |0 |0 |0 |
|1833 |15 |0 |0 |294 |0 |0 |0 |0 |0 |
|1454 |179 |2 |0 |0 |0 |0 |0 |0 |0 |
|1233 |227 |16 |0 |0 |0 |0 |0 |0 |0 |
|1037 |0 |0 |0 |28 |0 |0 |0 |0 |0 |
|50 |0 |0 |0 |0 |0 |0 |0 |0 |0 |
|61 |16 |0 |0 |0 |0 |0 |0 |0 |0 |

### Files with most errors

| filename | number of errors|
|:----:|:-----|
| test/createStore.spec.js | 65 |
| examples/todomvc/src/reducers/todos.spec.js | 54 |
| test/combineReducers.spec.js | 33 |
| src/combineReducers.js | 22 |
| examples/todos/src/reducers/todos.spec.js | 21 |
| examples/todos-flow/src/__tests__/reducers/todos.test.js | 20 |
| test/bindActionCreators.spec.js | 14 |
| examples/shopping-cart/src/reducers/cart.spec.js | 13 |
| examples/real-world/src/actions/index.js | 12 |
| examples/todomvc/src/components/MainSection.spec.js | 11 |

<details>
    <summary>Machine-readable report</summary>
```json
{
  "cl_report": {"\"": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 0}, "\u0027": {"f1-score": 1.0, "precision": 1.0, "recall": 1.0, "support": 2853}, "macro avg": {"f1-score": 0.4311070920504449, "precision": 0.4301369342359152, "recall": 0.4323819130668297, "support": 23554}, "micro avg": {"f1-score": 0.9724887492570264, "precision": 0.9724887492570264, "recall": 0.9724887492570264, "support": 23554}, "weighted avg": {"f1-score": 0.9626864841827377, "precision": 0.9534694007788042, "recall": 0.9724887492570264, "support": 23554}, "\u2205": {"f1-score": 0.9828147302312303, "precision": 0.9663728737438949, "recall": 0.9998257536156124, "support": 17217}, "\u23ce": {"f1-score": 0.9318541996830427, "precision": 0.9130434782608695, "recall": 0.9514563106796117, "support": 309}, "\u23ce\u23ce": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 28}, "\u23ce\u2423\u207a\u2423\u207a": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 181}, "\u23ce\u2423\u207b\u2423\u207b": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 243}, "\u23ce\u2423\u207b\u2423\u207b\u2423\u207b\u2423\u207b": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 16}, "\u2423": {"f1-score": 0.9652948985397307, "precision": 0.9918160561184723, "recall": 0.9401551533062431, "support": 2707}},
  "cl_report_full": {"\"": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 50}, "\u0027": {"f1-score": 0.9831150930392832, "precision": 1.0, "recall": 0.9667909183327685, "support": 2951}, "macro avg": {"f1-score": 0.2799107770726007, "precision": 0.4301369342359152, "recall": 0.24709200940519327, "support": 39893}, "micro avg": {"f1-score": 0.722051476035116, "precision": 0.9724887492570264, "recall": 0.5741859474093199, "support": 39893}, "weighted avg": {"f1-score": 0.6379256030232772, "precision": 0.8689158429815439, "recall": 0.5741859474093199, "support": 39893}, "\u2205": {"f1-score": 0.9282036073440996, "precision": 0.9663728737438949, "recall": 0.8929349517584811, "support": 19278}, "\u23ce": {"f1-score": 0.23863636363636362, "precision": 0.9130434782608695, "recall": 0.13725490196078433, "support": 2142}, "\u23ce\u23ce": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 1065}, "\u23ce\u2423\u207a\u2423\u207a": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 1635}, "\u23ce\u2423\u207b\u2423\u207b": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 1476}, "\u23ce\u2423\u207b\u2423\u207b\u2423\u207b\u2423\u207b": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 77}, "\u2423": {"f1-score": 0.36924192963365976, "precision": 0.9918160561184723, "recall": 0.22684731259470542, "support": 11219}},
  "ppcr": 0.5904293986413657
}
```
</details>
