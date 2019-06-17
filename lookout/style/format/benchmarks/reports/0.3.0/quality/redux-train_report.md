# Train report for javascript / file:///tmp/top-repos-quality-repos-_9qbbjnr/redux HEAD 902484ed735d38aec06683c847810a7218d8dba2

### Classification report

PPCR: 0.588

| Class | Precision | Recall | Full Recall | F1-score | Full F1-score | Support | Full Support | PPCR |
|------:|:----------|:-------|:------------|:---------|:---------|:--------|:-------------|:-----|
| `∅` | 0.966| 1.000| 0.873| 0.982| 0.917| 16829| 19278| 0.873 |
| `'` | 1.000| 1.000| 0.967| 1.000| 0.983| 2853| 2951| 0.967 |
| `␣` | 0.992| 0.940| 0.227| 0.965| 0.369| 2707| 11219| 0.241 |
| `⏎` | 0.913| 0.907| 0.137| 0.910| 0.239| 324| 2142| 0.151 |
| `⏎⏎` | 0.940| 0.905| 0.251| 0.922| 0.396| 295| 1065| 0.277 |
| `⏎␣⁻␣⁻` | 0.000| 0.000| 0.000| 0.000| 0.000| 245| 1476| 0.166 |
| `⏎␣⁺␣⁺` | 0.000| 0.000| 0.000| 0.000| 0.000| 181| 1635| 0.111 |
| `⏎␣⁻␣⁻␣⁻␣⁻` | 0.000| 0.000| 0.000| 0.000| 0.000| 16| 77| 0.208 |
| `"` | 0.000| 0.000| 0.000| 0.000| 0.000| 0| 50| 0.000 |
| `micro avg` | 0.972| 0.972| 0.571| 0.972| 0.719| 23450| 39893| 0.588 |
| `weighted avg` | 0.954| 0.972| 0.571| 0.962| 0.643| 23450| 39893| 0.588 |
| `macro avg` | 0.535| 0.528| 0.273| 0.531| 0.323| 23450| 39893| 0.588 |

### Confusion matrix

|refusal|  ∅| ␣| '| ⏎| ⏎␣⁺␣⁺| ⏎␣⁻␣⁻| ⏎⏎| "| ⏎␣⁻␣⁻␣⁻␣⁻| 
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
|0 |0 |0 |0 |0 |0 |0 |0 |0 |0 |
|2449 |16826 |3 |0 |0 |0 |0 |0 |0 |0 |
|8512 |162 |2545 |0 |0 |0 |0 |0 |0 |0 |
|98 |0 |0 |2853 |0 |0 |0 |0 |0 |0 |
|1818 |15 |0 |0 |294 |0 |0 |15 |0 |0 |
|1454 |179 |2 |0 |0 |0 |0 |0 |0 |0 |
|1231 |227 |16 |0 |0 |0 |0 |2 |0 |0 |
|770 |0 |0 |0 |28 |0 |0 |267 |0 |0 |
|50 |0 |0 |0 |0 |0 |0 |0 |0 |0 |
|61 |16 |0 |0 |0 |0 |0 |0 |0 |0 |

### Files with most errors

| filename | number of errors|
|:----:|:-----|
| test/createStore.spec.js | 72 |
| examples/todomvc/src/reducers/todos.spec.js | 54 |
| test/combineReducers.spec.js | 36 |
| src/combineReducers.js | 23 |
| examples/todos/src/reducers/todos.spec.js | 21 |
| examples/todos-flow/src/__tests__/reducers/todos.test.js | 20 |
| test/bindActionCreators.spec.js | 16 |
| examples/shopping-cart/src/reducers/cart.spec.js | 13 |
| examples/real-world/src/actions/index.js | 12 |
| examples/todomvc/src/components/MainSection.spec.js | 11 |

<details>
    <summary>Machine-readable report</summary>
```json
{
  "cl_report": {"\"": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 0}, "\u0027": {"f1-score": 1.0, "precision": 1.0, "recall": 1.0, "support": 2853}, "macro avg": {"f1-score": 0.5311352021914637, "precision": 0.5345138314166245, "recall": 0.528052115862768, "support": 23450}, "micro avg": {"f1-score": 0.9716417910447761, "precision": 0.9716417910447761, "recall": 0.9716417910447761, "support": 23450}, "weighted avg": {"f1-score": 0.9623144253174603, "precision": 0.9535821635223151, "recall": 0.9716417910447761, "support": 23450}, "\u2205": {"f1-score": 0.9824254101710749, "precision": 0.9656241032998565, "recall": 0.9998217362885495, "support": 16829}, "\u23ce": {"f1-score": 0.9102167182662538, "precision": 0.9130434782608695, "recall": 0.9074074074074074, "support": 324}, "\u23ce\u23ce": {"f1-score": 0.9222797927461139, "precision": 0.9401408450704225, "recall": 0.9050847457627119, "support": 295}, "\u23ce\u2423\u207a\u2423\u207a": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 181}, "\u23ce\u2423\u207b\u2423\u207b": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 245}, "\u23ce\u2423\u207b\u2423\u207b\u2423\u207b\u2423\u207b": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 16}, "\u2423": {"f1-score": 0.9652948985397307, "precision": 0.9918160561184723, "recall": 0.9401551533062431, "support": 2707}},
  "cl_report_full": {"\"": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 50}, "\u0027": {"f1-score": 0.9831150930392832, "precision": 1.0, "recall": 0.9667909183327685, "support": 2951}, "macro avg": {"f1-score": 0.3226350492549339, "precision": 0.5345138314166245, "recall": 0.2727117489836306, "support": 39893}, "micro avg": {"f1-score": 0.7194165101116146, "precision": 0.9716417910447761, "recall": 0.5711528338304965, "support": 39893}, "weighted avg": {"f1-score": 0.6430180477883708, "precision": 0.8936523933382436, "recall": 0.5711528338304965, "support": 39893}, "\u2205": {"f1-score": 0.9168732801133421, "precision": 0.9656241032998565, "recall": 0.8728083826123042, "support": 19278}, "\u23ce": {"f1-score": 0.23863636363636362, "precision": 0.9130434782608695, "recall": 0.13725490196078433, "support": 2142}, "\u23ce\u23ce": {"f1-score": 0.3958487768717568, "precision": 0.9401408450704225, "recall": 0.2507042253521127, "support": 1065}, "\u23ce\u2423\u207a\u2423\u207a": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 1635}, "\u23ce\u2423\u207b\u2423\u207b": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 1476}, "\u23ce\u2423\u207b\u2423\u207b\u2423\u207b\u2423\u207b": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 77}, "\u2423": {"f1-score": 0.36924192963365976, "precision": 0.9918160561184723, "recall": 0.22684731259470542, "support": 11219}},
  "ppcr": 0.5878224249868398
}
```
</details>
