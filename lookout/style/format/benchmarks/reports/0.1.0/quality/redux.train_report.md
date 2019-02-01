# Train report for javascript / file:///tmp/top-repos-quality-repos-7y4u1kvh/redux HEAD 902484ed735d38aec06683c847810a7218d8dba2

### Classification report

PPCR: 0.942

| Class | Precision | Recall | Full Recall | F1-score | Full F1-score | Support | Full Support | PPCR |
|------:|:----------|:-------|:------------|:---------|:---------|:--------|:-------------|:-----|
| `∅` | 0.982| 0.999| 0.987| 0.990| 0.985| 14241| 14409| 0.988 |
| `␣` | 0.962| 0.978| 0.978| 0.970| 0.970| 4037| 4037| 1.000 |
| `'` | 1.000| 1.000| 0.357| 1.000| 0.526| 485| 1359| 0.357 |
| `'⏎` | 0.996| 1.000| 0.980| 0.998| 0.988| 251| 256| 0.980 |
| `⏎␣⁺␣⁺` | 0.000| 0.000| 0.000| 0.000| 0.000| 163| 163| 1.000 |
| `⏎␣⁻␣⁻` | 0.000| 0.000| 0.000| 0.000| 0.000| 97| 97| 1.000 |
| `⏎` | 0.000| 0.000| 0.000| 0.000| 0.000| 59| 65| 0.908 |
| `'⏎⏎` | 0.000| 0.000| 0.000| 0.000| 0.000| 1| 14| 0.071 |
| `␣'` | 0.000| 0.000| 0.000| 0.000| 0.000| 0| 0| 0.000 |
| `⏎⏎` | 0.000| 0.000| 0.000| 0.000| 0.000| 0| 37| 0.000 |
| `'␣` | 0.000| 0.000| 0.000| 0.000| 0.000| 0| 98| 0.000 |
| `macro avg` | 0.358| 0.362| 0.300| 0.360| 0.315| 19334| 20535| 0.942 |
| `weighted avg` | 0.962| 0.978| 0.921| 0.970| 0.929| 19334| 20535| 0.942 |
| `micro avg` | 0.978| 0.978| 0.921| 0.978| 0.949| 19334| 20535| 0.942 |

### Confusion matrix

|refusal|  ∅| ␣| ⏎| ⏎␣⁺␣⁺| ⏎␣⁻␣⁻| '| ␣'| ⏎⏎| '⏎| '⏎⏎| '␣| 
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
|0 |0 |0 |0 |0 |0 |0 |0 |0 |0 |0 |
|168 |14227 |14 |0 |0 |0 |0 |0 |0 |0 |0 |
|0 |90 |3947 |0 |0 |0 |0 |0 |0 |0 |0 |
|6 |18 |41 |0 |0 |0 |0 |0 |0 |0 |0 |
|0 |76 |87 |0 |0 |0 |0 |0 |0 |0 |0 |
|0 |81 |16 |0 |0 |0 |0 |0 |0 |0 |0 |
|874 |0 |0 |0 |0 |0 |485 |0 |0 |0 |0 |
|37 |0 |0 |0 |0 |0 |0 |0 |0 |0 |0 |
|5 |0 |0 |0 |0 |0 |0 |0 |251 |0 |0 |
|13 |0 |0 |0 |0 |0 |0 |0 |1 |0 |0 |
|98 |0 |0 |0 |0 |0 |0 |0 |0 |0 |0 |

### Files with most errors

| filename | number of errors|
|:----:|:-----|
| test/createStore.spec.js | 62 |
| examples/todomvc/src/reducers/todos.spec.js | 54 |
| examples/todos-flow/src/__tests__/reducers/todos.test.js | 22 |
| examples/todos/src/reducers/todos.spec.js | 21 |
| src/combineReducers.js | 13 |
| test/bindActionCreators.spec.js | 11 |
| test/applyMiddleware.spec.js | 11 |
| test/helpers/reducers.js | 8 |
| src/utils/actionTypes.js | 8 |
| examples/todomvc/src/reducers/todos.js | 7 |

<details>
    <summary>Machine-readable report</summary>
```json
{
  "cl_report": {"\u0027": {"f1-score": 1.0, "precision": 1.0, "recall": 1.0, "support": 485}, "\u0027\u23ce": {"f1-score": 0.9980119284294234, "precision": 0.996031746031746, "recall": 1.0, "support": 251}, "\u0027\u23ce\u23ce": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 1}, "\u0027\u2423": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 0}, "macro avg": {"f1-score": 0.35980386294430394, "precision": 0.3581141953081884, "recall": 0.36152028549610243, "support": 19334}, "micro avg": {"f1-score": 0.9780697217337333, "precision": 0.9780697217337333, "recall": 0.9780697217337333, "support": 19334}, "weighted avg": {"f1-score": 0.9699107907337002, "precision": 0.9618915401832373, "recall": 0.9780697217337333, "support": 19334}, "\u2205": {"f1-score": 0.990289910555807, "precision": 0.9817140491305548, "recall": 0.9990169229688927, "support": 14241}, "\u23ce": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 59}, "\u23ce\u23ce": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 0}, "\u23ce\u2423\u207a\u2423\u207a": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 163}, "\u23ce\u2423\u207b\u2423\u207b": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 97}, "\u2423": {"f1-score": 0.9695406534021126, "precision": 0.961510353227771, "recall": 0.9777062174882338, "support": 4037}, "\u2423\u0027": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 0}},
  "cl_report_full": {"\u0027": {"f1-score": 0.5260303687635575, "precision": 1.0, "recall": 0.35688005886681384, "support": 1359}, "\u0027\u23ce": {"f1-score": 0.9881889763779528, "precision": 0.996031746031746, "recall": 0.98046875, "support": 256}, "\u0027\u23ce\u23ce": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 14}, "\u0027\u2423": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 98}, "macro avg": {"f1-score": 0.31529940051746946, "precision": 0.3581141953081884, "recall": 0.30022036653070294, "support": 20535}, "micro avg": {"f1-score": 0.9486066868995963, "precision": 0.9780697217337333, "recall": 0.9208668127587046, "support": 20535}, "weighted avg": {"f1-score": 0.9285624120034135, "precision": 0.9564703753049332, "recall": 0.9208668127587046, "support": 20535}, "\u2205": {"f1-score": 0.9845334071485417, "precision": 0.9817140491305548, "recall": 0.9873690054826845, "support": 14409}, "\u23ce": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 65}, "\u23ce\u23ce": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 37}, "\u23ce\u2423\u207a\u2423\u207a": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 163}, "\u23ce\u2423\u207b\u2423\u207b": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 97}, "\u2423": {"f1-score": 0.9695406534021126, "precision": 0.961510353227771, "recall": 0.9777062174882338, "support": 4037}, "\u2423\u0027": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 0}},
  "ppcr": 0.9415144874604334
}
```
</details>
