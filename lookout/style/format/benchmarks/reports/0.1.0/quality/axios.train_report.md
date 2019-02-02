# Train report for javascript / file:///tmp/top-repos-quality-repos-du7wfo2h/axios HEAD 21ae22dbd3ae3d3a55d9efd4eead3dd7fb6d8e6e

### Classification report

PPCR: 1.000

| Class | Precision | Recall | Full Recall | F1-score | Full F1-score | Support | Full Support | PPCR |
|------:|:----------|:-------|:------------|:---------|:---------|:--------|:-------------|:-----|
| `∅` | 0.976| 0.995| 0.995| 0.985| 0.985| 14284| 14288| 1.000 |
| `␣` | 0.966| 0.923| 0.923| 0.944| 0.944| 3294| 3294| 1.000 |
| `'` | 0.951| 1.000| 1.000| 0.975| 0.975| 2027| 2027| 1.000 |
| `⏎␣⁻␣⁻` | 0.991| 0.969| 0.969| 0.980| 0.980| 796| 796| 1.000 |
| `⏎⏎` | 0.977| 0.850| 0.850| 0.909| 0.909| 247| 247| 1.000 |
| `'␣` | 0.000| 0.000| 0.000| 0.000| 0.000| 104| 104| 1.000 |
| `⏎` | 0.000| 0.000| 0.000| 0.000| 0.000| 61| 62| 0.984 |
| `⏎␣⁺␣⁺` | 0.000| 0.000| 0.000| 0.000| 0.000| 27| 27| 1.000 |
| `␣'` | 0.000| 0.000| 0.000| 0.000| 0.000| 0| 0| 0.000 |
| `macro avg` | 0.540| 0.526| 0.526| 0.533| 0.533| 20840| 20845| 1.000 |
| `weighted avg` | 0.963| 0.972| 0.972| 0.968| 0.968| 20840| 20845| 1.000 |
| `micro avg` | 0.972| 0.972| 0.972| 0.972| 0.972| 20840| 20845| 1.000 |

### Confusion matrix

|refusal|  ∅| ␣| '| ⏎| ⏎␣⁺␣⁺| ⏎␣⁻␣⁻| ⏎⏎| ␣'| '␣| 
|:---|:---|:---|:---|:---|:---|:---|:---|:---|
|0 |0 |0 |0 |0 |0 |0 |0 |0 |
|4 |14215 |69 |0 |0 |0 |0 |0 |0 |
|0 |246 |3042 |0 |0 |0 |6 |0 |0 |
|0 |0 |0 |2027 |0 |0 |0 |0 |0 |
|1 |34 |21 |0 |0 |0 |1 |5 |0 |
|0 |19 |8 |0 |0 |0 |0 |0 |0 |
|0 |19 |6 |0 |0 |0 |771 |0 |0 |
|0 |34 |3 |0 |0 |0 |0 |210 |0 |
|0 |0 |0 |104 |0 |0 |0 |0 |0 |

### Files with most errors

| filename | number of errors|
|:----:|:-----|
| lib/adapters/http.js | 72 |
| lib/adapters/xhr.js | 63 |
| examples/server.js | 38 |
| karma.conf.js | 30 |
| test/specs/requests.spec.js | 28 |
| lib/helpers/isURLSameOrigin.js | 26 |
| lib/core/dispatchRequest.js | 23 |
| lib/helpers/cookies.js | 21 |
| lib/helpers/buildURL.js | 19 |
| lib/utils.js | 19 |

<details>
    <summary>Machine-readable report</summary>
```json
{
  "cl_report": {"\u0027": {"f1-score": 0.9749879749879751, "precision": 0.9511966213045518, "recall": 1.0, "support": 2027}, "\u0027\u2423": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 104}, "macro avg": {"f1-score": 0.5326041016707729, "precision": 0.5400889033678935, "recall": 0.5263846757848801, "support": 20840}, "micro avg": {"f1-score": 0.9724088291746641, "precision": 0.9724088291746641, "recall": 0.9724088291746641, "support": 20840}, "weighted avg": {"f1-score": 0.9676913747378536, "precision": 0.9634875681950996, "recall": 0.9724088291746641, "support": 20840}, "\u2205": {"f1-score": 0.9854077848254827, "precision": 0.9758357932312762, "recall": 0.9951694203304396, "support": 14284}, "\u23ce": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 61}, "\u23ce\u23ce": {"f1-score": 0.9090909090909091, "precision": 0.9767441860465116, "recall": 0.8502024291497976, "support": 247}, "\u23ce\u2423\u207a\u2423\u207a": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 27}, "\u23ce\u2423\u207b\u2423\u207b": {"f1-score": 0.9796696315120711, "precision": 0.9910025706940874, "recall": 0.9685929648241206, "support": 796}, "\u2423": {"f1-score": 0.9442806146205184, "precision": 0.9660209590346142, "recall": 0.9234972677595629, "support": 3294}, "\u2423\u0027": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 0}},
  "cl_report_full": {"\u0027": {"f1-score": 0.9749879749879751, "precision": 0.9511966213045518, "recall": 1.0, "support": 2027}, "\u0027\u2423": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 104}, "macro avg": {"f1-score": 0.5325889237461333, "precision": 0.5400889033678935, "recall": 0.5263537199114096, "support": 20845}, "micro avg": {"f1-score": 0.9722921914357683, "precision": 0.9724088291746641, "recall": 0.9721755816742624, "support": 20845}, "weighted avg": {"f1-score": 0.967554719164831, "precision": 0.963443716208146, "recall": 0.9721755816742624, "support": 20845}, "\u2205": {"f1-score": 0.9852711835037256, "precision": 0.9758357932312762, "recall": 0.9948908174692049, "support": 14288}, "\u23ce": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 62}, "\u23ce\u23ce": {"f1-score": 0.9090909090909091, "precision": 0.9767441860465116, "recall": 0.8502024291497976, "support": 247}, "\u23ce\u2423\u207a\u2423\u207a": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 27}, "\u23ce\u2423\u207b\u2423\u207b": {"f1-score": 0.9796696315120711, "precision": 0.9910025706940874, "recall": 0.9685929648241206, "support": 796}, "\u2423": {"f1-score": 0.9442806146205184, "precision": 0.9660209590346142, "recall": 0.9234972677595629, "support": 3294}, "\u2423\u0027": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 0}},
  "ppcr": 0.9997601343247782
}
```
</details>
