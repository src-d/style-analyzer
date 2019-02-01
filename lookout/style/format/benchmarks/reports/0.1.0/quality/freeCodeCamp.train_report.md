# Train report for javascript / file:///tmp/top-repos-quality-repos-hda3ai6m/freeCodeCamp HEAD cf65516cce60645a417e44c4fcea7418ca920572

### Classification report

PPCR: 0.760

| Class | Precision | Recall | Full Recall | F1-score | Full F1-score | Support | Full Support | PPCR |
|------:|:----------|:-------|:------------|:---------|:---------|:--------|:-------------|:-----|
| `∅` | 0.975| 0.997| 0.887| 0.986| 0.929| 52031| 58481| 0.890 |
| `␣` | 0.978| 0.978| 0.722| 0.978| 0.831| 19428| 26336| 0.738 |
| `⏎` | 0.944| 0.893| 0.430| 0.918| 0.591| 4327| 8987| 0.481 |
| `'` | 0.960| 0.998| 0.998| 0.979| 0.979| 3937| 3937| 1.000 |
| `␣'` | 0.998| 1.000| 1.000| 0.999| 0.999| 2415| 2415| 1.000 |
| `⏎⏎` | 0.943| 0.918| 0.362| 0.930| 0.523| 876| 2220| 0.395 |
| `⏎␣⁺␣⁺` | 0.000| 0.000| 0.000| 0.000| 0.000| 555| 4435| 0.125 |
| `⏎␣⁻␣⁻` | 0.000| 0.000| 0.000| 0.000| 0.000| 362| 3657| 0.099 |
| `'⏎` | 1.000| 0.976| 0.976| 0.988| 0.988| 337| 337| 1.000 |
| `'⏎␣⁻␣⁻` | 0.000| 0.000| 0.000| 0.000| 0.000| 101| 101| 1.000 |
| `'␣` | 0.000| 0.000| 0.000| 0.000| 0.000| 55| 55| 1.000 |
| `⏎␣⁻␣⁻␣⁻␣⁻` | 0.000| 0.000| 0.000| 0.000| 0.000| 33| 190| 0.174 |
| `"` | 0.000| 0.000| 0.000| 0.000| 0.000| 0| 0| 0.000 |
| `macro avg` | 0.523| 0.520| 0.414| 0.521| 0.449| 84457| 111151| 0.760 |
| `weighted avg` | 0.961| 0.974| 0.740| 0.967| 0.803| 84457| 111151| 0.760 |
| `micro avg` | 0.974| 0.974| 0.740| 0.974| 0.841| 84457| 111151| 0.760 |

### Confusion matrix

|refusal|  ∅| ␣| ⏎| '| ⏎␣⁺␣⁺| ⏎␣⁻␣⁻| ␣'| ⏎⏎| '␣| '⏎| '⏎␣⁻␣⁻| "| ⏎␣⁻␣⁻␣⁻␣⁻| 
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
|0 |0 |0 |0 |0 |0 |0 |0 |0 |0 |0 |0 |0 |
|6450 |51877 |154 |0 |0 |0 |0 |0 |0 |0 |0 |0 |0 |
|6908 |267 |19010 |149 |0 |0 |0 |0 |2 |0 |0 |0 |0 |
|4660 |369 |47 |3864 |0 |0 |0 |0 |47 |0 |0 |0 |0 |
|0 |0 |0 |0 |3931 |0 |0 |6 |0 |0 |0 |0 |0 |
|3880 |330 |223 |2 |0 |0 |0 |0 |0 |0 |0 |0 |0 |
|3295 |344 |9 |9 |0 |0 |0 |0 |0 |0 |0 |0 |0 |
|0 |0 |0 |0 |0 |0 |0 |2415 |0 |0 |0 |0 |0 |
|1344 |2 |0 |70 |0 |0 |0 |0 |804 |0 |0 |0 |0 |
|0 |0 |0 |0 |55 |0 |0 |0 |0 |0 |0 |0 |0 |
|0 |0 |0 |0 |8 |0 |0 |0 |0 |0 |329 |0 |0 |
|0 |0 |0 |0 |101 |0 |0 |0 |0 |0 |0 |0 |0 |
|157 |33 |0 |0 |0 |0 |0 |0 |0 |0 |0 |0 |0 |

### Files with most errors

| filename | number of errors|
|:----:|:-----|
| api-server/common/models/user.js | 139 |
| api-server/public/js/calculator.js | 109 |
| api-server/server/utils/map.js | 51 |
| api-server/server/boot/challenge.js | 46 |
| client/src/components/settings/Portfolio.js | 43 |
| client/src/templates/Challenges/rechallenge/throwers.js | 34 |
| curriculum/test/test-challenges.js | 33 |
| api-server/server/utils/user-stats.test.js | 32 |
| api-server/server/boot/certificate.js | 31 |
| curriculum/unpackedChallenge.js | 30 |

<details>
    <summary>Machine-readable report</summary>
```json
{
  "cl_report": {"\"": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 0}, "\u0027": {"f1-score": 0.9788346613545815, "precision": 0.95995115995116, "recall": 0.9984759969519938, "support": 3937}, "\u0027\u23ce": {"f1-score": 0.9879879879879879, "precision": 1.0, "recall": 0.9762611275964391, "support": 337}, "\u0027\u23ce\u2423\u207b\u2423\u207b": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 101}, "\u0027\u2423": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 55}, "macro avg": {"f1-score": 0.5213207876195198, "precision": 0.5227928481296696, "recall": 0.5200821299140123, "support": 84457}, "micro avg": {"f1-score": 0.9736315521507987, "precision": 0.9736315521507987, "recall": 0.9736315521507987, "support": 84457}, "weighted avg": {"f1-score": 0.9670824311384523, "precision": 0.9608009270364342, "recall": 0.9736315521507987, "support": 84457}, "\u2205": {"f1-score": 0.9857581256591261, "precision": 0.9747284957348465, "recall": 0.997040226019104, "support": 52031}, "\u23ce": {"f1-score": 0.917705735660848, "precision": 0.9438202247191011, "recall": 0.892997457822972, "support": 4327}, "\u23ce\u23ce": {"f1-score": 0.9300173510699827, "precision": 0.9425556858147714, "recall": 0.9178082191780822, "support": 876}, "\u23ce\u2423\u207a\u2423\u207a": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 555}, "\u23ce\u2423\u207b\u2423\u207b": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 362}, "\u23ce\u2423\u207b\u2423\u207b\u2423\u207b\u2423\u207b": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 33}, "\u2423": {"f1-score": 0.9781070721103138, "precision": 0.9777297742117986, "recall": 0.978484661313568, "support": 19428}, "\u2423\u0027": {"f1-score": 0.9987593052109182, "precision": 0.9975216852540273, "recall": 1.0, "support": 2415}},
  "cl_report_full": {"\"": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 0}, "\u0027": {"f1-score": 0.9788346613545815, "precision": 0.95995115995116, "recall": 0.9984759969519938, "support": 3937}, "\u0027\u23ce": {"f1-score": 0.9879879879879879, "precision": 1.0, "recall": 0.9762611275964391, "support": 337}, "\u0027\u23ce\u2423\u207b\u2423\u207b": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 101}, "\u0027\u2423": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 55}, "macro avg": {"f1-score": 0.44915227137156805, "precision": 0.5227928481296696, "recall": 0.41351951827685896, "support": 111151}, "micro avg": {"f1-score": 0.8407631589710032, "precision": 0.9736315521507987, "recall": 0.7398044102167322, "support": 111151}, "weighted avg": {"f1-score": 0.8030639765820904, "precision": 0.8983501458501232, "recall": 0.7398044102167322, "support": 111151}, "\u2205": {"f1-score": 0.9288380795502359, "precision": 0.9747284957348465, "recall": 0.8870744344316958, "support": 58481}, "\u23ce": {"f1-score": 0.590780521366868, "precision": 0.9438202247191011, "recall": 0.4299543785467898, "support": 8987}, "\u23ce\u23ce": {"f1-score": 0.5232671656361861, "precision": 0.9425556858147714, "recall": 0.3621621621621622, "support": 2220}, "\u23ce\u2423\u207a\u2423\u207a": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 4435}, "\u23ce\u2423\u207b\u2423\u207b": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 3657}, "\u23ce\u2423\u207b\u2423\u207b\u2423\u207b\u2423\u207b": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 190}, "\u2423": {"f1-score": 0.8305118067236069, "precision": 0.9777297742117986, "recall": 0.721825637910085, "support": 26336}, "\u2423\u0027": {"f1-score": 0.9987593052109182, "precision": 0.9975216852540273, "recall": 1.0, "support": 2415}},
  "ppcr": 0.759840217361967
}
```
</details>
