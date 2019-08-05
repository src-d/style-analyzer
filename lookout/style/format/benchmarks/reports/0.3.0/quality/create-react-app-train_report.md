# Train report for javascript / file:///tmp/top-repos-quality-repos-rzj_lvb6/create-react-app HEAD 32106d216e4c31fda30ec475f9f03186d116c893

### Classification report

PPCR: 0.485

| Class | Precision | Recall | Full Recall | F1-score | Full F1-score | Support | Full Support | PPCR |
|------:|:----------|:-------|:------------|:---------|:---------|:--------|:-------------|:-----|
| `∅` | 0.976| 0.978| 0.380| 0.977| 0.547| 4250| 10939| 0.389 |
| `'` | 1.000| 1.000| 0.970| 1.000| 0.985| 2882| 2972| 0.970 |
| `␣` | 0.959| 0.974| 0.508| 0.966| 0.664| 2548| 4888| 0.521 |
| `⏎` | 0.952| 1.000| 0.427| 0.975| 0.590| 893| 2091| 0.427 |
| `⏎␣⁻␣⁻` | 1.000| 0.908| 0.634| 0.952| 0.776| 619| 886| 0.699 |
| `⏎␣⁺␣⁺` | 0.000| 0.000| 0.000| 0.000| 0.000| 19| 905| 0.021 |
| `⏎⏎` | 0.000| 0.000| 0.000| 0.000| 0.000| 19| 459| 0.041 |
| `micro avg` | 0.977| 0.977| 0.474| 0.977| 0.639| 11230| 23140| 0.485 |
| `weighted avg` | 0.974| 0.977| 0.474| 0.976| 0.608| 11230| 23140| 0.485 |
| `macro avg` | 0.698| 0.694| 0.417| 0.696| 0.509| 11230| 23140| 0.485 |

### Confusion matrix

|refusal|  ∅| ␣| '| ⏎| ⏎␣⁺␣⁺| ⏎␣⁻␣⁻| ⏎⏎| 
|:---|:---|:---|:---|:---|:---|:---|:---|
|0 |0 |0 |0 |0 |0 |0 |0 |
|6689 |4158 |92 |0 |0 |0 |0 |0 |
|2340 |53 |2481 |0 |14 |0 |0 |0 |
|90 |0 |0 |2882 |0 |0 |0 |0 |
|1198 |0 |0 |0 |893 |0 |0 |0 |
|886 |8 |11 |0 |0 |0 |0 |0 |
|267 |42 |2 |0 |13 |0 |562 |0 |
|440 |1 |0 |0 |18 |0 |0 |0 |

### Files with most errors

| filename | number of errors|
|:----:|:-----|
| docusaurus/website/pages/en/index.js | 49 |
| docusaurus/website/core/Footer.js | 32 |
| packages/create-react-app/createReactApp.js | 23 |
| fixtures/browser/graphql-with-mjs/src/App.js | 17 |
| packages/eslint-config-react-app/index.js | 14 |
| fixtures/utils.js | 13 |
| packages/react-scripts/config/webpack.config.prod.js | 12 |
| packages/react-scripts/fixtures/kitchensink/integration/syntax.test.js | 11 |
| packages/react-scripts/config/webpack.config.dev.js | 8 |
| fixtures/smoke/relative-paths/index.test.js | 6 |

<details>
    <summary>Machine-readable report</summary>
```json
{
  "cl_report": {"\u0027": {"f1-score": 1.0, "precision": 1.0, "recall": 1.0, "support": 2882}, "macro avg": {"f1-score": 0.6958043749586666, "precision": 0.6981458069637656, "recall": 0.6942819716109206, "support": 11230}, "micro avg": {"f1-score": 0.9773820124666073, "precision": 0.9773820124666073, "recall": 0.9773820124666073, "support": 11230}, "weighted avg": {"f1-score": 0.9756858544885814, "precision": 0.9743539262974282, "recall": 0.9773820124666073, "support": 11230}, "\u2205": {"f1-score": 0.9769736842105263, "precision": 0.975598310652276, "recall": 0.9783529411764705, "support": 4250}, "\u23ce": {"f1-score": 0.9754232659748772, "precision": 0.9520255863539445, "recall": 1.0, "support": 893}, "\u23ce\u23ce": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 19}, "\u23ce\u2423\u207a\u2423\u207a": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 19}, "\u23ce\u2423\u207b\u2423\u207b": {"f1-score": 0.9517358171041491, "precision": 1.0, "recall": 0.9079159935379645, "support": 619}, "\u2423": {"f1-score": 0.966497857421114, "precision": 0.9593967517401392, "recall": 0.9737048665620094, "support": 2548}},
  "cl_report_full": {"\u0027": {"f1-score": 0.9846258968226853, "precision": 1.0, "recall": 0.9697173620457604, "support": 2972}, "macro avg": {"f1-score": 0.5087819043163828, "precision": 0.6981458069637656, "recall": 0.4169678131162979, "support": 23140}, "micro avg": {"f1-score": 0.6386965376782078, "precision": 0.9773820124666073, "recall": 0.47433016421780466, "support": 23140}, "weighted avg": {"f1-score": 0.6083199958146557, "precision": 0.9166070330076554, "recall": 0.47433016421780466, "support": 23140}, "\u2205": {"f1-score": 0.547069271758437, "precision": 0.975598310652276, "recall": 0.3801078709205595, "support": 10939}, "\u23ce": {"f1-score": 0.589633542423242, "precision": 0.9520255863539445, "recall": 0.42706838833094213, "support": 2091}, "\u23ce\u23ce": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 459}, "\u23ce\u2423\u207a\u2423\u207a": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 905}, "\u23ce\u2423\u207b\u2423\u207b": {"f1-score": 0.7762430939226519, "precision": 1.0, "recall": 0.6343115124153499, "support": 886}, "\u2423": {"f1-score": 0.6639015252876639, "precision": 0.9593967517401392, "recall": 0.507569558101473, "support": 4888}},
  "ppcr": 0.4853068280034572
}
```
</details>
