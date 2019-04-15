# Train report for javascript / file:///tmp/top-repos-quality-repos-384uwhei/create-react-app HEAD 32106d216e4c31fda30ec475f9f03186d116c893

### Classification report

PPCR: 0.487

| Class | Precision | Recall | Full Recall | F1-score | Full F1-score | Support | Full Support | PPCR |
|------:|:----------|:-------|:------------|:---------|:---------|:--------|:-------------|:-----|
| `∅` | 0.976| 0.978| 0.380| 0.977| 0.547| 4250| 10939| 0.389 |
| `'` | 1.000| 1.000| 0.970| 1.000| 0.985| 2882| 2972| 0.970 |
| `␣` | 0.959| 0.970| 0.508| 0.965| 0.664| 2557| 4888| 0.523 |
| `⏎` | 0.944| 1.000| 0.436| 0.971| 0.597| 912| 2091| 0.436 |
| `⏎␣⁻␣⁻` | 1.000| 0.908| 0.634| 0.952| 0.776| 619| 886| 0.699 |
| `⏎␣⁺␣⁺` | 0.000| 0.000| 0.000| 0.000| 0.000| 19| 905| 0.021 |
| `⏎⏎` | 0.000| 0.000| 0.000| 0.000| 0.000| 19| 459| 0.041 |
| `micro avg` | 0.977| 0.977| 0.475| 0.977| 0.639| 11258| 23140| 0.487 |
| `macro avg` | 0.697| 0.694| 0.418| 0.695| 0.510| 11258| 23140| 0.487 |
| `weighted avg` | 0.974| 0.977| 0.475| 0.975| 0.609| 11258| 23140| 0.487 |

### Confusion matrix

|refusal|  ∅| ␣| '| ⏎| ⏎␣⁺␣⁺| ⏎␣⁻␣⁻| ⏎⏎| 
|:---|:---|:---|:---|:---|:---|:---|:---|
|0 |0 |0 |0 |0 |0 |0 |0 |
|6689 |4158 |92 |0 |0 |0 |0 |0 |
|2331 |53 |2481 |0 |23 |0 |0 |0 |
|90 |0 |0 |2882 |0 |0 |0 |0 |
|1179 |0 |0 |0 |912 |0 |0 |0 |
|886 |8 |11 |0 |0 |0 |0 |0 |
|267 |42 |2 |0 |13 |0 |562 |0 |
|440 |1 |0 |0 |18 |0 |0 |0 |

### Files with most errors

| filename | number of errors|
|:----:|:-----|
| docusaurus/website/pages/en/index.js | 49 |
| docusaurus/website/core/Footer.js | 32 |
| packages/create-react-app/createReactApp.js | 24 |
| fixtures/browser/graphql-with-mjs/src/App.js | 19 |
| fixtures/utils.js | 18 |
| packages/eslint-config-react-app/index.js | 14 |
| packages/react-scripts/config/webpack.config.prod.js | 12 |
| packages/react-scripts/fixtures/kitchensink/integration/syntax.test.js | 11 |
| packages/react-scripts/config/webpack.config.dev.js | 8 |
| fixtures/smoke/relative-paths/index.test.js | 6 |

<details>
    <summary>Machine-readable report</summary>
```json
{
  "cl_report": {"\u0027": {"f1-score": 1.0, "precision": 1.0, "recall": 1.0, "support": 2882}, "macro avg": {"f1-score": 0.6949660058366155, "precision": 0.6970134916106289, "recall": 0.6937923719797089, "support": 11258}, "micro avg": {"f1-score": 0.976638834606502, "precision": 0.976638834606502, "recall": 0.976638834606502, "support": 11258}, "weighted avg": {"f1-score": 0.9749555242297901, "precision": 0.9736621911540311, "recall": 0.976638834606502, "support": 11258}, "\u2205": {"f1-score": 0.9769736842105263, "precision": 0.975598310652276, "recall": 0.9783529411764705, "support": 4250}, "\u23ce": {"f1-score": 0.9712460063897763, "precision": 0.9440993788819876, "recall": 1.0, "support": 912}, "\u23ce\u23ce": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 19}, "\u23ce\u2423\u207a\u2423\u207a": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 19}, "\u23ce\u2423\u207b\u2423\u207b": {"f1-score": 0.9517358171041491, "precision": 1.0, "recall": 0.9079159935379645, "support": 619}, "\u2423": {"f1-score": 0.9648065331518568, "precision": 0.9593967517401392, "recall": 0.9702776691435275, "support": 2557}},
  "cl_report_full": {"\u0027": {"f1-score": 0.9846258968226853, "precision": 1.0, "recall": 0.9697173620457604, "support": 2972}, "macro avg": {"f1-score": 0.5097861690396012, "precision": 0.6970134916106289, "recall": 0.41826589332399067, "support": 23140}, "micro avg": {"f1-score": 0.6392813535670678, "precision": 0.976638834606502, "recall": 0.4751512532411409, "support": 23140}, "weighted avg": {"f1-score": 0.6089552344816282, "precision": 0.9158907970602111, "recall": 0.4751512532411409, "support": 23140}, "\u2205": {"f1-score": 0.547069271758437, "precision": 0.975598310652276, "recall": 0.3801078709205595, "support": 10939}, "\u23ce": {"f1-score": 0.5966633954857704, "precision": 0.9440993788819876, "recall": 0.43615494978479197, "support": 2091}, "\u23ce\u23ce": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 459}, "\u23ce\u2423\u207a\u2423\u207a": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 905}, "\u23ce\u2423\u207b\u2423\u207b": {"f1-score": 0.7762430939226519, "precision": 1.0, "recall": 0.6343115124153499, "support": 886}, "\u2423": {"f1-score": 0.6639015252876639, "precision": 0.9593967517401392, "recall": 0.507569558101473, "support": 4888}},
  "ppcr": 0.48651685393258426
}
```
</details>
