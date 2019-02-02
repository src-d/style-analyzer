# Train report for javascript / file:///tmp/top-repos-quality-repos-rvax3_kv/create-react-app HEAD 32106d216e4c31fda30ec475f9f03186d116c893

### Classification report

PPCR: 0.949

| Class | Precision | Recall | Full Recall | F1-score | Full F1-score | Support | Full Support | PPCR |
|------:|:----------|:-------|:------------|:---------|:---------|:--------|:-------------|:-----|
| `∅` | 0.947| 0.998| 0.982| 0.972| 0.964| 8725| 8869| 0.984 |
| `␣` | 0.981| 0.925| 0.925| 0.952| 0.952| 2279| 2279| 1.000 |
| `'` | 0.999| 1.000| 0.974| 0.999| 0.986| 2203| 2262| 0.974 |
| `⏎` | 0.958| 0.948| 0.948| 0.953| 0.953| 1144| 1144| 1.000 |
| `⏎␣⁻␣⁻` | 1.000| 0.888| 0.888| 0.941| 0.941| 633| 633| 1.000 |
| `⏎␣⁺␣⁺` | 0.000| 0.000| 0.000| 0.000| 0.000| 221| 221| 1.000 |
| `⏎⏎` | 0.000| 0.000| 0.000| 0.000| 0.000| 32| 65| 0.492 |
| `␣'` | 0.000| 0.000| 0.000| 0.000| 0.000| 3| 585| 0.005 |
| `macro avg` | 0.611| 0.595| 0.590| 0.602| 0.600| 15240| 16058| 0.949 |
| `weighted avg` | 0.947| 0.962| 0.913| 0.954| 0.912| 15240| 16058| 0.949 |
| `micro avg` | 0.962| 0.962| 0.913| 0.962| 0.937| 15240| 16058| 0.949 |

### Confusion matrix

|refusal|  ∅| ␣| ⏎| '| ⏎␣⁺␣⁺| ⏎␣⁻␣⁻| ⏎⏎| ␣'| 
|:---|:---|:---|:---|:---|:---|:---|:---|:---|
|0 |0 |0 |0 |0 |0 |0 |0 |0 |
|144 |8706 |19 |0 |0 |0 |0 |0 |0 |
|0 |156 |2109 |14 |0 |0 |0 |0 |0 |
|0 |60 |0 |1084 |0 |0 |0 |0 |0 |
|59 |0 |0 |0 |2203 |0 |0 |0 |0 |
|0 |195 |22 |4 |0 |0 |0 |0 |0 |
|0 |58 |0 |13 |0 |0 |562 |0 |0 |
|33 |16 |0 |16 |0 |0 |0 |0 |0 |
|582 |0 |0 |0 |3 |0 |0 |0 |0 |

### Files with most errors

| filename | number of errors|
|:----:|:-----|
| packages/create-react-app/createReactApp.js | 144 |
| packages/react-scripts/fixtures/kitchensink/integration/syntax.test.js | 40 |
| packages/react-scripts/config/webpack.config.prod.js | 36 |
| fixtures/output/webpack-message-formatting/index.test.js | 32 |
| fixtures/utils.js | 28 |
| packages/react-scripts/fixtures/kitchensink/integration/webpack.test.js | 27 |
| packages/react-scripts/config/webpack.config.dev.js | 23 |
| docusaurus/website/core/Footer.js | 21 |
| packages/babel-preset-react-app/create.js | 19 |
| docusaurus/website/pages/en/index.js | 18 |

<details>
    <summary>Machine-readable report</summary>
```json
{
  "cl_report": {"\u0027": {"f1-score": 0.9993195735994557, "precision": 0.9986400725294651, "recall": 1.0, "support": 2203}, "macro avg": {"f1-score": 0.6021375688009029, "precision": 0.6106556433647663, "recall": 0.5948270474870072, "support": 15240}, "micro avg": {"f1-score": 0.9622047244094488, "precision": 0.9622047244094488, "recall": 0.9622047244094488, "support": 15240}, "weighted avg": {"f1-score": 0.9538758447425645, "precision": 0.9468237670624848, "recall": 0.9622047244094488, "support": 15240}, "\u2205": {"f1-score": 0.9718687206965839, "precision": 0.9472309868349472, "recall": 0.9978223495702006, "support": 8725}, "\u23ce": {"f1-score": 0.9529670329670331, "precision": 0.9584438549955792, "recall": 0.9475524475524476, "support": 1144}, "\u23ce\u23ce": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 32}, "\u23ce\u2423\u207a\u2423\u207a": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 221}, "\u23ce\u2423\u207b\u2423\u207b": {"f1-score": 0.9405857740585774, "precision": 1.0, "recall": 0.8878357030015798, "support": 633}, "\u2423": {"f1-score": 0.9523594490855722, "precision": 0.9809302325581395, "recall": 0.9254058797718298, "support": 2279}, "\u2423\u0027": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 3}},
  "cl_report_full": {"\u0027": {"f1-score": 0.9861235452103849, "precision": 0.9986400725294651, "recall": 0.9739168877099912, "support": 2262}, "macro avg": {"f1-score": 0.599519425331309, "precision": 0.6106556433647663, "recall": 0.5895415369835937, "support": 16058}, "micro avg": {"f1-score": 0.9370566809380791, "precision": 0.9622047244094488, "recall": 0.9131896873832358, "support": 16058}, "weighted avg": {"f1-score": 0.9115332213607749, "precision": 0.9107557128170222, "recall": 0.9131896873832358, "support": 16058}, "\u2205": {"f1-score": 0.9641196013289037, "precision": 0.9472309868349472, "recall": 0.9816213778329012, "support": 8869}, "\u23ce": {"f1-score": 0.9529670329670331, "precision": 0.9584438549955792, "recall": 0.9475524475524476, "support": 1144}, "\u23ce\u23ce": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 65}, "\u23ce\u2423\u207a\u2423\u207a": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 221}, "\u23ce\u2423\u207b\u2423\u207b": {"f1-score": 0.9405857740585774, "precision": 1.0, "recall": 0.8878357030015798, "support": 633}, "\u2423": {"f1-score": 0.9523594490855722, "precision": 0.9809302325581395, "recall": 0.9254058797718298, "support": 2279}, "\u2423\u0027": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 585}},
  "ppcr": 0.9490596587370781
}
```
</details>
