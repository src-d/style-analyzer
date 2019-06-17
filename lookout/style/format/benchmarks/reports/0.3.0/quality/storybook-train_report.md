# Train report for javascript / file:///tmp/top-repos-quality-repos-f7q5vkiy/storybook HEAD b28217f887af533a17cb1498887d6b4bd41bd643

### Classification report

PPCR: 0.855

| Class | Precision | Recall | Full Recall | F1-score | Full F1-score | Support | Full Support | PPCR |
|------:|:----------|:-------|:------------|:---------|:---------|:--------|:-------------|:-----|
| `∅` | 0.970| 0.997| 0.943| 0.983| 0.957| 88000| 92989| 0.946 |
| `␣` | 0.978| 0.966| 0.792| 0.972| 0.875| 40640| 49578| 0.820 |
| `'` | 1.000| 1.000| 0.999| 1.000| 1.000| 18687| 18696| 1.000 |
| `⏎␣⁻␣⁻` | 0.986| 0.883| 0.786| 0.931| 0.875| 5518| 6197| 0.890 |
| `⏎␣⁺␣⁺` | 0.889| 0.801| 0.663| 0.842| 0.759| 5280| 6377| 0.828 |
| `⏎` | 0.925| 0.801| 0.285| 0.859| 0.436| 4014| 11277| 0.356 |
| `"` | 0.998| 0.993| 0.993| 0.995| 0.995| 882| 882| 1.000 |
| `⏎⏎` | 1.000| 0.637| 0.041| 0.778| 0.078| 317| 4955| 0.064 |
| `⏎␣⁻␣⁻␣⁻␣⁻` | 0.000| 0.000| 0.000| 0.000| 0.000| 112| 131| 0.855 |
| `⏎⏎␣⁻␣⁻` | 0.000| 0.000| 0.000| 0.000| 0.000| 2| 101| 0.020 |
| `micro avg` | 0.973| 0.973| 0.832| 0.973| 0.897| 163452| 191183| 0.855 |
| `weighted avg` | 0.972| 0.973| 0.832| 0.972| 0.876| 163452| 191183| 0.855 |
| `macro avg` | 0.775| 0.708| 0.550| 0.736| 0.598| 163452| 191183| 0.855 |

### Confusion matrix

|refusal|  ∅| ␣| '| ⏎| ⏎␣⁺␣⁺| ⏎␣⁻␣⁻| ⏎⏎| "| ⏎␣⁻␣⁻␣⁻␣⁻| ⏎⏎␣⁻␣⁻| 
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
|0 |0 |0 |0 |0 |0 |0 |0 |0 |0 |0 |
|4989 |87710 |278 |0 |0 |2 |10 |0 |0 |0 |0 |
|8938 |676 |39257 |0 |177 |527 |3 |0 |0 |0 |0 |
|9 |0 |0 |18685 |0 |0 |0 |0 |2 |0 |0 |
|7263 |608 |191 |0 |3215 |0 |0 |0 |0 |0 |0 |
|1097 |770 |282 |0 |0 |4228 |0 |0 |0 |0 |0 |
|679 |536 |96 |0 |15 |0 |4871 |0 |0 |0 |0 |
|4638 |30 |21 |0 |64 |0 |0 |202 |0 |0 |0 |
|0 |0 |0 |6 |0 |0 |0 |0 |876 |0 |0 |
|19 |53 |0 |0 |1 |0 |58 |0 |0 |0 |0 |
|99 |0 |0 |0 |2 |0 |0 |0 |0 |0 |0 |

### Files with most errors

| filename | number of errors|
|:----:|:-----|
| examples/official-storybook/stories/addon-info.stories.js | 107 |
| examples/official-storybook/stories/addon-actions.stories.js | 82 |
| lib/ui/src/modules/ui/components/stories_panel/stories_tree/index.test.js | 74 |
| lib/cli/lib/initiate.js | 63 |
| addons/jest/src/components/Result.js | 56 |
| lib/core/src/server/mergeConfigs.js | 53 |
| lib/components/src/tabs/tabs.stories.js | 52 |
| lib/core/src/client/preview/client_api.test.js | 50 |
| lib/codemod/src/transforms/__testfixtures__/update-addon-info/update-addon-info.input.js | 48 |
| examples/official-storybook/components/FlowTypeButton.js | 44 |

<details>
    <summary>Machine-readable report</summary>
```json
{
  "cl_report": {"\"": {"f1-score": 0.9954545454545456, "precision": 0.9977220956719818, "recall": 0.9931972789115646, "support": 882}, "\u0027": {"f1-score": 0.9997859703568944, "precision": 0.9996789898881815, "recall": 0.9998929737250495, "support": 18687}, "macro avg": {"f1-score": 0.7361726874539857, "precision": 0.7746069518932208, "recall": 0.7077439895634348, "support": 163452}, "micro avg": {"f1-score": 0.9730318380931405, "precision": 0.9730318380931405, "recall": 0.9730318380931405, "support": 163452}, "weighted avg": {"f1-score": 0.9720759373294517, "precision": 0.97204459071545, "recall": 0.9730318380931405, "support": 163452}, "\u2205": {"f1-score": 0.9833896727827204, "precision": 0.9704258544195258, "recall": 0.9967045454545455, "support": 88000}, "\u23ce": {"f1-score": 0.8587072649572649, "precision": 0.9254461715601612, "recall": 0.8009466865969108, "support": 4014}, "\u23ce\u23ce": {"f1-score": 0.7784200385356455, "precision": 1.0, "recall": 0.637223974763407, "support": 317}, "\u23ce\u23ce\u2423\u207b\u2423\u207b": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 2}, "\u23ce\u2423\u207a\u2423\u207a": {"f1-score": 0.8424828135897181, "precision": 0.8887954593231028, "recall": 0.8007575757575758, "support": 5280}, "\u23ce\u2423\u207b\u2423\u207b": {"f1-score": 0.9313575525812621, "precision": 0.9856333468231485, "recall": 0.8827473722363175, "support": 5518}, "\u23ce\u2423\u207b\u2423\u207b\u2423\u207b\u2423\u207b": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 112}, "\u2423": {"f1-score": 0.9721290162818053, "precision": 0.9783676012461059, "recall": 0.9659694881889764, "support": 40640}},
  "cl_report_full": {"\"": {"f1-score": 0.9954545454545456, "precision": 0.9977220956719818, "recall": 0.9931972789115646, "support": 882}, "\u0027": {"f1-score": 0.9995452964934335, "precision": 0.9996789898881815, "recall": 0.9994116388532306, "support": 18696}, "macro avg": {"f1-score": 0.5975203642317748, "precision": 0.7746069518932208, "recall": 0.5502555383597626, "support": 191183}, "micro avg": {"f1-score": 0.8969447460064572, "precision": 0.9730318380931405, "recall": 0.8318940491570903, "support": 191183}, "weighted avg": {"f1-score": 0.8760337306529142, "precision": 0.9701778032857484, "recall": 0.8318940491570903, "support": 191183}, "\u2205": {"f1-score": 0.9566346007023973, "precision": 0.9704258544195258, "recall": 0.9432298443901966, "support": 92989}, "\u23ce": {"f1-score": 0.4359026506677513, "precision": 0.9254461715601612, "recall": 0.2850935532499778, "support": 11277}, "\u23ce\u23ce": {"f1-score": 0.07834012022493697, "precision": 1.0, "recall": 0.04076690211907164, "support": 4955}, "\u23ce\u23ce\u2423\u207b\u2423\u207b": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 101}, "\u23ce\u2423\u207a\u2423\u207a": {"f1-score": 0.7594754805101491, "precision": 0.8887954593231028, "recall": 0.6630076838638859, "support": 6377}, "\u23ce\u2423\u207b\u2423\u207b": {"f1-score": 0.8745847921716491, "precision": 0.9856333468231485, "recall": 0.7860254962078425, "support": 6197}, "\u23ce\u2423\u207b\u2423\u207b\u2423\u207b\u2423\u207b": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 131}, "\u2423": {"f1-score": 0.8752661560928843, "precision": 0.9783676012461059, "recall": 0.7918229860018556, "support": 49578}},
  "ppcr": 0.8549504924601037
}
```
</details>
