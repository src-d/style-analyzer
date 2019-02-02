# Train report for javascript / file:///tmp/top-repos-quality-repos-uxy0l7qn/storybook HEAD b28217f887af533a17cb1498887d6b4bd41bd643

### Classification report

PPCR: 0.997

| Class | Precision | Recall | Full Recall | F1-score | Full F1-score | Support | Full Support | PPCR |
|------:|:----------|:-------|:------------|:---------|:---------|:--------|:-------------|:-----|
| `∅` | 0.958| 0.992| 0.990| 0.975| 0.974| 91836| 91948| 0.999 |
| `␣` | 0.957| 0.964| 0.964| 0.960| 0.960| 39794| 39794| 1.000 |
| `'` | 0.988| 0.999| 0.999| 0.993| 0.993| 12013| 12013| 1.000 |
| `⏎␣⁻␣⁻` | 0.930| 0.904| 0.904| 0.917| 0.917| 5532| 5532| 1.000 |
| `␣'` | 0.998| 0.996| 0.996| 0.997| 0.997| 5403| 5403| 1.000 |
| `⏎␣⁺␣⁺` | 0.958| 0.737| 0.737| 0.833| 0.833| 4138| 4138| 1.000 |
| `⏎` | 0.960| 0.387| 0.387| 0.552| 0.552| 3559| 3559| 1.000 |
| `⏎⏎` | 0.969| 0.862| 0.862| 0.912| 0.912| 1508| 1508| 1.000 |
| `"` | 1.000| 1.000| 0.791| 1.000| 0.883| 417| 527| 0.791 |
| `'␣` | 0.000| 0.000| 0.000| 0.000| 0.000| 129| 129| 1.000 |
| `⏎␣⁻␣⁻␣⁻␣⁻` | 0.000| 0.000| 0.000| 0.000| 0.000| 74| 74| 1.000 |
| `"␣` | 0.000| 0.000| 0.000| 0.000| 0.000| 0| 197| 0.000 |
| `"⏎` | 0.000| 0.000| 0.000| 0.000| 0.000| 0| 107| 0.000 |
| `macro avg` | 0.671| 0.603| 0.587| 0.626| 0.617| 164403| 164929| 0.997 |
| `weighted avg` | 0.959| 0.961| 0.958| 0.957| 0.954| 164403| 164929| 0.997 |
| `micro avg` | 0.961| 0.961| 0.958| 0.961| 0.959| 164403| 164929| 0.997 |

### Confusion matrix

|refusal|  ∅| ␣| '| ⏎| ⏎␣⁺␣⁺| ⏎␣⁻␣⁻| ␣'| ⏎⏎| "| '␣| "␣| ⏎␣⁻␣⁻␣⁻␣⁻| "⏎| 
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
|0 |0 |0 |0 |0 |0 |0 |0 |0 |0 |0 |0 |0 |0 |
|112 |91063 |520 |0 |3 |45 |205 |0 |0 |0 |0 |0 |0 |0 |
|0 |1236 |38349 |0 |3 |90 |115 |0 |1 |0 |0 |0 |0 |0 |
|0 |0 |0 |12001 |0 |0 |0 |12 |0 |0 |0 |0 |0 |0 |
|0 |1309 |831 |0 |1379 |0 |0 |0 |40 |0 |0 |0 |0 |0 |
|0 |878 |208 |0 |4 |3048 |0 |0 |0 |0 |0 |0 |0 |0 |
|0 |505 |25 |0 |1 |0 |5001 |0 |0 |0 |0 |0 |0 |0 |
|0 |0 |0 |22 |0 |0 |0 |5381 |0 |0 |0 |0 |0 |0 |
|0 |17 |145 |0 |46 |0 |0 |0 |1300 |0 |0 |0 |0 |0 |
|110 |0 |0 |0 |0 |0 |0 |0 |0 |417 |0 |0 |0 |0 |
|0 |0 |0 |129 |0 |0 |0 |0 |0 |0 |0 |0 |0 |0 |
|197 |0 |0 |0 |0 |0 |0 |0 |0 |0 |0 |0 |0 |0 |
|0 |14 |0 |0 |0 |0 |59 |0 |1 |0 |0 |0 |0 |0 |
|107 |0 |0 |0 |0 |0 |0 |0 |0 |0 |0 |0 |0 |0 |

### Files with most errors

| filename | number of errors|
|:----:|:-----|
| examples/official-storybook/stories/addon-info.stories.js | 142 |
| lib/components/src/layout/desktop.js | 91 |
| lib/ui/src/modules/ui/components/stories_panel/stories_tree/index.test.js | 88 |
| addons/jest/src/components/Result.js | 81 |
| lib/codemod/src/transforms/__testfixtures__/update-addon-info/update-addon-info.input.js | 78 |
| examples/official-storybook/stories/addon-actions.stories.js | 68 |
| lib/cli/lib/initiate.js | 64 |
| examples/official-storybook/stories/addon-a11y.stories.js | 63 |
| lib/components/src/tabs/tabs.stories.js | 61 |
| addons/info/src/components/Story.js | 60 |

<details>
    <summary>Machine-readable report</summary>
```json
{
  "cl_report": {"\"": {"f1-score": 1.0, "precision": 1.0, "recall": 1.0, "support": 417}, "\"\u23ce": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 0}, "\"\u2423": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 0}, "\u0027": {"f1-score": 0.9932547072211876, "precision": 0.9875740618828176, "recall": 0.9990010821609923, "support": 12013}, "\u0027\u2423": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 129}, "macro avg": {"f1-score": 0.626058025070248, "precision": 0.6705149972251998, "recall": 0.6031029368582835, "support": 164403}, "micro avg": {"f1-score": 0.9606819826888804, "precision": 0.9606819826888804, "recall": 0.9606819826888804, "support": 164403}, "weighted avg": {"f1-score": 0.9568866102317921, "precision": 0.9594838403378708, "recall": 0.9606819826888804, "support": 164403}, "\u2205": {"f1-score": 0.9746759571439274, "precision": 0.9583359643030035, "recall": 0.9915828215514613, "support": 91836}, "\u23ce": {"f1-score": 0.5521521521521522, "precision": 0.9603064066852368, "recall": 0.38746838999719024, "support": 3559}, "\u23ce\u23ce": {"f1-score": 0.9122807017543859, "precision": 0.9687034277198212, "recall": 0.8620689655172413, "support": 1508}, "\u23ce\u2423\u207a\u2423\u207a": {"f1-score": 0.8326731320857805, "precision": 0.9575871819038643, "recall": 0.736587723537941, "support": 4138}, "\u23ce\u2423\u207b\u2423\u207b": {"f1-score": 0.9166055718475073, "precision": 0.9295539033457249, "recall": 0.9040130151843818, "support": 5532}, "\u23ce\u2423\u207b\u2423\u207b\u2423\u207b\u2423\u207b": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 74}, "\u2423": {"f1-score": 0.9602614182692307, "precision": 0.9568591247068217, "recall": 0.9636879931647987, "support": 39794}, "\u2423\u0027": {"f1-score": 0.9968506854390515, "precision": 0.9977748933803078, "recall": 0.9959281880436794, "support": 5403}},
  "cl_report_full": {"\"": {"f1-score": 0.8834745762711864, "precision": 1.0, "recall": 0.7912713472485768, "support": 527}, "\"\u23ce": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 107}, "\"\u2423": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 197}, "\u0027": {"f1-score": 0.9932547072211876, "precision": 0.9875740618828176, "recall": 0.9990010821609923, "support": 12013}, "\u0027\u2423": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 129}, "macro avg": {"f1-score": 0.6170496188751873, "precision": 0.6705149972251998, "recall": 0.5869539768782266, "support": 164929}, "micro avg": {"f1-score": 0.959147607885052, "precision": 0.9606819826888804, "recall": 0.9576181265878044, "support": 164929}, "weighted avg": {"f1-score": 0.9544658594009822, "precision": 0.9577415459444302, "recall": 0.9576181265878044, "support": 164929}, "\u2205": {"f1-score": 0.9740921003369524, "precision": 0.9583359643030035, "recall": 0.9903749945621438, "support": 91948}, "\u23ce": {"f1-score": 0.5521521521521522, "precision": 0.9603064066852368, "recall": 0.38746838999719024, "support": 3559}, "\u23ce\u23ce": {"f1-score": 0.9122807017543859, "precision": 0.9687034277198212, "recall": 0.8620689655172413, "support": 1508}, "\u23ce\u2423\u207a\u2423\u207a": {"f1-score": 0.8326731320857805, "precision": 0.9575871819038643, "recall": 0.736587723537941, "support": 4138}, "\u23ce\u2423\u207b\u2423\u207b": {"f1-score": 0.9166055718475073, "precision": 0.9295539033457249, "recall": 0.9040130151843818, "support": 5532}, "\u23ce\u2423\u207b\u2423\u207b\u2423\u207b\u2423\u207b": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 74}, "\u2423": {"f1-score": 0.9602614182692307, "precision": 0.9568591247068217, "recall": 0.9636879931647987, "support": 39794}, "\u2423\u0027": {"f1-score": 0.9968506854390515, "precision": 0.9977748933803078, "recall": 0.9959281880436794, "support": 5403}},
  "ppcr": 0.9968107488676946
}
```
</details>
