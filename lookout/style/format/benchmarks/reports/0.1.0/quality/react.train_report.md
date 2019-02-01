# Train report for javascript / file:///tmp/top-repos-quality-repos-0omz14ne/react HEAD 1034e26fe5e42ba07492a736da7bdf5bf2108bc6

### Classification report

PPCR: 0.995

| Class | Precision | Recall | Full Recall | F1-score | Full F1-score | Support | Full Support | PPCR |
|------:|:----------|:-------|:------------|:---------|:---------|:--------|:-------------|:-----|
| `∅` | 0.980| 0.990| 0.990| 0.985| 0.985| 209055| 209096| 1.000 |
| `␣` | 0.948| 0.974| 0.974| 0.961| 0.961| 64761| 64761| 1.000 |
| `'` | 0.985| 0.997| 0.952| 0.991| 0.968| 25641| 26836| 0.955 |
| `⏎` | 0.943| 0.933| 0.933| 0.938| 0.938| 17264| 17264| 1.000 |
| `⏎␣⁻␣⁻` | 0.965| 0.927| 0.927| 0.945| 0.945| 14191| 14191| 1.000 |
| `⏎␣⁺␣⁺` | 0.976| 0.777| 0.777| 0.866| 0.866| 12905| 12905| 1.000 |
| `␣'` | 0.994| 0.968| 0.891| 0.981| 0.940| 6502| 7068| 0.920 |
| `⏎⏎` | 0.913| 0.854| 0.854| 0.882| 0.882| 2410| 2410| 1.000 |
| `"` | 1.000| 1.000| 1.000| 1.000| 1.000| 1347| 1347| 1.000 |
| `'␣` | 0.952| 0.923| 0.923| 0.938| 0.938| 1019| 1019| 1.000 |
| `"␣` | 1.000| 1.000| 1.000| 1.000| 1.000| 826| 826| 1.000 |
| `⏎␣⁻␣⁻␣⁻␣⁻` | 0.000| 0.000| 0.000| 0.000| 0.000| 313| 313| 1.000 |
| `'⏎␣⁻␣⁻` | 0.000| 0.000| 0.000| 0.000| 0.000| 112| 112| 1.000 |
| `"⏎` | 0.000| 0.000| 0.000| 0.000| 0.000| 0| 0| 0.000 |
| `⏎⏎'` | 0.000| 0.000| 0.000| 0.000| 0.000| 0| 3| 0.000 |
| `macro avg` | 0.710| 0.690| 0.681| 0.699| 0.695| 356346| 358151| 0.995 |
| `weighted avg` | 0.971| 0.972| 0.967| 0.971| 0.968| 356346| 358151| 0.995 |
| `micro avg` | 0.972| 0.972| 0.967| 0.972| 0.969| 356346| 358151| 0.995 |

### Confusion matrix

|refusal|  ∅| ␣| '| ⏎| ⏎␣⁺␣⁺| ⏎␣⁻␣⁻| ␣'| ⏎⏎| "| '␣| "␣| ⏎␣⁻␣⁻␣⁻␣⁻| "⏎| '⏎␣⁻␣⁻| ⏎⏎'| 
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
|0 |0 |0 |0 |0 |0 |0 |0 |0 |0 |0 |0 |0 |0 |0 |
|41 |206901 |1918 |0 |0 |143 |93 |0 |0 |0 |0 |0 |0 |0 |0 |
|0 |952 |63104 |0 |543 |80 |81 |0 |1 |0 |0 |0 |0 |0 |0 |
|1195 |0 |0 |25556 |0 |0 |0 |39 |0 |0 |46 |0 |0 |0 |0 |
|0 |244 |637 |0 |16114 |19 |54 |0 |196 |0 |0 |0 |0 |0 |0 |
|0 |2113 |736 |0 |23 |10033 |0 |0 |0 |0 |0 |0 |0 |0 |0 |
|0 |816 |144 |0 |78 |0 |13153 |0 |0 |0 |0 |0 |0 |0 |0 |
|566 |0 |0 |205 |0 |0 |0 |6297 |0 |0 |0 |0 |0 |0 |0 |
|0 |25 |14 |0 |314 |0 |0 |0 |2057 |0 |0 |0 |0 |0 |0 |
|0 |0 |0 |0 |0 |0 |0 |0 |0 |1347 |0 |0 |0 |0 |0 |
|0 |0 |0 |78 |0 |0 |0 |0 |0 |0 |941 |0 |0 |0 |0 |
|0 |0 |0 |0 |0 |0 |0 |0 |0 |0 |0 |826 |0 |0 |0 |
|0 |45 |6 |0 |9 |0 |253 |0 |0 |0 |0 |0 |0 |0 |0 |
|0 |0 |0 |111 |0 |0 |0 |0 |0 |0 |1 |0 |0 |0 |0 |
|3 |0 |0 |0 |0 |0 |0 |0 |0 |0 |0 |0 |0 |0 |0 |

### Files with most errors

| filename | number of errors|
|:----:|:-----|
| scripts/bench/benchmarks/pe-class-components/benchmark.js | 465 |
| packages/react/src/__tests__/ReactProfiler-test.internal.js | 388 |
| packages/events/__tests__/ResponderEventPlugin-test.internal.js | 297 |
| scripts/rollup/build.js | 262 |
| scripts/bench/benchmarks/pe-functional-components/benchmark.js | 235 |
| packages/react-reconciler/src/__tests__/ReactSuspenseWithNoopRenderer-test.internal.js | 232 |
| packages/react-reconciler/src/__tests__/ReactIncrementalSideEffects-test.internal.js | 204 |
| packages/react-reconciler/src/__tests__/ReactNewContext-test.internal.js | 202 |
| packages/react/src/__tests__/ReactChildren-test.js | 181 |
| fixtures/fiber-debugger/src/Fibers.js | 166 |

<details>
    <summary>Machine-readable report</summary>
```json
{
  "cl_report": {"\"": {"f1-score": 1.0, "precision": 1.0, "recall": 1.0, "support": 1347}, "\"\u23ce": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 0}, "\"\u2423": {"f1-score": 1.0, "precision": 1.0, "recall": 1.0, "support": 826}, "\u0027": {"f1-score": 0.9907154348626699, "precision": 0.9848169556840077, "recall": 0.9966849966849967, "support": 25641}, "\u0027\u23ce\u2423\u207b\u2423\u207b": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 112}, "\u0027\u2423": {"f1-score": 0.9377179870453413, "precision": 0.9524291497975709, "recall": 0.9234543670264965, "support": 1019}, "macro avg": {"f1-score": 0.6991262254764264, "precision": 0.7104309964510328, "recall": 0.6895960482682439, "support": 356346}, "micro avg": {"f1-score": 0.971889680254584, "precision": 0.971889680254584, "recall": 0.971889680254584, "support": 356346}, "weighted avg": {"f1-score": 0.9708516720290724, "precision": 0.9707826076895137, "recall": 0.971889680254584, "support": 356346}, "\u2205": {"f1-score": 0.9848887661816823, "precision": 0.980127524917573, "recall": 0.9896964913539499, "support": 209055}, "\u23ce": {"f1-score": 0.938360751201048, "precision": 0.943387389497102, "recall": 0.9333873957367933, "support": 17264}, "\u23ce\u23ce": {"f1-score": 0.8820754716981132, "precision": 0.9125998225377108, "recall": 0.8535269709543568, "support": 2410}, "\u23ce\u23ce\u0027": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 0}, "\u23ce\u2423\u207a\u2423\u207a": {"f1-score": 0.865660051768766, "precision": 0.9764476885644768, "recall": 0.7774506005424254, "support": 12905}, "\u23ce\u2423\u207b\u2423\u207b": {"f1-score": 0.9454088050314465, "precision": 0.9647205515622708, "recall": 0.9268550489747023, "support": 14191}, "\u23ce\u2423\u207b\u2423\u207b\u2423\u207b\u2423\u207b": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 313}, "\u2423": {"f1-score": 0.9610721900700578, "precision": 0.9480911672350847, "recall": 0.9744136131313599, "support": 64761}, "\u2423\u0027": {"f1-score": 0.9809939242872721, "precision": 0.993844696969697, "recall": 0.968471239618579, "support": 6502}},
  "cl_report_full": {"\"": {"f1-score": 1.0, "precision": 1.0, "recall": 1.0, "support": 1347}, "\"\u23ce": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 0}, "\"\u2423": {"f1-score": 1.0, "precision": 1.0, "recall": 1.0, "support": 826}, "\u0027": {"f1-score": 0.9682870458076006, "precision": 0.9848169556840077, "recall": 0.952302876732747, "support": 26836}, "\u0027\u23ce\u2423\u207b\u2423\u207b": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 112}, "\u0027\u2423": {"f1-score": 0.9377179870453413, "precision": 0.9524291497975709, "recall": 0.9234543670264965, "support": 1019}, "macro avg": {"f1-score": 0.6948630164240801, "precision": 0.7104309964510328, "recall": 0.6814540073836233, "support": 358151}, "micro avg": {"f1-score": 0.9694344412922657, "precision": 0.971889680254584, "recall": 0.9669915761787626, "support": 358151}, "weighted avg": {"f1-score": 0.9683733206451731, "precision": 0.970858818511679, "recall": 0.9669915761787626, "support": 358151}, "\u2205": {"f1-score": 0.9847926662097326, "precision": 0.980127524917573, "recall": 0.9895024295060642, "support": 209096}, "\u23ce": {"f1-score": 0.938360751201048, "precision": 0.943387389497102, "recall": 0.9333873957367933, "support": 17264}, "\u23ce\u23ce": {"f1-score": 0.8820754716981132, "precision": 0.9125998225377108, "recall": 0.8535269709543568, "support": 2410}, "\u23ce\u23ce\u0027": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 3}, "\u23ce\u2423\u207a\u2423\u207a": {"f1-score": 0.865660051768766, "precision": 0.9764476885644768, "recall": 0.7774506005424254, "support": 12905}, "\u23ce\u2423\u207b\u2423\u207b": {"f1-score": 0.9454088050314465, "precision": 0.9647205515622708, "recall": 0.9268550489747023, "support": 14191}, "\u23ce\u2423\u207b\u2423\u207b\u2423\u207b\u2423\u207b": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 313}, "\u2423": {"f1-score": 0.9610721900700578, "precision": 0.9480911672350847, "recall": 0.9744136131313599, "support": 64761}, "\u2423\u0027": {"f1-score": 0.9395702775290959, "precision": 0.993844696969697, "recall": 0.8909168081494058, "support": 7068}},
  "ppcr": 0.9949602262732758
}
```
</details>
