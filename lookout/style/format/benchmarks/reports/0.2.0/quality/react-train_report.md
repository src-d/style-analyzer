# Train report for javascript / file:///tmp/top-repos-quality-repos-6e5mbbl_/react HEAD 1034e26fe5e42ba07492a736da7bdf5bf2108bc6

### Classification report

PPCR: 0.898

| Class | Precision | Recall | Full Recall | F1-score | Full F1-score | Support | Full Support | PPCR |
|------:|:----------|:-------|:------------|:---------|:---------|:--------|:-------------|:-----|
| `∅` | 0.983| 0.996| 0.962| 0.990| 0.972| 204508| 211773| 0.966 |
| `␣` | 0.977| 0.981| 0.883| 0.979| 0.927| 68625| 76285| 0.900 |
| `'` | 1.000| 1.000| 0.972| 1.000| 0.986| 34489| 35457| 0.973 |
| `⏎` | 0.937| 0.950| 0.470| 0.943| 0.626| 13751| 27799| 0.495 |
| `⏎␣⁻␣⁻` | 0.981| 0.931| 0.841| 0.955| 0.905| 13501| 14956| 0.903 |
| `⏎␣⁺␣⁺` | 0.986| 0.826| 0.653| 0.899| 0.786| 12221| 15443| 0.791 |
| `"` | 0.995| 1.000| 1.000| 0.998| 0.998| 2634| 2634| 1.000 |
| `⏎⏎` | 0.946| 0.819| 0.191| 0.878| 0.318| 1637| 7000| 0.234 |
| `⏎␣⁻␣⁻␣⁻␣⁻` | 0.000| 0.000| 0.000| 0.000| 0.000| 248| 398| 0.623 |
| `⏎␣⁺␣⁺␣⁺␣⁺` | 0.000| 0.000| 0.000| 0.000| 0.000| 68| 90| 0.756 |
| `micro avg` | 0.982| 0.982| 0.881| 0.982| 0.929| 351682| 391835| 0.898 |
| `macro avg` | 0.780| 0.750| 0.597| 0.764| 0.652| 351682| 391835| 0.898 |
| `weighted avg` | 0.981| 0.982| 0.881| 0.981| 0.918| 351682| 391835| 0.898 |

### Confusion matrix

|refusal|  ∅| ␣| '| ⏎| ⏎␣⁺␣⁺| ⏎␣⁻␣⁻| ⏎⏎| "| ⏎␣⁻␣⁻␣⁻␣⁻| ⏎␣⁺␣⁺␣⁺␣⁺| 
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
|0 |0 |0 |0 |0 |0 |0 |0 |0 |0 |0 |
|7265 |203719 |587 |0 |0 |106 |96 |0 |0 |0 |0 |
|7660 |783 |67346 |0 |479 |13 |4 |0 |0 |0 |0 |
|968 |0 |0 |34477 |0 |0 |0 |0 |12 |0 |0 |
|14048 |267 |336 |0 |13058 |13 |0 |77 |0 |0 |0 |
|3222 |1680 |443 |0 |7 |10091 |0 |0 |0 |0 |0 |
|1455 |607 |172 |0 |151 |0 |12571 |0 |0 |0 |0 |
|5363 |67 |6 |0 |224 |0 |0 |1340 |0 |0 |0 |
|0 |0 |0 |0 |0 |0 |0 |0 |2634 |0 |0 |
|150 |88 |5 |0 |10 |0 |145 |0 |0 |0 |0 |
|22 |4 |48 |0 |0 |16 |0 |0 |0 |0 |0 |

### Files with most errors

| filename | number of errors|
|:----:|:-----|
| packages/react/src/__tests__/ReactProfiler-test.internal.js | 292 |
| scripts/bench/benchmarks/pe-class-components/benchmark.js | 279 |
| scripts/bench/benchmarks/pe-functional-components/benchmark.js | 275 |
| scripts/bench/benchmarks/pe-no-components/benchmark.js | 178 |
| packages/react-reconciler/src/__tests__/ReactSuspenseWithNoopRenderer-test.internal.js | 174 |
| packages/react-reconciler/src/__tests__/ReactNewContext-test.internal.js | 171 |
| packages/events/__tests__/ResponderEventPlugin-test.internal.js | 129 |
| packages/react-reconciler/src/__tests__/ReactIncremental-test.internal.js | 126 |
| scripts/rollup/wrappers.js | 106 |
| fixtures/attribute-behavior/src/attributes.js | 101 |

<details>
    <summary>Machine-readable report</summary>
```json
{
  "cl_report": {"\"": {"f1-score": 0.9977272727272727, "precision": 0.9954648526077098, "recall": 1.0, "support": 2634}, "\u0027": {"f1-score": 0.9998260012179915, "precision": 1.0, "recall": 0.9996520629766013, "support": 34489}, "macro avg": {"f1-score": 0.7641201830633246, "precision": 0.7804986593839134, "recall": 0.7502156778833013, "support": 351682}, "micro avg": {"f1-score": 0.9816709413617984, "precision": 0.9816709413617984, "recall": 0.9816709413617984, "support": 351682}, "weighted avg": {"f1-score": 0.980917923645695, "precision": 0.9808023250720342, "recall": 0.9816709413617984, "support": 351682}, "\u2205": {"f1-score": 0.9895925172992521, "precision": 0.98312863451005, "recall": 0.9961419602167152, "support": 204508}, "\u23ce": {"f1-score": 0.9434971098265896, "precision": 0.9374685907100294, "recall": 0.9496036651879863, "support": 13751}, "\u23ce\u23ce": {"f1-score": 0.8775376555337264, "precision": 0.9456598447424136, "recall": 0.8185705558949298, "support": 1637}, "\u23ce\u2423\u207a\u2423\u207a": {"f1-score": 0.8985752448797864, "precision": 0.9855454634241625, "recall": 0.8257098437116439, "support": 12221}, "\u23ce\u2423\u207a\u2423\u207a\u2423\u207a\u2423\u207a": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 68}, "\u23ce\u2423\u207b\u2423\u207b": {"f1-score": 0.9553520538055249, "precision": 0.9808832709113608, "recall": 0.9311162136138064, "support": 13501}, "\u23ce\u2423\u207b\u2423\u207b\u2423\u207b\u2423\u207b": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 248}, "\u2423": {"f1-score": 0.9790939753431032, "precision": 0.9768359369334088, "recall": 0.9813624772313297, "support": 68625}},
  "cl_report_full": {"\"": {"f1-score": 0.9977272727272727, "precision": 0.9954648526077098, "recall": 1.0, "support": 2634}, "\u0027": {"f1-score": 0.9859867875425401, "precision": 1.0, "recall": 0.9723608878359703, "support": 35457}, "macro avg": {"f1-score": 0.65190081330252, "precision": 0.7804986593839134, "recall": 0.5972275739422889, "support": 391835}, "micro avg": {"f1-score": 0.9286566413410857, "precision": 0.9816709413617984, "recall": 0.881074942258859, "support": 391835}, "weighted avg": {"f1-score": 0.9176735438313828, "precision": 0.9783895293799311, "recall": 0.881074942258859, "support": 391835}, "\u2205": {"f1-score": 0.9724335780499681, "precision": 0.98312863451005, "recall": 0.9619687117810108, "support": 211773}, "\u23ce": {"f1-score": 0.6258627300613497, "precision": 0.9374685907100294, "recall": 0.46972912694701247, "support": 27799}, "\u23ce\u23ce": {"f1-score": 0.3184032315551859, "precision": 0.9456598447424136, "recall": 0.19142857142857142, "support": 7000}, "\u23ce\u2423\u207a\u2423\u207a": {"f1-score": 0.7858422241258469, "precision": 0.9855454634241625, "recall": 0.6534352133652788, "support": 15443}, "\u23ce\u2423\u207a\u2423\u207a\u2423\u207a\u2423\u207a": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 90}, "\u23ce\u2423\u207b\u2423\u207b": {"f1-score": 0.9053003024629123, "precision": 0.9808832709113608, "recall": 0.840532227868414, "support": 14956}, "\u23ce\u2423\u207b\u2423\u207b\u2423\u207b\u2423\u207b": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 398}, "\u2423": {"f1-score": 0.9274520065001239, "precision": 0.9768359369334088, "recall": 0.882821000196631, "support": 76285}},
  "ppcr": 0.897525744254597
}
```
</details>
