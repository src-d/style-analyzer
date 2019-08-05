# Train report for javascript / file:///tmp/top-repos-quality-repos-qazqevzg/react HEAD 1034e26fe5e42ba07492a736da7bdf5bf2108bc6

### Classification report

PPCR: 0.898

| Class | Precision | Recall | Full Recall | F1-score | Full F1-score | Support | Full Support | PPCR |
|------:|:----------|:-------|:------------|:---------|:---------|:--------|:-------------|:-----|
| `∅` | 0.983| 0.996| 0.962| 0.990| 0.972| 204512| 211773| 0.966 |
| `␣` | 0.977| 0.981| 0.883| 0.979| 0.928| 68620| 76285| 0.900 |
| `'` | 1.000| 1.000| 0.972| 1.000| 0.986| 34479| 35457| 0.972 |
| `⏎` | 0.937| 0.949| 0.470| 0.943| 0.626| 13763| 27799| 0.495 |
| `⏎␣⁻␣⁻` | 0.981| 0.931| 0.841| 0.955| 0.905| 13501| 14956| 0.903 |
| `⏎␣⁺␣⁺` | 0.986| 0.826| 0.653| 0.899| 0.786| 12221| 15443| 0.791 |
| `"` | 1.000| 1.000| 1.000| 1.000| 1.000| 2634| 2634| 1.000 |
| `⏎⏎` | 0.946| 0.821| 0.191| 0.879| 0.318| 1633| 7000| 0.233 |
| `⏎␣⁻␣⁻␣⁻␣⁻` | 0.000| 0.000| 0.000| 0.000| 0.000| 248| 398| 0.623 |
| `⏎␣⁺␣⁺␣⁺␣⁺` | 0.000| 0.000| 0.000| 0.000| 0.000| 68| 90| 0.756 |
| `weighted avg` | 0.981| 0.982| 0.881| 0.981| 0.918| 351679| 391835| 0.898 |
| `macro avg` | 0.781| 0.750| 0.597| 0.764| 0.652| 351679| 391835| 0.898 |
| `micro avg` | 0.982| 0.982| 0.881| 0.982| 0.929| 351679| 391835| 0.898 |

### Confusion matrix

|refusal|  ∅| ␣| '| ⏎| ⏎␣⁺␣⁺| ⏎␣⁻␣⁻| ⏎⏎| "| ⏎␣⁻␣⁻␣⁻␣⁻| ⏎␣⁺␣⁺␣⁺␣⁺| 
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
|0 |0 |0 |0 |0 |0 |0 |0 |0 |0 |0 |
|7261 |203725 |585 |0 |0 |106 |96 |0 |0 |0 |0 |
|7665 |783 |67341 |0 |479 |13 |4 |0 |0 |0 |0 |
|978 |0 |0 |34479 |0 |0 |0 |0 |0 |0 |0 |
|14036 |286 |329 |0 |13058 |13 |0 |77 |0 |0 |0 |
|3222 |1680 |443 |0 |7 |10091 |0 |0 |0 |0 |0 |
|1455 |607 |172 |0 |151 |0 |12571 |0 |0 |0 |0 |
|5367 |68 |1 |0 |224 |0 |0 |1340 |0 |0 |0 |
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
| packages/react-reconciler/src/__tests__/ReactIncremental-test.internal.js | 126 |
| packages/events/__tests__/ResponderEventPlugin-test.internal.js | 124 |
| fixtures/attribute-behavior/src/attributes.js | 101 |
| fixtures/fiber-debugger/src/Fibers.js | 97 |

<details>
    <summary>Machine-readable report</summary>
```json
{
  "cl_report": {"\"": {"f1-score": 1.0, "precision": 1.0, "recall": 1.0, "support": 2634}, "\u0027": {"f1-score": 1.0, "precision": 1.0, "recall": 1.0, "support": 34479}, "macro avg": {"f1-score": 0.764444575293271, "precision": 0.7809624088611707, "recall": 0.7503690322235951, "support": 351679}, "micro avg": {"f1-score": 0.9816878460186704, "precision": 0.9816878460186704, "recall": 0.9816878460186704, "support": 351679}, "weighted avg": {"f1-score": 0.9809350021058704, "precision": 0.9808182476112197, "recall": 0.9816878460186704, "support": 351679}, "\u2205": {"f1-score": 0.9895495600517787, "precision": 0.9830342451541925, "recall": 0.9961518150524175, "support": 204512}, "\u23ce": {"f1-score": 0.9430882565361838, "precision": 0.9374685907100294, "recall": 0.9487757029717359, "support": 13763}, "\u23ce\u23ce": {"f1-score": 0.878688524590164, "precision": 0.9456598447424136, "recall": 0.8205756276791182, "support": 1633}, "\u23ce\u2423\u207a\u2423\u207a": {"f1-score": 0.8985752448797864, "precision": 0.9855454634241625, "recall": 0.8257098437116439, "support": 12221}, "\u23ce\u2423\u207a\u2423\u207a\u2423\u207a\u2423\u207a": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 68}, "\u23ce\u2423\u207b\u2423\u207b": {"f1-score": 0.9553520538055249, "precision": 0.9808832709113608, "recall": 0.9311162136138064, "support": 13501}, "\u23ce\u2423\u207b\u2423\u207b\u2423\u207b\u2423\u207b": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 248}, "\u2423": {"f1-score": 0.9791921130692725, "precision": 0.977032673669549, "recall": 0.9813611192072282, "support": 68620}},
  "cl_report_full": {"\"": {"f1-score": 1.0, "precision": 1.0, "recall": 1.0, "support": 2634}, "\u0027": {"f1-score": 0.986015785861359, "precision": 1.0, "recall": 0.9724172941873255, "support": 35457}, "macro avg": {"f1-score": 0.6521330644376715, "precision": 0.7809624088611707, "recall": 0.5972294934313032, "support": 391835}, "micro avg": {"f1-score": 0.9286684581594966, "precision": 0.9816878460186704, "recall": 0.881082598542754, "support": 391835}, "weighted avg": {"f1-score": 0.9176845307467933, "precision": 0.9784073035019205, "recall": 0.881082598542754, "support": 391835}, "\u2205": {"f1-score": 0.9724018767869331, "precision": 0.9830342451541925, "recall": 0.9619970440046653, "support": 211773}, "\u23ce": {"f1-score": 0.6258627300613497, "precision": 0.9374685907100294, "recall": 0.46972912694701247, "support": 27799}, "\u23ce\u23ce": {"f1-score": 0.3184032315551859, "precision": 0.9456598447424136, "recall": 0.19142857142857142, "support": 7000}, "\u23ce\u2423\u207a\u2423\u207a": {"f1-score": 0.7858422241258469, "precision": 0.9855454634241625, "recall": 0.6534352133652788, "support": 15443}, "\u23ce\u2423\u207a\u2423\u207a\u2423\u207a\u2423\u207a": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 90}, "\u23ce\u2423\u207b\u2423\u207b": {"f1-score": 0.9053003024629123, "precision": 0.9808832709113608, "recall": 0.840532227868414, "support": 14956}, "\u23ce\u2423\u207b\u2423\u207b\u2423\u207b\u2423\u207b": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 398}, "\u2423": {"f1-score": 0.9275044935231287, "precision": 0.977032673669549, "recall": 0.8827554565117651, "support": 76285}},
  "ppcr": 0.897518087970702
}
```
</details>
