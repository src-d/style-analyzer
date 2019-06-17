# Train report for javascript / file:///tmp/top-repos-quality-repos-oxz90m02/core-js HEAD 4a85fe5f9678296bc9ffd5cfc44b32d34b18e52f

### Classification report

PPCR: 0.974

| Class | Precision | Recall | Full Recall | F1-score | Full F1-score | Support | Full Support | PPCR |
|------:|:----------|:-------|:------------|:---------|:---------|:--------|:-------------|:-----|
| `∅` | 0.987| 0.998| 0.991| 0.992| 0.989| 162603| 163622| 0.994 |
| `␣` | 0.982| 0.972| 0.935| 0.977| 0.958| 73773| 76736| 0.961 |
| `'` | 1.000| 1.000| 0.992| 1.000| 0.996| 34667| 34952| 0.992 |
| `⏎` | 0.956| 0.961| 0.844| 0.959| 0.897| 19863| 22610| 0.879 |
| `⏎␣⁺␣⁺` | 0.959| 0.933| 0.906| 0.946| 0.932| 5030| 5182| 0.971 |
| `⏎␣⁻␣⁻` | 0.975| 0.956| 0.947| 0.965| 0.961| 4969| 5018| 0.990 |
| `⏎⏎` | 0.987| 0.469| 0.283| 0.636| 0.439| 1514| 2513| 0.602 |
| `"` | 0.000| 0.000| 0.000| 0.000| 0.000| 0| 0| 0.000 |
| `micro avg` | 0.985| 0.985| 0.959| 0.985| 0.972| 302419| 310633| 0.974 |
| `weighted avg` | 0.985| 0.985| 0.959| 0.984| 0.970| 302419| 310633| 0.974 |
| `macro avg` | 0.856| 0.786| 0.737| 0.809| 0.771| 302419| 310633| 0.974 |

### Confusion matrix

|refusal|  ∅| ␣| '| ⏎| ⏎␣⁺␣⁺| ⏎␣⁻␣⁻| ⏎⏎| "| 
|:---|:---|:---|:---|:---|:---|:---|:---|
|0 |0 |0 |0 |0 |0 |0 |0 |
|1019 |162197 |401 |0 |5 |0 |0 |0 |
|2963 |1681 |71712 |0 |78 |200 |102 |0 |
|285 |0 |0 |34667 |0 |0 |0 |0 |
|2747 |231 |509 |0 |19094 |0 |20 |9 |
|152 |110 |206 |0 |19 |4695 |0 |0 |
|49 |39 |156 |0 |23 |0 |4751 |0 |
|999 |45 |9 |0 |750 |0 |0 |710 |

### Files with most errors

| filename | number of errors|
|:----:|:-----|
| tests/compat/tests.js | 221 |
| .eslintrc.js | 220 |
| packages/core-js/modules/web.url.js | 197 |
| tests/pure/web.url.js | 77 |
| packages/core-js/modules/es.string.split.js | 76 |
| tests/tests/web.url.js | 76 |
| packages/core-js/modules/es.symbol.js | 72 |
| packages/core-js/modules/web.url-search-params.js | 61 |
| packages/core-js/internals/array-buffer.js | 60 |
| packages/core-js/internals/collection.js | 57 |

<details>
    <summary>Machine-readable report</summary>
```json
{
  "cl_report": {"\"": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 0}, "\u0027": {"f1-score": 1.0, "precision": 1.0, "recall": 1.0, "support": 34667}, "macro avg": {"f1-score": 0.8094671558178868, "precision": 0.8559254195734001, "recall": 0.7861668593324225, "support": 302419}, "micro avg": {"f1-score": 0.9848124621799557, "precision": 0.9848124621799557, "recall": 0.9848124621799557, "support": 302419}, "weighted avg": {"f1-score": 0.9843158603019486, "precision": 0.9847955175339702, "recall": 0.9848124621799557, "support": 302419}, "\u2205": {"f1-score": 0.9923158339094418, "precision": 0.9871822182187787, "recall": 0.9975031210986267, "support": 162603}, "\u23ce": {"f1-score": 0.9587266519381402, "precision": 0.9561820822274526, "recall": 0.9612848008860696, "support": 19863}, "\u23ce\u23ce": {"f1-score": 0.6359158083296015, "precision": 0.9874826147426982, "recall": 0.46895640686922063, "support": 1514}, "\u23ce\u2423\u207a\u2423\u207a": {"f1-score": 0.9460957178841309, "precision": 0.9591419816138917, "recall": 0.9333996023856859, "support": 5030}, "\u23ce\u2423\u207b\u2423\u207b": {"f1-score": 0.9654541759804918, "precision": 0.9749640878309049, "recall": 0.9561279935600725, "support": 4969}, "\u2423": {"f1-score": 0.9772290585012878, "precision": 0.982450371953475, "recall": 0.9720629498597048, "support": 73773}},
  "cl_report_full": {"\"": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 0}, "\u0027": {"f1-score": 0.9959062899495827, "precision": 1.0, "recall": 0.9918459601739529, "support": 34952}, "macro avg": {"f1-score": 0.7714694899639755, "precision": 0.8559254195734001, "recall": 0.7371878068360644, "support": 310633}, "micro avg": {"f1-score": 0.9716174158146454, "precision": 0.9848124621799557, "recall": 0.9587712831540757, "support": 310633}, "weighted avg": {"f1-score": 0.9696508926560726, "precision": 0.9845364300737032, "recall": 0.9587712831540757, "support": 310633}, "\u2205": {"f1-score": 0.9892322939696577, "precision": 0.9871822182187787, "recall": 0.9912909022014155, "support": 163622}, "\u23ce": {"f1-score": 0.896874045891167, "precision": 0.9561820822274526, "recall": 0.8444935869084476, "support": 22610}, "\u23ce\u23ce": {"f1-score": 0.43935643564356436, "precision": 0.9874826147426982, "recall": 0.2825308396339037, "support": 2513}, "\u23ce\u2423\u207a\u2423\u207a": {"f1-score": 0.9318249479011611, "precision": 0.9591419816138917, "recall": 0.9060208413739869, "support": 5182}, "\u23ce\u2423\u207b\u2423\u207b": {"f1-score": 0.9606713173592154, "precision": 0.9749640878309049, "recall": 0.9467915504184934, "support": 5018}, "\u2423": {"f1-score": 0.9578905889974554, "precision": 0.982450371953475, "recall": 0.9345287739783152, "support": 76736}},
  "ppcr": 0.9735572202567017
}
```
</details>
