# Train report for javascript / file:///tmp/top-repos-quality-repos-0bn8dtxb/core-js HEAD 4a85fe5f9678296bc9ffd5cfc44b32d34b18e52f

### Classification report

PPCR: 0.975

| Class | Precision | Recall | Full Recall | F1-score | Full F1-score | Support | Full Support | PPCR |
|------:|:----------|:-------|:------------|:---------|:---------|:--------|:-------------|:-----|
| `∅` | 0.987| 0.998| 0.991| 0.992| 0.989| 162603| 163622| 0.994 |
| `␣` | 0.982| 0.972| 0.938| 0.977| 0.960| 74074| 76736| 0.965 |
| `'` | 1.000| 1.000| 0.992| 1.000| 0.996| 34667| 34952| 0.992 |
| `⏎` | 0.956| 0.961| 0.844| 0.959| 0.897| 19863| 22610| 0.879 |
| `⏎␣⁺␣⁺` | 0.959| 0.932| 0.906| 0.946| 0.932| 5034| 5182| 0.971 |
| `⏎␣⁻␣⁻` | 0.975| 0.956| 0.947| 0.965| 0.961| 4969| 5018| 0.990 |
| `⏎⏎` | 0.987| 0.469| 0.283| 0.636| 0.439| 1514| 2513| 0.602 |
| `"` | 0.000| 0.000| 0.000| 0.000| 0.000| 0| 0| 0.000 |
| `micro avg` | 0.985| 0.985| 0.960| 0.985| 0.972| 302724| 310633| 0.975 |
| `macro avg` | 0.856| 0.786| 0.738| 0.809| 0.772| 302724| 310633| 0.975 |
| `weighted avg` | 0.985| 0.985| 0.960| 0.984| 0.970| 302724| 310633| 0.975 |

### Confusion matrix

|refusal|  ∅| ␣| '| ⏎| ⏎␣⁺␣⁺| ⏎␣⁻␣⁻| ⏎⏎| "| 
|:---|:---|:---|:---|:---|:---|:---|:---|
|0 |0 |0 |0 |0 |0 |0 |0 |
|1019 |162197 |401 |0 |5 |0 |0 |0 |
|2662 |1681 |72013 |0 |78 |200 |102 |0 |
|285 |0 |0 |34667 |0 |0 |0 |0 |
|2747 |231 |509 |0 |19094 |0 |20 |9 |
|148 |110 |211 |0 |19 |4694 |0 |0 |
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
  "cl_report": {"\"": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 0}, "\u0027": {"f1-score": 1.0, "precision": 1.0, "recall": 1.0, "support": 34667}, "macro avg": {"f1-score": 0.809413721305941, "precision": 0.8559250072939018, "recall": 0.7860635089384282, "support": 302724}, "micro avg": {"f1-score": 0.9848112472086785, "precision": 0.9848112472086785, "recall": 0.9848112472086785, "support": 302724}, "weighted avg": {"f1-score": 0.9843148524926572, "precision": 0.9847939437255662, "recall": 0.9848112472086785, "support": 302724}, "\u2205": {"f1-score": 0.9923158339094418, "precision": 0.9871822182187787, "recall": 0.9975031210986267, "support": 162603}, "\u23ce": {"f1-score": 0.9587266519381402, "precision": 0.9561820822274526, "recall": 0.9612848008860696, "support": 19863}, "\u23ce\u23ce": {"f1-score": 0.6359158083296015, "precision": 0.9874826147426982, "recall": 0.46895640686922063, "support": 1514}, "\u23ce\u2423\u207a\u2423\u207a": {"f1-score": 0.9456083803384367, "precision": 0.9591336330200245, "recall": 0.9324592769169646, "support": 5034}, "\u23ce\u2423\u207b\u2423\u207b": {"f1-score": 0.9654541759804918, "precision": 0.9749640878309049, "recall": 0.9561279935600725, "support": 4969}, "\u2423": {"f1-score": 0.9772889199514159, "precision": 0.9824554223113549, "recall": 0.9721764721764722, "support": 74074}},
  "cl_report_full": {"\"": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 0}, "\u0027": {"f1-score": 0.9959062899495827, "precision": 1.0, "recall": 0.9918459601739529, "support": 34952}, "macro avg": {"f1-score": 0.7717135830064668, "precision": 0.8559250072939018, "recall": 0.7376540023274748, "support": 310633}, "micro avg": {"f1-score": 0.9721124891376474, "precision": 0.9848112472086785, "recall": 0.9597370530497403, "support": 310633}, "weighted avg": {"f1-score": 0.9701577008802758, "precision": 0.9845375383971875, "recall": 0.9597370530497403, "support": 310633}, "\u2205": {"f1-score": 0.9892322939696577, "precision": 0.9871822182187787, "recall": 0.9912909022014155, "support": 163622}, "\u23ce": {"f1-score": 0.896874045891167, "precision": 0.9561820822274526, "recall": 0.8444935869084476, "support": 22610}, "\u23ce\u23ce": {"f1-score": 0.43935643564356436, "precision": 0.9874826147426982, "recall": 0.2825308396339037, "support": 2513}, "\u23ce\u2423\u207a\u2423\u207a": {"f1-score": 0.9317189360857483, "precision": 0.9591336330200245, "recall": 0.9058278656889232, "support": 5182}, "\u23ce\u2423\u207b\u2423\u207b": {"f1-score": 0.9606713173592154, "precision": 0.9749640878309049, "recall": 0.9467915504184934, "support": 5018}, "\u2423": {"f1-score": 0.9599493451527978, "precision": 0.9824554223113549, "recall": 0.9384513135946623, "support": 76736}},
  "ppcr": 0.974539086317294
}
```
</details>
