# Train report for javascript / file:///tmp/top-repos-quality-repos-mu72yfu8/axios HEAD 21ae22dbd3ae3d3a55d9efd4eead3dd7fb6d8e6e

### Classification report

PPCR: 0.776

| Class | Precision | Recall | Full Recall | F1-score | Full F1-score | Support | Full Support | PPCR |
|------:|:----------|:-------|:------------|:---------|:---------|:--------|:-------------|:-----|
| `∅` | 0.982| 0.995| 0.940| 0.989| 0.961| 13764| 14575| 0.944 |
| `␣` | 0.968| 0.949| 0.548| 0.958| 0.700| 3359| 5819| 0.577 |
| `'` | 0.977| 1.000| 0.993| 0.989| 0.985| 2851| 2870| 0.993 |
| `⏎␣⁻␣⁻` | 0.991| 0.959| 0.716| 0.975| 0.831| 804| 1077| 0.747 |
| `⏎` | 0.979| 0.774| 0.150| 0.864| 0.259| 239| 1237| 0.193 |
| `⏎⏎` | 0.977| 0.929| 0.277| 0.952| 0.432| 226| 757| 0.299 |
| `"` | 0.000| 0.000| 0.000| 0.000| 0.000| 66| 66| 1.000 |
| `⏎␣⁺␣⁺` | 0.000| 0.000| 0.000| 0.000| 0.000| 26| 1081| 0.024 |
| `micro avg` | 0.980| 0.980| 0.760| 0.980| 0.856| 21335| 27482| 0.776 |
| `weighted avg` | 0.975| 0.980| 0.760| 0.977| 0.817| 21335| 27482| 0.776 |
| `macro avg` | 0.734| 0.701| 0.453| 0.716| 0.521| 21335| 27482| 0.776 |

### Confusion matrix

|refusal|  ∅| ␣| '| ⏎| ⏎␣⁺␣⁺| ⏎␣⁻␣⁻| ⏎⏎| "| 
|:---|:---|:---|:---|:---|:---|:---|:---|:---|
|0 |0 |0 |0 |0 |0 |0 |0 |0 |
|811 |13696 |68 |0 |0 |0 |0 |0 |0 |
|2460 |166 |3187 |0 |0 |0 |6 |0 |0 |
|19 |0 |0 |2851 |0 |0 |0 |0 |0 |
|998 |27 |21 |0 |185 |0 |1 |5 |0 |
|1055 |18 |8 |0 |0 |0 |0 |0 |0 |
|273 |27 |6 |0 |0 |0 |771 |0 |0 |
|531 |9 |3 |0 |4 |0 |0 |210 |0 |
|0 |0 |0 |66 |0 |0 |0 |0 |0 |

### Files with most errors

| filename | number of errors|
|:----:|:-----|
| examples/get/server.js | 69 |
| lib/adapters/http.js | 33 |
| test/specs/requests.spec.js | 27 |
| lib/adapters/xhr.js | 26 |
| lib/core/dispatchRequest.js | 23 |
| karma.conf.js | 22 |
| examples/server.js | 18 |
| lib/utils.js | 18 |
| lib/helpers/buildURL.js | 14 |
| lib/helpers/cookies.js | 13 |

<details>
    <summary>Machine-readable report</summary>
```json
{
  "cl_report": {"\"": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 66}, "\u0027": {"f1-score": 0.9885575589459085, "precision": 0.9773740143983545, "recall": 1.0, "support": 2851}, "macro avg": {"f1-score": 0.7158723915210329, "precision": 0.7342565343353529, "recall": 0.7007589001034125, "support": 21335}, "micro avg": {"f1-score": 0.9796109678931334, "precision": 0.9796109678931334, "recall": 0.9796109678931334, "support": 21335}, "weighted avg": {"f1-score": 0.9772691603481353, "precision": 0.9753452937521331, "recall": 0.9796109678931334, "support": 21335}, "\u2205": {"f1-score": 0.9886310318692028, "precision": 0.9822850175715413, "recall": 0.995059575704737, "support": 13764}, "\u23ce": {"f1-score": 0.8644859813084111, "precision": 0.9788359788359788, "recall": 0.7740585774058577, "support": 239}, "\u23ce\u23ce": {"f1-score": 0.9523809523809524, "precision": 0.9767441860465116, "recall": 0.9292035398230089, "support": 226}, "\u23ce\u2423\u207a\u2423\u207a": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 26}, "\u23ce\u2423\u207b\u2423\u207b": {"f1-score": 0.9747155499367889, "precision": 0.9910025706940874, "recall": 0.9589552238805971, "support": 804}, "\u2423": {"f1-score": 0.9582080577269995, "precision": 0.9678105071363499, "recall": 0.9487942840130992, "support": 3359}},
  "cl_report_full": {"\"": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 66}, "\u0027": {"f1-score": 0.9853119059961983, "precision": 0.9773740143983545, "recall": 0.9933797909407666, "support": 2870}, "macro avg": {"f1-score": 0.5210222304965777, "precision": 0.7342565343353529, "recall": 0.45295041185530627, "support": 27482}, "micro avg": {"f1-score": 0.8562590900710818, "precision": 0.9796109678931334, "recall": 0.76049778036533, "support": 27482}, "weighted avg": {"f1-score": 0.8165778433015116, "precision": 0.9377440548995618, "recall": 0.76049778036533, "support": 27482}, "\u2205": {"f1-score": 0.9605161652289783, "precision": 0.9822850175715413, "recall": 0.9396912521440823, "support": 14575}, "\u23ce": {"f1-score": 0.2594670406732118, "precision": 0.9788359788359788, "recall": 0.14955537590945836, "support": 1237}, "\u23ce\u23ce": {"f1-score": 0.4320987654320988, "precision": 0.9767441860465116, "recall": 0.2774108322324967, "support": 757}, "\u23ce\u2423\u207a\u2423\u207a": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 1081}, "\u23ce\u2423\u207b\u2423\u207b": {"f1-score": 0.831266846361186, "precision": 0.9910025706940874, "recall": 0.7158774373259053, "support": 1077}, "\u2423": {"f1-score": 0.6995171202809481, "precision": 0.9678105071363499, "recall": 0.5476886062897405, "support": 5819}},
  "ppcr": 0.7763263226839385
}
```
</details>
