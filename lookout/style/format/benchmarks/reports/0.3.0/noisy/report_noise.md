# Quality report on the artificial noisy dataset

### Rules filtering thresholds

* `Confidence: 0  `
* `Support: 80`

### Metrics table

| repository |  number of mistakes  | precision at max recall | empirical confidence threshold |    max recall     |        Number of rules (filtered / overall)          |
|:----------:|:--------------------:|:-----------------------:|:------------------------------:|:-----------------:|:----------------------------------------------------:|
|  axios  | 77 |  0.133 |  0.99  | 0.584 | `145 / 145` |
|  jquery  | 175 |  0.079 |  0.99  | 0.726 | `345 / 345` |

### Precision-recall curves

![Precision-Recall curve](./pr_curves.png)