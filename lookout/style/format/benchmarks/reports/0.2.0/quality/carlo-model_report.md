# Model report for file:///tmp/top-repos-quality-repos-sbxt95f2/carlo HEAD b8ce2bca042c757b13fc82a3e059980342ddd9a8

### Dump

```json
{'created_at': '2019-04-08 13:06:01',
 'datasets': [],
 'dependencies': [],
 'description': 'Model bound to style.format.analyzer.FormatAnalyzer Lookout analyzer.',
 'environment': {'packages': 'ConfigArgParse==0.13.0 Jinja2==2.10 MarkupSafe==1.1.1 PyStemmer==1.3.0 PyYAML==5.1 Pympler==0.5 SQLAlchemy==1.2.10 SQLAlchemy-Utils==0.33.3 asdf==2.3.2 bblfsh==2.12.7 boto==2.49.0 boto3==1.9.130 botocore==1.12.130 cachetools==2.0.1 certifi==2019.3.9 chardet==3.0.4 clint==0.5.1 docker==3.7.1 docker-pycreds==0.4.0 dulwich==0.19.11 grpcio==1.19.0 grpcio-tools==1.19.0 humanfriendly==4.16.1 humanize==0.5.1 idna==2.8 jmespath==0.9.4 jsonschema==2.6.0 lookout-sdk==0.4.1 lookout-sdk-ml==0.18.1 lookout-style==0.1.1 lz4==2.1.6 modelforge==0.12.1 numpy==1.16.2 packaging==19.0 pandas==0.22.0 pip==19.0.3 protobuf==3.7.0 psycopg2-binary==2.7.5 pygtrie==2.3 pyparsing==2.3.1 python-dateutil==2.8.0 python-igraph==0.7.1.post6 pytz==2018.9 requests==2.21.0 requirements-parser==0.2.0 scikit-learn==0.20.1 scikit-optimize==0.5.2 scipy==1.2.1 semantic-version==2.6.0 setuptools==40.8.0 six==1.12.0 smart-open==1.8.0 sourced-ml==0.8.2 spdx==2.5.0 stringcase==1.2.0 tabulate==0.8.2 tqdm==4.31.1 '
                             'urllib3==1.24.1 websocket-client==0.56.0 xxhash==1.3.0',
                 'platform': 'Linux-4.15.15-coreos-x86_64-with-Ubuntu-18.04-bionic',
                 'python': '3.6.7 (default, Oct 22 2018, 11:32:17) [GCC 8.2.0]'},
 'license': 'ODbL-1.0',
 'metrics': {},
 'model': 'style.format.analyzer.FormatAnalyzer',
 'references': [],
 'series': 'Lookout',
 'size': '16.2 kB',
 'tags': [],
 'uuid': '9dcfaaaf-878e-4a47-bc8a-d1f095672a9e',
 'vendor': 'source{d}',
 'version': [1]}
style.format.analyzer.FormatAnalyzer/[1] file:///tmp/top-repos-quality-repos-sbxt95f2/carlo b8ce2bca042c757b13fc82a3e059980342ddd9a8

# javascript
15 rules, avg.len. 6.8
## train
PPCR: 0.797422
### report
macro
{'f1-score': 0.35524831887938835,
 'precision': 0.353189687472835,
 'recall': 0.3578386065188967,
 'support': 10207}
micro
{'f1-score': 0.954443029293622,
 'precision': 0.954443029293622,
 'recall': 0.954443029293622,
 'support': 10207}
weighted
{'f1-score': 0.9449464728244127,
 'precision': 0.9362399719160538,
 'recall': 0.954443029293622,
 'support': 10207}
### report_full
macro
{'f1-score': 0.32917928859137435,
 'precision': 0.353189687472835,
 'recall': 0.30895269199216824,
 'support': 12800}
micro
{'f1-score': 0.8468726909201547,
 'precision': 0.954443029293622,
 'recall': 0.76109375,
 'support': 12800}
weighted
{'f1-score': 0.7933684447079021,
 'precision': 0.8295458593973276,
 'recall': 0.76109375,
 'support': 12800}
## test
PPCR: 0.803715
### report
macro
{'f1-score': 0.3503921832495888,
 'precision': 0.3542607087975413,
 'recall': 0.3492101572274918,
 'support': 1601}
micro
{'f1-score': 0.9469081823860087,
 'precision': 0.9469081823860087,
 'recall': 0.9469081823860087,
 'support': 1601}
weighted
{'f1-score': 0.9361291412331992,
 'precision': 0.9279389946208706,
 'recall': 0.9469081823860087,
 'support': 1601}
### report_full
macro
{'f1-score': 0.33322046920726844,
 'precision': 0.3542607087975413,
 'recall': 0.3150966863338768,
 'support': 1992}
micro
{'f1-score': 0.8438630670748678,
 'precision': 0.9469081823860087,
 'recall': 0.7610441767068273,
 'support': 1992}
weighted
{'f1-score': 0.7894305503459097,
 'precision': 0.8202830289656512,
 'recall': 0.7610441767068273,
 'support': 1992}
```

## javascript
### Summary
9 rules, avg.len. 5.7

| | |
|-|-|
|Min support|94|
|Max support|2318|
|Min confidence|0.923701286315918|
|Max confidence|0.9990403056144714|

### Configuration

```json
{'feature_extractor': {'cutoff_label_support': 80,
                       'debug_parsing': False,
                       'label_composites': '<cut>',
                       'left_features': ['length',
                                         'diff_offset',
                                         'diff_col',
                                         'diff_line',
                                         'internal_type',
                                         'label',
                                         'reserved',
                                         'roles'],
                       'left_siblings_window': 5,
                       'no_labels_on_right': True,
                       'node_features': ['start_line', 'start_col'],
                       'parent_features': ['internal_type', 'roles'],
                       'parents_depth': 2,
                       'return_sibling_indices': False,
                       'right_features': ['length', 'internal_type', 'reserved', 'roles'],
                       'right_siblings_window': 5,
                       'select_features_number': 500,
                       'selected_features': '<cut>'},
 'line_length_limit': 500,
 'lines_ratio_train_trigger': 0.2,
 'lower_bound_instances': 500,
 'optimizer': {'base_model_name_categories': ['sklearn.ensemble.RandomForestClassifier',
                                              'sklearn.tree.DecisionTreeClassifier'],
               'cv': 3,
               'max_depth_categories': [None, 5, 10],
               'max_features_categories': [None, 'auto'],
               'min_samples_leaf_max': 120,
               'min_samples_leaf_min': 90,
               'min_samples_split_max': 240,
               'min_samples_split_min': 180,
               'n_iter': 50,
               'n_jobs': -1},
 'overall_size_limit': 5242880,
 'random_state': 42,
 'test_dataset_ratio': 0.2,
 'trainable_rules': {'attribute_similarity_threshold': 0.98,
                     'base_model_name': 'sklearn.tree.DecisionTreeClassifier',
                     'confidence_threshold': 0.8,
                     'min_samples_leaf': 90,
                     'min_samples_split': 180,
                     'n_estimators': 10,
                     'prune_attributes': True,
                     'prune_branches_algorithms': ['reduced-error'],
                     'prune_dataset_ratio': 0.2,
                     'top_down_greedy_budget': [False, 0.5]}}
```

### Rules

| rule number | description |
|----:|:-----|
| 1 | `  ^1.internal_type = MemberExpression<br>⇒ y = ∅<br>Confidence: 0.992. Support: 2318.` |
| 2 | `  -1.internal_type = StringLiteral<br>	∧ ^1.internal_type not in {MemberExpression}<br>	∧ ^2.roles not in {MAP}<br>⇒ y = '<br>Confidence: 0.992. Support: 317.` |
| 3 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = (<br>	∧ +1.roles in {STRING}<br>	∧ +1.length ≥ 2<br>	∧ ^1.internal_type not in {MemberExpression}<br>⇒ y = '<br>Confidence: 0.997. Support: 168.` |
| 4 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = (<br>	∧ +1.roles not in {STRING}<br>	∧ +1.length ≥ 2<br>	∧ ^1.internal_type not in {MemberExpression}<br>⇒ y = ∅<br>Confidence: 0.999. Support: 521.` |
| 5 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {(, ;, {}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.length ≥ 9<br>	∧ +2.length ≥ 2<br>	∧ ^1.internal_type not in {MemberExpression}<br>⇒ y = ∅<br>Confidence: 0.941. Support: 94.` |
| 6 | `  -1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = =<br>	∧ +1.length ≤ 1<br>	∧ ^1.internal_type not in {MemberExpression}<br>⇒ y = ␣<br>Confidence: 0.998. Support: 303.` |
| 7 | `  -1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = {<br>	∧ +1.length ≤ 1<br>	∧ ^1.internal_type not in {MemberExpression}<br>⇒ y = ␣<br>Confidence: 0.926. Support: 262.` |
| 8 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {;}<br>	∧ -1.roles in {EXPRESSION}<br>	∧ +1.reserved not in {=, {, }}<br>	∧ +1.roles not in {EXPRESSION}<br>	∧ +1.length ≤ 1<br>	∧ ^1.internal_type not in {MemberExpression}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.986. Support: 1618.` |
| 9 | `  -1.diff_col ≤ 1<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {;}<br>	∧ -1.roles not in {EXPRESSION}<br>	∧ +1.reserved not in {=, {, }}<br>	∧ +1.roles not in {EXPRESSION}<br>	∧ +1.length ≤ 1<br>	∧ ^1.internal_type not in {MemberExpression}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.924. Support: 924.` |

### Note
All statistics are calculated with respect to the "analyze" config. This means that the rules were filtered by
`confidence_threshold` and `support_threshold` values.

<details>
    <summary>Machine-readable report</summary>
```json
{"javascript": {"avg_rule_len": 5.666666666666667, "max_conf": 0.9990403056144714, "max_support": 2318, "min_conf": 0.923701286315918, "min_support": 94, "num_rules": 9}}
```
</details>
