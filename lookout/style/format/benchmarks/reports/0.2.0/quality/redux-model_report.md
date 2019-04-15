# Model report for file:///tmp/top-repos-quality-repos-lui79obx/redux HEAD 902484ed735d38aec06683c847810a7218d8dba2

### Dump

```json
{'created_at': '2019-04-08 13:17:07',
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
 'size': '18.6 kB',
 'tags': [],
 'uuid': '23c422df-d5b6-4fc9-920b-a93fe392f545',
 'vendor': 'source{d}',
 'version': [1]}
style.format.analyzer.FormatAnalyzer/[1] file:///tmp/top-repos-quality-repos-lui79obx/redux 902484ed735d38aec06683c847810a7218d8dba2

# javascript
33 rules, avg.len. 7.7
## train
PPCR: 0.917716
### report
macro
{'f1-score': 0.6819400068687477,
 'precision': 0.7042645707535596,
 'recall': 0.6636866934977785,
 'support': 32946}
micro
{'f1-score': 0.9387786074181995,
 'precision': 0.9387786074181995,
 'recall': 0.9387786074181995,
 'support': 32946}
weighted
{'f1-score': 0.9354232842352583,
 'precision': 0.9337573134593934,
 'recall': 0.9387786074181995,
 'support': 32946}
### report_full
macro
{'f1-score': 0.6169089577169681,
 'precision': 0.7042645707535596,
 'recall': 0.5596220434460656,
 'support': 35900}
micro
{'f1-score': 0.8984980972024519,
 'precision': 0.9387786074181995,
 'recall': 0.8615320334261839,
 'support': 35900}
weighted
{'f1-score': 0.8898809975357319,
 'precision': 0.9304800503461712,
 'recall': 0.8615320334261839,
 'support': 35900}
## test
PPCR: 0.907526
### report
macro
{'f1-score': 0.6833575216322204,
 'precision': 0.7123341706704616,
 'recall': 0.6591114894416368,
 'support': 4956}
micro
{'f1-score': 0.9253430185633575,
 'precision': 0.9253430185633575,
 'recall': 0.9253430185633575,
 'support': 4956}
weighted
{'f1-score': 0.9215855537415905,
 'precision': 0.9200728226738047,
 'recall': 0.9253430185633575,
 'support': 4956}
### report_full
macro
{'f1-score': 0.6240759193440711,
 'precision': 0.7123341706704616,
 'recall': 0.5652973651298755,
 'support': 5461}
micro
{'f1-score': 0.8804838245176155,
 'precision': 0.9253430185633575,
 'recall': 0.8397729353598242,
 'support': 5461}
weighted
{'f1-score': 0.8731683753648134,
 'precision': 0.9190103507592141,
 'recall': 0.8397729353598242,
 'support': 5461}
```

## javascript
### Summary
17 rules, avg.len. 6.7

| | |
|-|-|
|Min support|122|
|Max support|4159|
|Min confidence|0.9232364892959595|
|Max confidence|0.9995958209037781|

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
| 1 | `  ^1.roles in {IDENTIFIER}<br>⇒ y = ∅<br>Confidence: 0.974. Support: 4159.` |
| 2 | `  -1.reserved = (<br>	∧ +1.internal_type = StringLiteral<br>	∧ +1.length ≥ 2<br>	∧ ^1.roles not in {IDENTIFIER}<br>⇒ y = '<br>Confidence: 0.976. Support: 358.` |
| 3 | `  -1.reserved = (<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.length ≥ 2<br>	∧ ^1.roles not in {IDENTIFIER}<br>⇒ y = ∅<br>Confidence: 0.938. Support: 996.` |
| 4 | `  -1.label in {<space>}<br>	∧ -1.reserved not in {(}<br>	∧ +1.length ≥ 2<br>	∧ ^1.roles not in {IDENTIFIER}<br>⇒ y = '<br>Confidence: 0.998. Support: 678.` |
| 5 | `  -1.label not in {<space>}<br>	∧ -1.reserved = {<br>	∧ -3.length ≥ 4<br>	∧ +1.length ≥ 2<br>	∧ +2.reserved = }<br>	∧ ^1.roles not in {IDENTIFIER, INCOMPLETE}<br>⇒ y = ␣<br>Confidence: 0.988. Support: 122.` |
| 6 | `  -1.internal_type = StringLiteral<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved not in {(, {}<br>	∧ +1.length ≥ 2<br>	∧ ^1.roles not in {IDENTIFIER}<br>⇒ y = '<br>Confidence: 0.948. Support: 357.` |
| 7 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved not in {(, {}<br>	∧ -1.length ≤ 2<br>	∧ +1.length ≥ 2<br>	∧ ^1.internal_type = JSXElement<br>	∧ ^1.roles not in {IDENTIFIER}<br>⇒ y = ∅<br>Confidence: 0.999. Support: 336.` |
| 8 | `  •••start_col ≥ 8<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved not in {(, {}<br>	∧ -1.length ≤ 2<br>	∧ +1.reserved = import<br>	∧ +1.length ≥ 2<br>	∧ ^1.internal_type not in {BlockStatement, JSXElement, ObjectExpression}<br>	∧ ^1.roles in {MODULE} and not in {IDENTIFIER}<br>⇒ y = ⏎<br>Confidence: 0.923. Support: 241.` |
| 9 | `  •••start_col ≤ 7<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved not in {(, {}<br>	∧ -1.length ≤ 2<br>	∧ -3.diff_col = 0<br>	∧ +1.length ≥ 2<br>	∧ ^1.internal_type not in {JSXElement}<br>⇒ y = ∅<br>Confidence: 0.997. Support: 158.` |
| 10 | `  -1.internal_type = StringLiteral<br>	∧ +1.length ≤ 1<br>	∧ ^1.roles not in {IDENTIFIER}<br>⇒ y = '<br>Confidence: 0.978. Support: 757.` |
| 11 | `  -1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = =<br>	∧ +1.length ≤ 1<br>	∧ ^1.internal_type = JSXAttribute<br>	∧ ^1.roles not in {IDENTIFIER}<br>⇒ y = ∅<br>Confidence: 0.972. Support: 160.` |
| 12 | `  -1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = =<br>	∧ +1.length ≤ 1<br>	∧ ^1.internal_type not in {JSXAttribute}<br>	∧ ^1.roles not in {IDENTIFIER}<br>⇒ y = ␣<br>Confidence: 1.000. Support: 1237.` |
| 13 | `  -1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = }<br>	∧ +1.length ≤ 1<br>	∧ ^1.roles in {DECLARATION} and not in {IDENTIFIER}<br>⇒ y = ␣<br>Confidence: 0.936. Support: 258.` |
| 14 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = (<br>	∧ +1.reserved = {<br>	∧ +1.length ≤ 1<br>	∧ ^1.roles not in {IDENTIFIER}<br>⇒ y = ∅<br>Confidence: 0.991. Support: 174.` |
| 15 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = :<br>	∧ +1.reserved not in {=, {, }}<br>	∧ +1.length ≤ 1<br>	∧ ^1.roles not in {IDENTIFIER}<br>⇒ y = ␣<br>Confidence: 0.998. Support: 237.` |
| 16 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, :, >}<br>	∧ -1.roles in {EXPRESSION}<br>	∧ +1.reserved not in {=, {, }}<br>	∧ +1.length ≤ 1<br>	∧ ^1.roles not in {IF, OPERATOR, VARIABLE}<br>⇒ y = ∅<br>Confidence: 0.990. Support: 4112.` |
| 17 | `  -1.diff_col ≤ 1<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, :, >}<br>	∧ -1.roles not in {EXPRESSION}<br>	∧ -3.reserved not in {{}<br>	∧ +1.reserved not in {=, {, }}<br>	∧ +1.length ≤ 1<br>	∧ ^1.roles not in {IF, OPERATOR, VARIABLE}<br>⇒ y = ∅<br>Confidence: 0.931. Support: 2694.` |

### Note
All statistics are calculated with respect to the "analyze" config. This means that the rules were filtered by
`confidence_threshold` and `support_threshold` values.

<details>
    <summary>Machine-readable report</summary>
```json
{"javascript": {"avg_rule_len": 6.705882352941177, "max_conf": 0.9995958209037781, "max_support": 4159, "min_conf": 0.9232364892959595, "min_support": 122, "num_rules": 17}}
```
</details>
