# Model report for file:///tmp/top-repos-quality-repos-xh6434di/telescope HEAD 534030114f47696fe3f3b08ea7ca49467428f2af

### Dump

```json
{'created_at': '2019-04-08 13:04:59',
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
 'size': '12.7 kB',
 'tags': [],
 'uuid': 'bf507b11-c896-4552-8520-8609635ef1ed',
 'vendor': 'source{d}',
 'version': [1]}
style.format.analyzer.FormatAnalyzer/[1] file:///tmp/top-repos-quality-repos-xh6434di/telescope 534030114f47696fe3f3b08ea7ca49467428f2af

# javascript
3 rules, avg.len. 2.7
## train
PPCR: 0.566543
### report
macro
{'f1-score': 0.47750424448217316,
 'precision': 0.45872274143302183,
 'recall': 0.5,
 'support': 613}
micro
{'f1-score': 0.9135399673735726,
 'precision': 0.9135399673735726,
 'recall': 0.9135399673735726,
 'support': 613}
weighted
{'f1-score': 0.8741999185724137,
 'precision': 0.8413552672368668,
 'recall': 0.9135399673735726,
 'support': 613}
### report_full
macro
{'f1-score': 0.43368527763432224,
 'precision': 0.45872274143302183,
 'recall': 0.4114648033126294,
 'support': 1082}
micro
{'f1-score': 0.6607669616519174,
 'precision': 0.9135399673735726,
 'recall': 0.5175600739371534,
 'support': 1082}
weighted
{'f1-score': 0.5453939522955382,
 'precision': 0.5767443467445195,
 'recall': 0.5175600739371534,
 'support': 1082}
## test
PPCR: 0.553055
### report
macro
{'f1-score': 0.4722222222222222,
 'precision': 0.45,
 'recall': 0.5,
 'support': 172}
micro
{'f1-score': 0.8255813953488372,
 'precision': 0.8255813953488372,
 'recall': 0.8255813953488372,
 'support': 172}
weighted
{'f1-score': 0.7480620155038761,
 'precision': 0.686046511627907,
 'recall': 0.8255813953488372,
 'support': 172}
### report_full
macro
{'f1-score': 0.40511974584555227,
 'precision': 0.45,
 'recall': 0.36860652436568664,
 'support': 311}
micro
{'f1-score': 0.587991718426501,
 'precision': 0.8255813953488372,
 'recall': 0.4565916398713826,
 'support': 311}
weighted
{'f1-score': 0.5088793756463085,
 'precision': 0.57491961414791,
 'recall': 0.4565916398713826,
 'support': 311}
```

## javascript
### Summary
2 rules, avg.len. 2.0

| | |
|-|-|
|Min support|104|
|Max support|129|
|Min confidence|0.995192289352417|
|Max confidence|0.9961240291595459|

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
                     'max_depth': 5,
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
| 1 | `  -1.internal_type = StringLiteral<br>⇒ y = '<br>Confidence: 0.996. Support: 129.` |
| 2 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.label in {<space>}<br>	∧ +1.roles in {LITERAL}<br>⇒ y = '<br>Confidence: 0.995. Support: 104.` |

### Note
All statistics are calculated with respect to the "analyze" config. This means that the rules were filtered by
`confidence_threshold` and `support_threshold` values.

<details>
    <summary>Machine-readable report</summary>
```json
{"javascript": {"avg_rule_len": 2.0, "max_conf": 0.9961240291595459, "max_support": 129, "min_conf": 0.995192289352417, "min_support": 104, "num_rules": 2}}
```
</details>
