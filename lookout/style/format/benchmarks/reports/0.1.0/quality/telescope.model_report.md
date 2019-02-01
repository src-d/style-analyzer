# Model report for file:///tmp/top-repos-quality-repos-u22ofaub/telescope HEAD 534030114f47696fe3f3b08ea7ca49467428f2af

### Dump

```json
{'created_at': datetime.datetime(2019, 1, 31, 11, 13, 13, 764234),
 'dependencies': [],
 'environment': {'packages': [['ConfigArgParse', '0.13.0'],
                              ['Jinja2', '2.10'],
                              ['MarkupSafe', '1.1.0'],
                              ['PyYAML', '3.13'],
                              ['Pympler', '0.5'],
                              ['SQLAlchemy', '1.2.10'],
                              ['SQLAlchemy-Utils', '0.33.3'],
                              ['asdf', '2.3.1'],
                              ['bblfsh', '2.12.7'],
                              ['cachetools', '2.0.1'],
                              ['certifi', '2018.11.29'],
                              ['chardet', '3.0.4'],
                              ['clint', '0.5.1'],
                              ['dulwich', '0.19.10'],
                              ['grpcio', '1.18.0'],
                              ['grpcio-tools', '1.18.0'],
                              ['humanfriendly', '4.16.1'],
                              ['humanize', '0.5.1'],
                              ['idna', '2.8'],
                              ['jsonschema', '2.6.0'],
                              ['lookout-sdk', '0.4.1'],
                              ['lookout-sdk-ml', '0.8.1'],
                              ['lz4', '2.1.6'],
                              ['modelforge', '0.9.3'],
                              ['numpy', '1.16.0'],
                              ['pip', '19.0.1'],
                              ['protobuf', '3.6.1'],
                              ['psycopg2-binary', '2.7.5'],
                              ['pygtrie', '2.3'],
                              ['python-dateutil', '2.7.5'],
                              ['python-igraph', '0.7.1.post6'],
                              ['requests', '2.21.0'],
                              ['requirements-parser', '0.2.0'],
                              ['scikit-learn', '0.20.1'],
                              ['scikit-optimize', '0.5.2'],
                              ['scipy', '1.2.0'],
                              ['semantic-version', '2.6.0'],
                              ['setuptools', '40.7.1'],
                              ['six', '1.12.0'],
                              ['stringcase', '1.2.0'],
                              ['tabulate', '0.8.2'],
                              ['tqdm', '4.30.0'],
                              ['urllib3', '1.24.1'],
                              ['xxhash', '1.3.0']],
                 'platform': 'Linux-4.15.15-coreos-x86_64-with-Ubuntu-18.04-bionic',
                 'python': '3.6.7 (default, Oct 22 2018, 11:32:17) \n'
                           '[GCC 8.2.0]'},
 'model': 'code-format',
 'uuid': 'be0da947-362c-4280-8cb3-214407651940',
 'version': [1]}
style.format.analyzer.FormatAnalyzer/[1] file:///tmp/top-repos-quality-repos-u22ofaub/telescope 534030114f47696fe3f3b08ea7ca49467428f2af

# javascript
2 rules, avg.len. 2.0
## train
PPCR: 0.526483
### report
macro
{'f1-score': 0.3807504078303426,
 'precision': 0.36488095238095236,
 'recall': 0.4,
 'support': 497}
micro
{'f1-score': 0.8812877263581489,
 'precision': 0.8812877263581489,
 'recall': 0.8812877263581489,
 'support': 497}
weighted
{'f1-score': 0.8276444966700692,
 'precision': 0.7834207626712657,
 'recall': 0.8812877263581489,
 'support': 497}
### report_full
macro
{'f1-score': 0.338658193110421,
 'precision': 0.36488095238095236,
 'recall': 0.31765288087663485,
 'support': 944}
micro
{'f1-score': 0.607911172796669,
 'precision': 0.8812877263581489,
 'recall': 0.4639830508474576,
 'support': 944}
weighted
{'f1-score': 0.488364390051242,
 'precision': 0.5184530115012106,
 'recall': 0.4639830508474576,
 'support': 944}
## test
PPCR: 0.570470
### report
macro
{'f1-score': 0.37651245551601425,
 'precision': 0.3579617834394905,
 'recall': 0.4,
 'support': 170}
micro
{'f1-score': 0.8058823529411765,
 'precision': 0.8058823529411765,
 'recall': 0.8058823529411765,
 'support': 170}
weighted
{'f1-score': 0.7202218965878167,
 'precision': 0.6525665043087299,
 'recall': 0.8058823529411765,
 'support': 170}
### report_full
macro
{'f1-score': 0.3150476190476191,
 'precision': 0.3579617834394905,
 'recall': 0.2814385857970131,
 'support': 298}
micro
{'f1-score': 0.5854700854700854,
 'precision': 0.8058823529411765,
 'recall': 0.4597315436241611,
 'support': 298}
weighted
{'f1-score': 0.508347714924896,
 'precision': 0.56856752019835,
 'recall': 0.4597315436241611,
 'support': 298}
```

## javascript
### Summary
2 rules, avg.len. 2.0

| | |
|-|-|
|Min support|127|
|Max support|268|
|Min confidence|0.8152984976768494|
|Max confidence|0.9960629940032959|

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
                     'min_samples_leaf': 106,
                     'min_samples_split': 181,
                     'n_estimators': 10,
                     'prune_attributes': True,
                     'prune_branches_algorithms': ['reduced-error'],
                     'prune_dataset_ratio': 0.2,
                     'top_down_greedy_budget': [False, 0.5]}}
```

### Rules

| rule number | description |
|----:|:-----|
| 1 | `  -1.internal_type = StringLiteral<br>⇒ y = '<br>Confidence: 0.996. Support: 127.` |
| 2 | `  -1.internal_type not in {StringLiteral}<br>	∧ +1.roles not in {STRING}<br>	∧ +1.length ≤ 1<br>⇒ y = ∅<br>Confidence: 0.815. Support: 268.` |

<details>
    <summary>Machine-readable report</summary>
```json
{"javascript": {"avg_rule_len": 2.0, "max_conf": 0.9960629940032959, "max_support": 268, "min_conf": 0.8152984976768494, "min_support": 127, "num_rules": 2}}
```
</details>
