# Model report for file:///tmp/top-repos-quality-repos-pshr68w7/axios HEAD 21ae22dbd3ae3d3a55d9efd4eead3dd7fb6d8e6e

### Dump

```json
{'created_at': '2019-04-08 13:14:24',
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
 'size': '16.9 kB',
 'tags': [],
 'uuid': '682bab69-b250-4e37-b907-85c308d35595',
 'vendor': 'source{d}',
 'version': [1]}
style.format.analyzer.FormatAnalyzer/[1] file:///tmp/top-repos-quality-repos-pshr68w7/axios 21ae22dbd3ae3d3a55d9efd4eead3dd7fb6d8e6e

# javascript
18 rules, avg.len. 6.8
## train
PPCR: 0.895327
### report
macro
{'f1-score': 0.8203499908275484,
 'precision': 0.8263544845738591,
 'recall': 0.8156218391896219,
 'support': 22205}
micro
{'f1-score': 0.9614050889439315,
 'precision': 0.9614050889439315,
 'recall': 0.9614050889439315,
 'support': 22205}
weighted
{'f1-score': 0.9595398498850602,
 'precision': 0.9578863745765588,
 'recall': 0.9614050889439315,
 'support': 22205}
### report_full
macro
{'f1-score': 0.6782080201826638,
 'precision': 0.8263544845738591,
 'recall': 0.6203848193822092,
 'support': 24801}
micro
{'f1-score': 0.9083095775007447,
 'precision': 0.9614050889439315,
 'recall': 0.860771743074876,
 'support': 24801}
weighted
{'f1-score': 0.894776859044224,
 'precision': 0.9555991036422505,
 'recall': 0.860771743074876,
 'support': 24801}
## test
PPCR: 0.905204
### report
macro
{'f1-score': 0.836304545086677,
 'precision': 0.86027745527439,
 'recall': 0.817914651585508,
 'support': 5166}
micro
{'f1-score': 0.967673248161053,
 'precision': 0.967673248161053,
 'recall': 0.967673248161053,
 'support': 5166}
weighted
{'f1-score': 0.9674364273503792,
 'precision': 0.9681784952681634,
 'recall': 0.967673248161053,
 'support': 5166}
### report_full
macro
{'f1-score': 0.6957645196582848,
 'precision': 0.86027745527439,
 'recall': 0.6303892650926392,
 'support': 5707}
micro
{'f1-score': 0.9195254299641312,
 'precision': 0.967673248161053,
 'recall': 0.8759418258279306,
 'support': 5707}
weighted
{'f1-score': 0.9078209801581401,
 'precision': 0.9692892259147202,
 'recall': 0.8759418258279306,
 'support': 5707}
```

## javascript
### Summary
14 rules, avg.len. 7.4

| | |
|-|-|
|Min support|121|
|Max support|8053|
|Min confidence|0.9375|
|Max confidence|0.9983766078948975|

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
                     'min_samples_leaf': 120,
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
| 1 | `  -1.roles in {STRING}<br>⇒ y = '<br>Confidence: 0.968. Support: 1053.` |
| 2 | `  -1.reserved = :<br>	∧ -1.roles not in {STRING}<br>	∧ -2.label not in {<space>}<br>	∧ +1.roles in {STRING}<br>⇒ y = ␣<br>Confidence: 0.996. Support: 121.` |
| 3 | `  -1.reserved not in {,, :}<br>	∧ -1.roles not in {STRING}<br>	∧ -2.label not in {<space>}<br>	∧ +1.roles in {STRING}<br>⇒ y = '<br>Confidence: 0.947. Support: 1051.` |
| 4 | `  -1.reserved = ;<br>	∧ -1.roles not in {STRING}<br>	∧ +1.reserved = }<br>	∧ +1.roles not in {STRING}<br>⇒ y = ⏎␣⁻␣⁻<br>Confidence: 0.990. Support: 541.` |
| 5 | `  -1.reserved = ;<br>	∧ -1.roles not in {STRING}<br>	∧ +1.reserved not in {}}<br>	∧ +1.roles not in {STRING}<br>	∧ +3.internal_type = StringLiteral<br>⇒ y = ⏎⏎<br>Confidence: 0.966. Support: 162.` |
| 6 | `  -1.reserved not in {;, {}<br>	∧ -1.roles not in {STRING}<br>	∧ +1.reserved = {<br>	∧ +1.roles not in {STRING}<br>⇒ y = ␣<br>Confidence: 0.938. Support: 840.` |
| 7 | `  -1.reserved not in {;, {}<br>	∧ -1.roles not in {STRING}<br>	∧ +1.reserved not in {{}<br>	∧ +1.roles not in {STRING}<br>	∧ ^1.internal_type = VariableDeclarator<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ␣<br>Confidence: 0.981. Support: 496.` |
| 8 | `  -1.reserved = ,<br>	∧ -1.roles not in {STRING}<br>	∧ +1.roles not in {STRING}<br>	∧ ^1.internal_type not in {ObjectExpression}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ␣<br>Confidence: 0.969. Support: 465.` |
| 9 | `  -1.reserved = var<br>	∧ -1.roles not in {STRING}<br>	∧ +1.roles not in {STRING}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ␣<br>Confidence: 0.998. Support: 308.` |
| 10 | `  -1.reserved = :<br>	∧ -1.roles not in {STRING}<br>	∧ +1.roles not in {STRING}<br>⇒ y = ␣<br>Confidence: 0.993. Support: 214.` |
| 11 | `  -1.internal_type = CommentLine<br>	∧ -1.reserved not in {,, :, ;, function, var, {}<br>	∧ -1.roles not in {STRING}<br>	∧ +1.reserved not in {{, }}<br>	∧ +1.roles not in {STRING}<br>	∧ ^1.internal_type not in {VariableDeclarator}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ⏎<br>Confidence: 0.948. Support: 124.` |
| 12 | `  -1.internal_type not in {CommentLine}<br>	∧ -1.reserved not in {,, :, ;, function, var, {}<br>	∧ -1.roles not in {STRING}<br>	∧ -3.diff_line ≥ 1<br>	∧ +1.reserved not in {{, }}<br>	∧ +1.roles in {COMMENT} and not in {STRING}<br>	∧ +1.length ≥ 2<br>	∧ ^1.internal_type not in {VariableDeclarator}<br>	∧ ^1.roles not in {IF, OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.997. Support: 150.` |
| 13 | `  -1.internal_type not in {CommentLine}<br>	∧ -1.reserved not in {,, :, ;, function, var, {}<br>	∧ -1.roles not in {STRING}<br>	∧ -3.diff_line ≥ 1<br>	∧ +1.reserved not in {{, }}<br>	∧ +1.roles not in {STRING}<br>	∧ +1.length ≤ 1<br>	∧ ^1.internal_type not in {VariableDeclarator}<br>	∧ ^1.roles not in {IF, OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.945. Support: 1915.` |
| 14 | `  -1.internal_type not in {CommentLine}<br>	∧ -1.reserved not in {,, :, ;, function, var, {}<br>	∧ -1.roles not in {STRING}<br>	∧ -3.diff_line = 0<br>	∧ +1.reserved not in {{, }}<br>	∧ +1.roles not in {STRING}<br>	∧ ^1.internal_type not in {VariableDeclarator}<br>	∧ ^1.roles not in {IF, OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.981. Support: 8053.` |

### Note
All statistics are calculated with respect to the "analyze" config. This means that the rules were filtered by
`confidence_threshold` and `support_threshold` values.

<details>
    <summary>Machine-readable report</summary>
```json
{"javascript": {"avg_rule_len": 7.428571428571429, "max_conf": 0.9983766078948975, "max_support": 8053, "min_conf": 0.9375, "min_support": 121, "num_rules": 14}}
```
</details>
