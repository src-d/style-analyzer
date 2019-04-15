# Model report for file:///tmp/top-repos-quality-repos-384uwhei/create-react-app HEAD 32106d216e4c31fda30ec475f9f03186d116c893

### Dump

```json
{'created_at': '2019-04-08 13:12:11',
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
 'size': '16.3 kB',
 'tags': [],
 'uuid': '764974e2-f057-43d8-b576-fbe5ee8fa7fa',
 'vendor': 'source{d}',
 'version': [1]}
style.format.analyzer.FormatAnalyzer/[1] file:///tmp/top-repos-quality-repos-384uwhei/create-react-app 32106d216e4c31fda30ec475f9f03186d116c893

# javascript
19 rules, avg.len. 7.4
## train
PPCR: 0.868220
### report
macro
{'f1-score': 0.7765111499431455,
 'precision': 0.8181493170874958,
 'recall': 0.7473314252973217,
 'support': 13493}
micro
{'f1-score': 0.9443415104128067,
 'precision': 0.9443415104128067,
 'recall': 0.9443415104128067,
 'support': 13493}
weighted
{'f1-score': 0.940497385148231,
 'precision': 0.9418708923991407,
 'recall': 0.9443415104128067,
 'support': 13493}
### report_full
macro
{'f1-score': 0.7052029513126835,
 'precision': 0.8181493170874958,
 'recall': 0.6326631968105785,
 'support': 15541}
micro
{'f1-score': 0.8777295584487153,
 'precision': 0.9443415104128067,
 'recall': 0.8198957596036291,
 'support': 15541}
weighted
{'f1-score': 0.8614737640871383,
 'precision': 0.9296914551372688,
 'recall': 0.8198957596036291,
 'support': 15541}
## test
PPCR: 0.814548
### report
macro
{'f1-score': 0.7491943329491403,
 'precision': 0.7737019896254126,
 'recall': 0.7334576882254623,
 'support': 3505}
micro
{'f1-score': 0.9138373751783168,
 'precision': 0.9138373751783166,
 'recall': 0.9138373751783166,
 'support': 3505}
weighted
{'f1-score': 0.9130062028642693,
 'precision': 0.9148142156021857,
 'recall': 0.9138373751783166,
 'support': 3505}
### report_full
macro
{'f1-score': 0.6688617410983844,
 'precision': 0.7737019896254126,
 'recall': 0.60826848134264,
 'support': 4303}
micro
{'f1-score': 0.8204405737704918,
 'precision': 0.9138373751783166,
 'recall': 0.7443643969323728,
 'support': 4303}
weighted
{'f1-score': 0.8040504381295405,
 'precision': 0.889671485897653,
 'recall': 0.7443643969323728,
 'support': 4303}
```

## javascript
### Summary
14 rules, avg.len. 6.4

| | |
|-|-|
|Min support|109|
|Max support|2205|
|Min confidence|0.9322429895401001|
|Max confidence|0.9994054436683655|

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
| 1 | `  -1.internal_type = StringLiteral<br>⇒ y = '<br>Confidence: 0.999. Support: 841.` |
| 2 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = :<br>	∧ +1.internal_type = StringLiteral<br>⇒ y = ␣<br>Confidence: 0.973. Support: 169.` |
| 3 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, :}<br>	∧ -2.label not in {<space>}<br>	∧ +1.internal_type = StringLiteral<br>⇒ y = '<br>Confidence: 0.961. Support: 874.` |
| 4 | `  -1.internal_type = CommentLine<br>⇒ y = ⏎<br>Confidence: 0.968. Support: 460.` |
| 5 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved = ,<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = }<br>⇒ y = ⏎␣⁻␣⁻<br>Confidence: 0.997. Support: 144.` |
| 6 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved = ,<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {}}<br>	∧ +1.roles in {MAP}<br>	∧ ^1.internal_type not in {CallExpression}<br>⇒ y = ⏎<br>Confidence: 0.932. Support: 214.` |
| 7 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved = ;<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = }<br>⇒ y = ⏎␣⁻␣⁻<br>Confidence: 0.997. Support: 175.` |
| 8 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {,, ;, {}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ ^1.internal_type = VariableDeclarator<br>⇒ y = ␣<br>Confidence: 0.987. Support: 419.` |
| 9 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved = :<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ ^1.internal_type not in {VariableDeclarator}<br>⇒ y = ␣<br>Confidence: 0.999. Support: 370.` |
| 10 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved = const<br>	∧ +1.internal_type not in {StringLiteral}<br>⇒ y = ␣<br>Confidence: 0.997. Support: 163.` |
| 11 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {,, ;, const, {}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +2.reserved = ><br>⇒ y = ␣<br>Confidence: 0.996. Support: 130.` |
| 12 | `  -1.diff_offset ≥ 3<br>	∧ -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {,, ;, const, {}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.length ≥ 3<br>	∧ ^1.roles in {STATEMENT}<br>⇒ y = ␣<br>Confidence: 0.996. Support: 135.` |
| 13 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {,, :, ;, const, {}<br>	∧ -3.diff_line = 0<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {{}<br>	∧ +2.reserved not in {>}<br>	∧ ^1.internal_type = Program<br>	∧ ^1.roles not in {OPERATOR, STATEMENT}<br>⇒ y = ∅<br>Confidence: 0.940. Support: 109.` |
| 14 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {,, :, ;, const, {}<br>	∧ -1.roles in {EXPRESSION}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {{}<br>	∧ +2.reserved not in {>}<br>	∧ ^1.internal_type not in {Program, VariableDeclarator}<br>	∧ ^1.roles not in {OPERATOR, STATEMENT}<br>⇒ y = ∅<br>Confidence: 0.978. Support: 2205.` |

### Note
All statistics are calculated with respect to the "analyze" config. This means that the rules were filtered by
`confidence_threshold` and `support_threshold` values.

<details>
    <summary>Machine-readable report</summary>
```json
{"javascript": {"avg_rule_len": 6.428571428571429, "max_conf": 0.9994054436683655, "max_support": 2205, "min_conf": 0.9322429895401001, "min_support": 109, "num_rules": 14}}
```
</details>
