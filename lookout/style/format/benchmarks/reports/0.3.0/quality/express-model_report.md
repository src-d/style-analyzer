# Model report for file:///tmp/top-repos-quality-repos-3vdy94xf/express HEAD b4eb1f59d39d801d7365c86b04500f16faeb0b1c

### Dump

```json
{'created_at': '2019-06-11 09:33:07',
 'datasets': [],
 'dependencies': [],
 'description': 'Model bound to style.format.analyzer.FormatAnalyzer Lookout analyzer.',
 'environment': {'packages': 'ConfigArgParse==0.13.0 Jinja2==2.10.1 MarkupSafe==1.1.1 PyStemmer==1.3.0 PyYAML==5.1 Pympler==0.5 SQLAlchemy==1.2.10 SQLAlchemy-Utils==0.33.3 asdf==2.3.3 bblfsh==2.12.7 boto==2.49.0 boto3==1.9.165 botocore==1.12.165 cachetools==2.0.1 certifi==2019.3.9 chardet==3.0.4 clint==0.5.1 docker==4.0.1 dulwich==0.19.11 google-auth==1.6.3 google-auth-httplib2==0.0.3 google-cloud-core==0.25.0 grpcio==1.20.1 grpcio-tools==1.20.1 httplib2==0.12.3 humanfriendly==4.16.1 humanize==0.5.1 idna==2.8 jmespath==0.9.4 jsonschema==2.6.0 lookout-sdk==0.4.1 lookout-sdk-ml==0.19.1 lookout-style==0.2.0 lz4==2.1.6 modelforge==0.13.4 numpy==1.16.3 packaging==19.0 pandas==0.22.0 pip==19.1.1 prometheus-client==0.6.0 protobuf==3.7.1 psycopg2-binary==2.7.5 pygtrie==2.3 pyparsing==2.4.0 python-dateutil==2.8.0 python-igraph==0.7.1.post6 pytz==2019.1 requests==2.22.0 requirements-parser==0.2.0 scikit-learn==0.20.1 scikit-optimize==0.5.2 scipy==1.3.0 semantic-version==2.6.0 setuptools==41.0.1 six==1.12.0 '
                             'smart-open==1.8.1 sortedcontainers==2.1.0 sourced-ml==0.8.2 spdx==2.5.0 stringcase==1.2.0 tabulate==0.8.2 tqdm==4.32.1 urllib3==1.24.3 websocket-client==0.56.0 xxhash==1.3.0',
                 'platform': 'Linux-4.15.0-51-generic-x86_64-with-Ubuntu-18.04-bionic',
                 'python': '3.6.8 (default, Jan 14 2019, 11:02:34) [GCC 8.0.1 20180414 (experimental) [trunk revision 259383]]'},
 'license': 'ODbL-1.0',
 'metrics': {},
 'model': 'style.format.analyzer.FormatAnalyzer',
 'references': [],
 'series': 'Lookout',
 'size': '15.1 kB',
 'tags': [],
 'uuid': '4b9aa025-4433-44e1-82f5-23aedd63a80c',
 'vendor': 'source{d}',
 'version': [1]}
style.format.analyzer.FormatAnalyzer/[1] file:///tmp/top-repos-quality-repos-3vdy94xf/express b4eb1f59d39d801d7365c86b04500f16faeb0b1c

# javascript
42 rules, avg.len. 8.7
## train
PPCR: 0.936035
### report
macro
{'f1-score': 0.9407837000611073,
 'precision': 0.9567170931743278,
 'recall': 0.9268900324182361,
 'support': 67256}
micro
{'f1-score': 0.9693410253360295,
 'precision': 0.9693410253360295,
 'recall': 0.9693410253360295,
 'support': 67256}
weighted
{'f1-score': 0.9688079143206172,
 'precision': 0.9689240782470719,
 'recall': 0.9693410253360295,
 'support': 67256}
### report_full
macro
{'f1-score': 0.8992554446010051,
 'precision': 0.9567170931743278,
 'recall': 0.854828117609037,
 'support': 71852}
micro
{'f1-score': 0.9373148920263391,
 'precision': 0.9693410253360295,
 'recall': 0.907337304459166,
 'support': 71852}
weighted
{'f1-score': 0.9352653488861141,
 'precision': 0.9680478345031822,
 'recall': 0.907337304459166,
 'support': 71852}
## test
PPCR: 0.929035
### report
macro
{'f1-score': 0.9156585989827993,
 'precision': 0.9446982027508062,
 'recall': 0.8924085666440117,
 'support': 14453}
micro
{'f1-score': 0.9575866602089531,
 'precision': 0.9575866602089531,
 'recall': 0.9575866602089531,
 'support': 14453}
weighted
{'f1-score': 0.9560831645065708,
 'precision': 0.9565873113187495,
 'recall': 0.9575866602089531,
 'support': 14453}
### report_full
macro
{'f1-score': 0.8733555562286369,
 'precision': 0.9446982027508062,
 'recall': 0.8228878707973962,
 'support': 15557}
micro
{'f1-score': 0.9223592135954681,
 'precision': 0.9575866602089531,
 'recall': 0.8896316770585588,
 'support': 15557}
weighted
{'f1-score': 0.9186367685228695,
 'precision': 0.9552096563572057,
 'recall': 0.8896316770585588,
 'support': 15557}
```

## javascript
### Summary
27 rules, avg.len. 7.8

| | |
|-|-|
|Min support|110|
|Max support|22584|
|Min confidence|0.9223194718360901|
|Max confidence|0.9998694658279419|

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
| 1 | `  -1.roles in {STRING}<br>⇒ y = '<br>Confidence: 1.000. Support: 3831.` |
| 2 | `  -1.reserved = ,<br>	∧ -1.roles not in {STRING}<br>	∧ +1.roles in {STRING}<br>⇒ y = ␣<br>Confidence: 0.965. Support: 592.` |
| 3 | `  -1.reserved = :<br>	∧ -1.roles not in {STRING}<br>	∧ +1.roles in {STRING}<br>⇒ y = ␣<br>Confidence: 0.998. Support: 222.` |
| 4 | `  -1.reserved not in {,, :}<br>	∧ -2.label not in {<space>}<br>	∧ +1.roles in {STRING}<br>⇒ y = '<br>Confidence: 0.988. Support: 3916.` |
| 5 | `  -1.reserved = ,<br>	∧ -1.roles not in {STRING}<br>	∧ +1.roles not in {STRING}<br>⇒ y = ␣<br>Confidence: 0.981. Support: 3028.` |
| 6 | `  -1.reserved not in {,}<br>	∧ -1.roles not in {STRING}<br>	∧ -2.diff_offset ≥ 3<br>	∧ +1.reserved = =<br>	∧ +1.roles not in {STRING}<br>⇒ y = ␣<br>Confidence: 1.000. Support: 1282.` |
| 7 | `  -1.reserved = var<br>	∧ -1.roles not in {STRING}<br>	∧ -2.diff_offset ≥ 3<br>	∧ +1.roles not in {STRING}<br>⇒ y = ␣<br>Confidence: 0.999. Support: 883.` |
| 8 | `  -1.reserved not in {,, var}<br>	∧ -1.roles not in {STRING}<br>	∧ -2.diff_offset ≥ 3<br>	∧ -4.internal_type not in {Identifier}<br>	∧ -4.label not in {<space>}<br>	∧ +1.reserved = }<br>	∧ +1.roles not in {STRING}<br>⇒ y = ⏎␣⁻␣⁻<br>Confidence: 0.986. Support: 257.` |
| 9 | `  -1.reserved not in {,, var}<br>	∧ -1.roles not in {STRING}<br>	∧ -2.diff_offset ≥ 3<br>	∧ -4.roles in {CALLEE}<br>	∧ +1.reserved not in {=, }}<br>	∧ +1.roles not in {STRING}<br>	∧ +4.internal_type = StringLiteral<br>⇒ y = ⏎<br>Confidence: 0.954. Support: 441.` |
| 10 | `  -1.reserved not in {var}<br>	∧ -1.roles not in {STRING}<br>	∧ -2.diff_offset ≥ 3<br>	∧ -4.roles not in {CALLEE}<br>	∧ +1.reserved not in {}}<br>	∧ +1.roles not in {STRING}<br>	∧ ^1.roles in {OPERATOR}<br>⇒ y = ␣<br>Confidence: 0.929. Support: 404.` |
| 11 | `  -1.internal_type not in {CommentLine}<br>	∧ -1.reserved = if<br>	∧ -1.roles not in {STRING}<br>	∧ -2.diff_offset ≥ 3<br>	∧ -4.roles not in {CALLEE}<br>	∧ +1.reserved not in {}}<br>	∧ +1.roles not in {STRING}<br>⇒ y = ␣<br>Confidence: 0.993. Support: 201.` |
| 12 | `  -1.internal_type not in {CommentLine}<br>	∧ -1.reserved = :<br>	∧ -1.roles not in {STRING}<br>	∧ -2.diff_offset ≥ 3<br>	∧ -4.roles not in {CALLEE}<br>	∧ +1.reserved not in {}}<br>	∧ +1.roles not in {STRING}<br>⇒ y = ␣<br>Confidence: 0.965. Support: 186.` |
| 13 | `  -1.internal_type not in {CommentLine}<br>	∧ -1.reserved = return<br>	∧ -1.roles not in {STRING}<br>	∧ -2.diff_offset ≥ 3<br>	∧ -4.roles not in {CALLEE}<br>	∧ +1.reserved not in {}}<br>	∧ +1.roles not in {STRING}<br>⇒ y = ␣<br>Confidence: 0.997. Support: 174.` |
| 14 | `  -1.internal_type not in {CommentLine}<br>	∧ -1.roles not in {STRING}<br>	∧ -2.diff_offset ≥ 3<br>	∧ -4.label in {<space>}<br>	∧ -4.roles not in {CALLEE}<br>	∧ +1.reserved = {<br>	∧ +1.roles not in {STRING}<br>⇒ y = ␣<br>Confidence: 0.997. Support: 184.` |
| 15 | `  -1.internal_type not in {CommentLine}<br>	∧ -1.roles not in {STRING}<br>	∧ -2.diff_offset ≥ 3<br>	∧ -4.label not in {<space>}<br>	∧ -4.roles not in {CALLEE}<br>	∧ +1.reserved = {<br>	∧ +1.roles not in {STRING}<br>	∧ ^1.roles in {STATEMENT}<br>⇒ y = ␣<br>Confidence: 0.929. Support: 119.` |
| 16 | `  -1.internal_type not in {CommentLine}<br>	∧ -1.reserved not in {,, :, var}<br>	∧ -1.roles not in {STRING}<br>	∧ -2.diff_offset ≥ 3<br>	∧ -4.label not in {<space>}<br>	∧ -4.roles not in {CALLEE}<br>	∧ -4.length ≥ 3<br>	∧ +1.reserved = {<br>	∧ +1.roles not in {STRING}<br>	∧ ^1.roles not in {OPERATOR, STATEMENT}<br>⇒ y = ∅<br>Confidence: 0.922. Support: 457.` |
| 17 | `  -1.internal_type not in {CommentLine}<br>	∧ -1.reserved = new<br>	∧ -1.roles not in {STRING}<br>	∧ -2.diff_offset ≥ 3<br>	∧ -4.roles not in {CALLEE}<br>	∧ +1.reserved not in {{, }}<br>	∧ +1.roles not in {STRING}<br>⇒ y = ␣<br>Confidence: 0.995. Support: 110.` |
| 18 | `  -1.internal_type not in {CommentLine}<br>	∧ -1.reserved not in {,, ;, function, new, var}<br>	∧ -1.roles not in {STRING}<br>	∧ -2.diff_offset ≥ 3<br>	∧ -2.label in {<-space>}<br>	∧ -2.roles not in {LITERAL}<br>	∧ -4.roles not in {CALLEE}<br>	∧ +1.reserved not in {=, {, }}<br>	∧ +1.roles not in {STRING}<br>	∧ +1.length ≤ 1<br>	∧ ^1.roles not in {IF, OPERATOR}<br>	∧ ^2.internal_type not in {BlockStatement}<br>⇒ y = ∅<br>Confidence: 0.999. Support: 1354.` |
| 19 | `  -1.internal_type not in {CommentLine}<br>	∧ -1.reserved not in {,, :, ;, function, var}<br>	∧ -1.roles not in {STRING}<br>	∧ -2.diff_offset ≥ 3<br>	∧ -2.roles not in {LITERAL}<br>	∧ -4.roles not in {CALLEE}<br>	∧ +1.reserved not in {=, {, }}<br>	∧ +1.roles not in {STRING}<br>	∧ ^1.roles not in {IF, OPERATOR}<br>	∧ ^2.internal_type not in {BlockStatement}<br>⇒ y = ∅<br>Confidence: 0.993. Support: 22584.` |
| 20 | `  -1.reserved = {<br>	∧ -1.roles not in {STRING}<br>	∧ -2.diff_offset ≤ 2<br>	∧ +1.roles not in {STRING}<br>	∧ +3.roles not in {VALUE}<br>⇒ y = ⏎␣⁺␣⁺<br>Confidence: 0.975. Support: 1766.` |
| 21 | `  -1.reserved not in {,, {}<br>	∧ -1.roles not in {STRING}<br>	∧ -2.diff_offset ≤ 2<br>	∧ +1.reserved = }<br>	∧ +1.roles not in {STRING}<br>⇒ y = ⏎␣⁻␣⁻<br>Confidence: 0.960. Support: 1479.` |
| 22 | `  -1.reserved = =<br>	∧ -1.roles not in {STRING}<br>	∧ -2.diff_offset ≤ 2<br>	∧ +1.reserved not in {}}<br>	∧ +1.roles not in {STRING}<br>⇒ y = ␣<br>Confidence: 0.999. Support: 1291.` |
| 23 | `  •••start_col ≤ 11<br>	∧ -1.reserved = ;<br>	∧ -1.roles not in {STRING}<br>	∧ -2.diff_offset ≤ 2<br>	∧ -4.diff_col ≤ 8<br>	∧ +1.reserved not in {}}<br>	∧ +1.roles not in {STRING}<br>	∧ +3.length ≥ 2<br>⇒ y = ⏎⏎<br>Confidence: 0.969. Support: 114.` |
| 24 | `  -1.reserved not in {,, ;, =, {}<br>	∧ -1.roles not in {STRING}<br>	∧ -2.diff_offset ≤ 2<br>	∧ +1.reserved = .<br>	∧ +1.roles not in {STRING}<br>	∧ +4.roles in {CALL}<br>⇒ y = ⏎<br>Confidence: 0.968. Support: 663.` |
| 25 | `  -1.reserved not in {,, ;, =, {}<br>	∧ -1.roles not in {STRING}<br>	∧ -2.diff_offset ≤ 2<br>	∧ +1.reserved not in {., }}<br>	∧ +1.roles in {CALLEE} and not in {STRING}<br>⇒ y = ⏎⏎<br>Confidence: 0.948. Support: 472.` |
| 26 | `  -1.reserved not in {,, ;, =, {}<br>	∧ -1.roles not in {STRING}<br>	∧ -2.diff_offset ≤ 2<br>	∧ -3.reserved = function<br>	∧ -5.diff_col ≥ 1<br>	∧ +1.reserved = {<br>	∧ +1.roles not in {CALLEE, STRING}<br>	∧ +3.reserved not in {=}<br>	∧ ^1.internal_type not in {BinaryExpression}<br>⇒ y = ∅<br>Confidence: 0.963. Support: 362.` |
| 27 | `  -1.reserved not in {,, ;, =, {}<br>	∧ -1.roles not in {STRING}<br>	∧ -2.diff_offset ≤ 2<br>	∧ -5.diff_col ≥ 1<br>	∧ +1.reserved not in {., {, }}<br>	∧ +1.roles not in {CALLEE, STRING}<br>	∧ +2.reserved not in {.}<br>	∧ +3.reserved not in {=}<br>	∧ ^1.internal_type not in {BinaryExpression}<br>⇒ y = ∅<br>Confidence: 0.951. Support: 3648.` |

### Note
All statistics are calculated with respect to the "analyze" config. This means that the rules were filtered by
`confidence_threshold` and `support_threshold` values.

<details>
    <summary>Machine-readable report</summary>
```json
{"javascript": {"avg_rule_len": 7.814814814814815, "max_conf": 0.9998694658279419, "max_support": 22584, "min_conf": 0.9223194718360901, "min_support": 110, "num_rules": 27}}
```
</details>
