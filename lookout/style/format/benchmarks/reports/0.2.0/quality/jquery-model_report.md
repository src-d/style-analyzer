# Model report for file:///tmp/top-repos-quality-repos-4md0ht25/jquery HEAD dae5f3ce3d2df27873d01f0d9682f6a91ad66b87

### Dump

```json
{'created_at': '2019-04-08 14:32:51',
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
 'size': '20.9 kB',
 'tags': [],
 'uuid': 'ef28bb21-d5b0-49b0-8419-ebd2bd75a190',
 'vendor': 'source{d}',
 'version': [1]}
style.format.analyzer.FormatAnalyzer/[1] file:///tmp/top-repos-quality-repos-4md0ht25/jquery dae5f3ce3d2df27873d01f0d9682f6a91ad66b87

# javascript
46 rules, avg.len. 10.0
## train
PPCR: 0.943534
### report
macro
{'f1-score': 0.6217047652585485,
 'precision': 0.676881079490666,
 'recall': 0.5860707613814946,
 'support': 172044}
micro
{'f1-score': 0.9709260421752575,
 'precision': 0.9709260421752575,
 'recall': 0.9709260421752575,
 'support': 172044}
weighted
{'f1-score': 0.9684318067166278,
 'precision': 0.967555430894947,
 'recall': 0.9709260421752575,
 'support': 172044}
### report_full
macro
{'f1-score': 0.5459747556438544,
 'precision': 0.676881079490666,
 'recall': 0.49612124793404927,
 'support': 182340}
micro
{'f1-score': 0.9427175041762607,
 'precision': 0.9709260421752575,
 'recall': 0.9161017878688165,
 'support': 182340}
weighted
{'f1-score': 0.9266393255637579,
 'precision': 0.9627942566701893,
 'recall': 0.9161017878688165,
 'support': 182340}
## test
PPCR: 0.953065
### report
macro
{'f1-score': 0.6163461085659603,
 'precision': 0.6705622388657188,
 'recall': 0.5783801424832475,
 'support': 45912}
micro
{'f1-score': 0.9766074228959749,
 'precision': 0.9766074228959749,
 'recall': 0.9766074228959749,
 'support': 45912}
weighted
{'f1-score': 0.974916140784098,
 'precision': 0.9745338813463374,
 'recall': 0.9766074228959749,
 'support': 45912}
### report_full
macro
{'f1-score': 0.5385524732488102,
 'precision': 0.6705622388657188,
 'recall': 0.48668501462943026,
 'support': 48173}
micro
{'f1-score': 0.9531381197853005,
 'precision': 0.9766074228959749,
 'recall': 0.9307703485354867,
 'support': 48173}
weighted
{'f1-score': 0.938648283111629,
 'precision': 0.9702611429040474,
 'recall': 0.9307703485354867,
 'support': 48173}
```

## javascript
### Summary
33 rules, avg.len. 9.7

| | |
|-|-|
|Min support|97|
|Max support|33760|
|Min confidence|0.9260355234146118|
|Max confidence|0.9999546408653259|

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
                     'min_samples_split': 238,
                     'n_estimators': 10,
                     'prune_attributes': True,
                     'prune_branches_algorithms': ['reduced-error'],
                     'prune_dataset_ratio': 0.2,
                     'top_down_greedy_budget': [False, 0.5]}}
```

### Rules

| rule number | description |
|----:|:-----|
| 1 | `  -1.internal_type = StringLiteral<br>⇒ y = "<br>Confidence: 1.000. Support: 10282.` |
| 2 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.label in {<space>}<br>⇒ y = "<br>Confidence: 1.000. Support: 9083.` |
| 3 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<space>}<br>	∧ +2.roles in {EXPRESSION}<br>	∧ ^1.roles in {OPERATOR}<br>⇒ y = ␣<br>Confidence: 0.962. Support: 3535.` |
| 4 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<space>}<br>	∧ +1.reserved = {<br>	∧ +2.roles in {EXPRESSION}<br>	∧ ^1.internal_type not in {ArrayExpression}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ␣<br>Confidence: 0.984. Support: 2409.` |
| 5 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<space>}<br>	∧ +1.reserved = =<br>	∧ +2.roles in {EXPRESSION}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ␣<br>Confidence: 0.999. Support: 961.` |
| 6 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<space>}<br>	∧ +1.internal_type = CommentLine<br>	∧ +1.reserved not in {=, {}<br>	∧ +1.length ≥ 2<br>	∧ +2.roles in {EXPRESSION}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.998. Support: 654.` |
| 7 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved = if<br>	∧ +1.reserved not in {{}<br>	∧ +1.length ≤ 1<br>	∧ +2.roles in {EXPRESSION}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ␣<br>Confidence: 0.999. Support: 440.` |
| 8 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved = (<br>	∧ +1.length ≤ 1<br>	∧ +2.roles in {EXPRESSION}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ␣<br>Confidence: 0.991. Support: 377.` |
| 9 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved not in {(, if}<br>	∧ +1.length ≤ 1<br>	∧ +2.roles in {EXPRESSION}<br>	∧ ^1.internal_type = ConditionalExpression<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ␣<br>Confidence: 0.977. Support: 285.` |
| 10 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved not in {(, if}<br>	∧ +1.reserved = }<br>	∧ +1.length ≤ 1<br>	∧ +2.roles in {EXPRESSION}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ⏎⇥⁻<br>Confidence: 0.971. Support: 154.` |
| 11 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved not in {(, ,, :, if}<br>	∧ +1.reserved not in {=, {, }}<br>	∧ +1.length ≤ 1<br>	∧ +2.roles in {EXPRESSION}<br>	∧ ^1.internal_type not in {File}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.988. Support: 33760.` |
| 12 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved = .<br>	∧ +2.roles not in {EXPRESSION}<br>⇒ y = ∅<br>Confidence: 1.000. Support: 11018.` |
| 13 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved = ;<br>	∧ +1.reserved not in {}}<br>	∧ +2.roles not in {EXPRESSION}<br>	∧ +4.length = 0<br>	∧ ^1.roles not in {STATEMENT}<br>⇒ y = ⏎<br>Confidence: 0.995. Support: 97.` |
| 14 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved not in {., ;}<br>	∧ +1.reserved = ;<br>	∧ +2.roles not in {EXPRESSION}<br>⇒ y = ∅<br>Confidence: 0.981. Support: 3859.` |
| 15 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved = if<br>	∧ +1.reserved = (<br>	∧ +2.roles not in {EXPRESSION}<br>⇒ y = ␣<br>Confidence: 0.996. Support: 135.` |
| 16 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved not in {;, if}<br>	∧ +1.reserved = (<br>	∧ +2.roles not in {EXPRESSION}<br>⇒ y = ∅<br>Confidence: 0.982. Support: 2805.` |
| 17 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved = {<br>	∧ +1.reserved not in {(}<br>	∧ +2.internal_type = CommentLine<br>	∧ +2.roles not in {EXPRESSION}<br>	∧ +3.roles not in {VALUE}<br>⇒ y = ⏎⏎⇥⁺<br>Confidence: 0.926. Support: 169.` |
| 18 | `  -1.diff_col ≥ 13<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.label in {<newline>}<br>	∧ -1.reserved not in {., ;, {}<br>	∧ +1.reserved not in {(, ;}<br>	∧ +2.roles not in {EXPRESSION}<br>⇒ y = "<br>Confidence: 0.999. Support: 978.` |
| 19 | `  -1.diff_col ≥ 13<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<newline>, <space>}<br>	∧ -1.reserved not in {., ;, {}<br>	∧ -1.roles not in {EXPRESSION}<br>	∧ +1.reserved not in {(, ;}<br>	∧ +2.roles not in {EXPRESSION}<br>⇒ y = ⏎<br>Confidence: 0.951. Support: 1321.` |
| 20 | `  -1.diff_col ≤ 12<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved not in {., ;, {}<br>	∧ +1.roles in {COMMENT}<br>	∧ +2.roles not in {EXPRESSION}<br>⇒ y = ∅<br>Confidence: 0.998. Support: 868.` |
| 21 | `  -1.diff_col ≤ 12<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {., ;, {}<br>	∧ -3.diff_line ≥ 1<br>	∧ +1.reserved not in {(, ;}<br>	∧ +1.roles in {KEY}<br>	∧ +2.roles not in {EXPRESSION}<br>⇒ y = "<br>Confidence: 0.964. Support: 98.` |
| 22 | `  -1.diff_col ≤ 12<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved not in {., ;, {}<br>	∧ +1.reserved = ,<br>	∧ +1.roles not in {KEY}<br>	∧ +2.roles not in {EXPRESSION}<br>⇒ y = ∅<br>Confidence: 0.985. Support: 688.` |
| 23 | `  -1.diff_col ≤ 12<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved not in {., ;, {}<br>	∧ -1.roles in {KEY}<br>	∧ +1.reserved not in {,}<br>	∧ +1.roles not in {KEY}<br>	∧ +2.roles not in {EXPRESSION}<br>⇒ y = ∅<br>Confidence: 0.999. Support: 492.` |
| 24 | `  •••start_col ≤ 28<br>	∧ -1.diff_col ≤ 12<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved not in {., {}<br>	∧ -1.roles not in {KEY}<br>	∧ +1.reserved = }<br>	∧ +1.roles not in {COMMENT, KEY}<br>	∧ +2.roles not in {EXPRESSION}<br>⇒ y = ⏎⇥⁻<br>Confidence: 0.927. Support: 636.` |
| 25 | `  -1.diff_col ≤ 12<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.label in {<newline>}<br>	∧ -1.reserved not in {., ;, {}<br>	∧ +1.reserved not in {(, ,, ;, }}<br>	∧ +2.roles not in {EXPRESSION}<br>⇒ y = "<br>Confidence: 0.998. Support: 229.` |
| 26 | `  -1.diff_col ≤ 12<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<newline>, <space>}<br>	∧ -1.reserved = (<br>	∧ -1.roles not in {KEY}<br>	∧ +1.reserved = )<br>	∧ +1.roles not in {KEY}<br>	∧ +2.roles not in {EXPRESSION}<br>⇒ y = ∅<br>Confidence: 0.998. Support: 1990.` |
| 27 | `  -1.diff_col ≤ 12<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<newline>, <space>}<br>	∧ -1.reserved = (<br>	∧ -1.roles not in {KEY}<br>	∧ +1.reserved not in {(, ), ,, }}<br>	∧ +1.roles not in {COMMENT, KEY}<br>	∧ +2.roles not in {COMMENT, EXPRESSION}<br>	∧ ^1.internal_type not in {VariableDeclaration}<br>⇒ y = ␣<br>Confidence: 0.986. Support: 10684.` |
| 28 | `  -1.diff_col ≤ 12<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<newline>, <space>}<br>	∧ -1.reserved not in {(, ., ;, {}<br>	∧ -1.roles not in {KEY}<br>	∧ -5.label not in {<newline>}<br>	∧ +1.reserved not in {(, ,, ;, }}<br>	∧ +1.roles not in {KEY}<br>	∧ +2.roles not in {EXPRESSION}<br>	∧ ^1.internal_type = ArrayExpression<br>⇒ y = ␣<br>Confidence: 0.930. Support: 494.` |
| 29 | `  -1.diff_col ≤ 12<br>	∧ -1.internal_type not in {Identifier, StringLiteral}<br>	∧ -1.label not in {<newline>, <space>}<br>	∧ -1.reserved = }<br>	∧ -1.roles not in {KEY}<br>	∧ -5.diff_offset ≥ 2<br>	∧ +1.reserved not in {,, }}<br>	∧ +1.roles not in {KEY}<br>	∧ +1.length ≥ 2<br>	∧ +2.roles not in {EXPRESSION}<br>	∧ ^1.internal_type not in {ArrayExpression, VariableDeclaration}<br>	∧ ^1.roles not in {SCOPE}<br>⇒ y = ␣<br>Confidence: 0.939. Support: 173.` |
| 30 | `  -1.diff_col ≤ 12<br>	∧ -1.internal_type not in {Identifier, StringLiteral}<br>	∧ -1.label not in {<newline>, <space>}<br>	∧ -1.reserved not in {(, ;, {, }}<br>	∧ -1.roles not in {KEY}<br>	∧ -3.reserved = (<br>	∧ -4.diff_offset ≤ 5<br>	∧ +1.reserved not in {,, }}<br>	∧ +1.roles not in {KEY}<br>	∧ +1.length ≥ 2<br>	∧ +2.roles not in {EXPRESSION}<br>	∧ ^1.internal_type not in {ArrayExpression}<br>⇒ y = ∅<br>Confidence: 0.981. Support: 187.` |
| 31 | `  -1.diff_col ≤ 12<br>	∧ -1.internal_type not in {Identifier, StringLiteral}<br>	∧ -1.label not in {<newline>, <space>}<br>	∧ -1.reserved not in {(, ., ;, {, }}<br>	∧ -1.roles not in {KEY}<br>	∧ -3.reserved not in {(}<br>	∧ -5.diff_offset ≥ 2<br>	∧ +1.reserved not in {(, ,, ;, }}<br>	∧ +1.roles not in {KEY}<br>	∧ +1.length ≥ 2<br>	∧ +2.roles not in {EXPRESSION}<br>	∧ ^1.internal_type not in {ArrayExpression, VariableDeclaration}<br>⇒ y = ␣<br>Confidence: 0.940. Support: 13730.` |
| 32 | `  -1.diff_col ≤ 12<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<newline>, <space>}<br>	∧ -1.reserved not in {(, ., ;, {}<br>	∧ -1.roles not in {KEY}<br>	∧ -4.label in {<newline>}<br>	∧ -5.diff_offset ≥ 2<br>	∧ +1.reserved not in {(, ,, ;, }}<br>	∧ +1.roles not in {KEY}<br>	∧ +1.length ≤ 1<br>	∧ +2.roles not in {EXPRESSION}<br>	∧ ^1.internal_type not in {ArrayExpression, VariableDeclaration}<br>⇒ y = ␣<br>Confidence: 0.947. Support: 459.` |
| 33 | `  -1.diff_col ≤ 12<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<newline>, <space>}<br>	∧ -1.reserved not in {(, ., ;, {}<br>	∧ -1.roles not in {KEY}<br>	∧ -4.label not in {<newline>}<br>	∧ -5.diff_offset ≥ 2<br>	∧ +1.reserved not in {(, ,, ;, }}<br>	∧ +1.roles not in {KEY}<br>	∧ +1.length ≤ 1<br>	∧ +2.roles not in {EXPRESSION}<br>	∧ ^1.internal_type not in {ArrayExpression, VariableDeclaration}<br>⇒ y = ␣<br>Confidence: 0.984. Support: 15301.` |

### Note
All statistics are calculated with respect to the "analyze" config. This means that the rules were filtered by
`confidence_threshold` and `support_threshold` values.

<details>
    <summary>Machine-readable report</summary>
```json
{"javascript": {"avg_rule_len": 9.666666666666666, "max_conf": 0.9999546408653259, "max_support": 33760, "min_conf": 0.9260355234146118, "min_support": 97, "num_rules": 33}}
```
</details>
