# Model report for file:///tmp/top-repos-quality-repos-dhds81z3/reveal.js HEAD 0b3e7839ebf4ed8b6c180aca0abafa28c67aee6d

### Dump

```json
{'created_at': '2019-06-11 09:02:10',
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
 'size': '16.8 kB',
 'tags': [],
 'uuid': '183cd522-764a-4f91-8b47-d5867c187ac5',
 'vendor': 'source{d}',
 'version': [1]}
style.format.analyzer.FormatAnalyzer/[1] file:///tmp/top-repos-quality-repos-dhds81z3/reveal.js 0b3e7839ebf4ed8b6c180aca0abafa28c67aee6d

# javascript
37 rules, avg.len. 10.6
## train
PPCR: 0.837033
### report
macro
{'f1-score': 0.7817118994529552,
 'precision': 0.823048187019577,
 'recall': 0.7587351940264546,
 'support': 43745}
micro
{'f1-score': 0.9514687392844896,
 'precision': 0.9514687392844896,
 'recall': 0.9514687392844896,
 'support': 43745}
weighted
{'f1-score': 0.9505508877759605,
 'precision': 0.9523038909477964,
 'recall': 0.9514687392844896,
 'support': 43745}
### report_full
macro
{'f1-score': 0.6411099991710985,
 'precision': 0.823048187019577,
 'recall': 0.5723890346742362,
 'support': 52262}
micro
{'f1-score': 0.8670617767454457,
 'precision': 0.9514687392844896,
 'recall': 0.7964103937851594,
 'support': 52262}
weighted
{'f1-score': 0.8512868149213197,
 'precision': 0.9455412295477055,
 'recall': 0.7964103937851594,
 'support': 52262}
## test
PPCR: 0.816094
### report
macro
{'f1-score': 0.6366387228353018,
 'precision': 0.6340717179963151,
 'recall': 0.6792387139023746,
 'support': 8813}
micro
{'f1-score': 0.9489390672869624,
 'precision': 0.9489390672869624,
 'recall': 0.9489390672869624,
 'support': 8813}
weighted
{'f1-score': 0.9494587573255722,
 'precision': 0.9528072571144169,
 'recall': 0.9489390672869624,
 'support': 8813}
### report_full
macro
{'f1-score': 0.503298921361764,
 'precision': 0.6340717179963151,
 'recall': 0.44980407924458454,
 'support': 10799}
micro
{'f1-score': 0.8528451968182744,
 'precision': 0.9489390672869624,
 'recall': 0.7744235577368275,
 'support': 10799}
weighted
{'f1-score': 0.8395783334220888,
 'precision': 0.9370334732242648,
 'recall': 0.7744235577368275,
 'support': 10799}
```

## javascript
### Summary
19 rules, avg.len. 9.1

| | |
|-|-|
|Min support|91|
|Max support|7187|
|Min confidence|0.9318181872367859|
|Max confidence|0.9995073676109314|

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
| 1 | `  +1.reserved not in {)}<br>	∧ ^1.internal_type = MemberExpression<br>⇒ y = ∅<br>Confidence: 0.995. Support: 7187.` |
| 2 | `  -1.label not in {<space>}<br>	∧ -1.roles not in {STRING}<br>	∧ -2.reserved not in {[}<br>	∧ +1.reserved not in {)}<br>	∧ ^1.internal_type not in {MemberExpression}<br>	∧ ^1.roles in {OPERATOR}<br>⇒ y = ␣<br>Confidence: 0.944. Support: 4975.` |
| 3 | `  -1.internal_type = Identifier<br>	∧ +1.reserved = =<br>	∧ ^1.internal_type not in {MemberExpression}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ␣<br>Confidence: 0.999. Support: 477.` |
| 4 | `  -1.internal_type = Identifier<br>	∧ -2.reserved = (<br>	∧ -3.reserved not in {(}<br>	∧ +1.reserved = )<br>	∧ ^1.internal_type not in {MemberExpression}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.998. Support: 258.` |
| 5 | `  -1.internal_type = Identifier<br>	∧ -2.reserved not in {(}<br>	∧ -3.reserved not in {(}<br>	∧ -4.label not in {<space>}<br>	∧ -5.reserved = (<br>	∧ +1.reserved = )<br>	∧ ^1.internal_type not in {MemberExpression}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.995. Support: 101.` |
| 6 | `  -1.internal_type = Identifier<br>	∧ +1.reserved not in {), =}<br>	∧ ^1.internal_type not in {ConditionalExpression, MemberExpression}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.981. Support: 4181.` |
| 7 | `  -1.internal_type not in {Identifier, StringLiteral}<br>	∧ +1.reserved = ;<br>	∧ ^1.internal_type not in {MemberExpression}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.999. Support: 1839.` |
| 8 | `  -1.internal_type = CommentLine<br>	∧ -1.reserved not in {{}<br>	∧ +1.reserved not in {;, }}<br>	∧ ^1.internal_type not in {MemberExpression}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ⏎<br>Confidence: 0.986. Support: 898.` |
| 9 | `  -1.internal_type not in {CommentLine, Identifier}<br>	∧ -1.reserved not in {{}<br>	∧ +1.reserved not in {;, }}<br>	∧ +1.roles in {COMMENT}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 1.000. Support: 1015.` |
| 10 | `  -1.internal_type not in {CommentLine, Identifier}<br>	∧ -1.reserved = ;<br>	∧ +1.reserved not in {;, }}<br>	∧ +1.roles not in {COMMENT}<br>	∧ ^1.internal_type not in {MemberExpression}<br>	∧ ^1.roles in {FOR} and not in {FILE, OPERATOR}<br>⇒ y = ␣<br>Confidence: 0.984. Support: 91.` |
| 11 | `  -1.internal_type not in {CommentLine, Identifier}<br>	∧ -1.reserved not in {;, {}<br>	∧ -1.roles in {STRING}<br>	∧ -3.reserved = (<br>	∧ +1.reserved not in {;, }}<br>	∧ +1.roles not in {COMMENT}<br>	∧ ^1.internal_type not in {MemberExpression}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = '<br>Confidence: 0.999. Support: 341.` |
| 12 | `  -1.internal_type not in {CommentLine, Identifier}<br>	∧ -1.label in {<space>}<br>	∧ -1.reserved not in {;, {}<br>	∧ -1.roles not in {STRING}<br>	∧ -2.reserved = (<br>	∧ +1.reserved not in {;, }}<br>	∧ +1.roles not in {COMMENT}<br>	∧ ^1.internal_type not in {MemberExpression}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = '<br>Confidence: 0.999. Support: 403.` |
| 13 | `  -1.internal_type not in {CommentLine, Identifier}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved not in {{}<br>	∧ -1.roles not in {STRING}<br>	∧ +1.reserved = ,<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.993. Support: 507.` |
| 14 | `  -1.internal_type not in {CommentLine, Identifier}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved = (<br>	∧ -1.roles not in {STRING}<br>	∧ -2.diff_col ≥ 3<br>	∧ -2.internal_type not in {Identifier}<br>	∧ +1.reserved not in {,, }}<br>	∧ +1.roles not in {COMMENT}<br>	∧ +1.length ≥ 2<br>	∧ ^1.internal_type not in {MemberExpression}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ␣<br>Confidence: 0.996. Support: 386.` |
| 15 | `  -1.internal_type not in {CommentLine, Identifier}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved = (<br>	∧ -1.roles not in {STRING}<br>	∧ +1.reserved = )<br>	∧ +1.length ≤ 1<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.999. Support: 473.` |
| 16 | `  -1.internal_type not in {CommentLine, Identifier}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved = (<br>	∧ -1.roles not in {STRING}<br>	∧ -2.diff_col ≤ 2<br>	∧ +1.reserved not in {), ,, }}<br>	∧ +1.length ≤ 1<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.932. Support: 110.` |
| 17 | `  •••start_col ≥ 7<br>	∧ -1.internal_type not in {CommentLine, Identifier}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved not in {(, ;, {, }}<br>	∧ -1.roles not in {NUMBER, STRING}<br>	∧ -3.diff_col ≥ 4<br>	∧ +1.reserved not in {), ,, ;, }}<br>	∧ +1.roles not in {COMMENT, KEY}<br>	∧ +2.roles not in {ARGUMENT, COMMENT}<br>	∧ ^1.internal_type not in {MemberExpression}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ␣<br>Confidence: 0.933. Support: 4516.` |
| 18 | `  •••start_col ≤ 6<br>	∧ -1.internal_type not in {CommentLine, Identifier}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved not in {(, ;, {}<br>	∧ -1.roles not in {STRING}<br>	∧ -1.length ≥ 2<br>	∧ -2.internal_type = CommentBlock<br>	∧ +1.reserved not in {,, ;, }}<br>	∧ +1.roles not in {COMMENT}<br>	∧ ^1.internal_type not in {MemberExpression}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ⏎<br>Confidence: 0.980. Support: 124.` |
| 19 | `  •••start_col ≤ 6<br>	∧ -1.internal_type not in {CommentLine, Identifier}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved = var<br>	∧ -1.roles not in {STRING}<br>	∧ -1.length ≥ 2<br>	∧ -2.internal_type not in {CommentBlock}<br>	∧ +1.reserved not in {,, }}<br>	∧ +1.roles not in {COMMENT}<br>	∧ ^1.internal_type not in {MemberExpression}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ␣<br>Confidence: 0.996. Support: 128.` |

### Note
All statistics are calculated with respect to the "analyze" config. This means that the rules were filtered by
`confidence_threshold` and `support_threshold` values.

<details>
    <summary>Machine-readable report</summary>
```json
{"javascript": {"avg_rule_len": 9.052631578947368, "max_conf": 0.9995073676109314, "max_support": 7187, "min_conf": 0.9318181872367859, "min_support": 91, "num_rules": 19}}
```
</details>
