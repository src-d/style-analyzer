# Model report for file:///tmp/top-repos-quality-repos-_5k_hmxx/reveal.js HEAD 0b3e7839ebf4ed8b6c180aca0abafa28c67aee6d

### Dump

```json
{'created_at': datetime.datetime(2019, 1, 31, 11, 16, 36, 53749),
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
 'uuid': 'a7f236a9-18a4-4927-a879-06ffc21d0025',
 'version': [1]}
style.format.analyzer.FormatAnalyzer/[1] file:///tmp/top-repos-quality-repos-_5k_hmxx/reveal.js 0b3e7839ebf4ed8b6c180aca0abafa28c67aee6d

# javascript
32 rules, avg.len. 8.6
## train
PPCR: 0.795990
### report
macro
{'f1-score': 0.575074898624967,
 'precision': 0.5803477297515931,
 'recall': 0.5795926999967973,
 'support': 39821}
micro
{'f1-score': 0.9553753044875819,
 'precision': 0.9553753044875819,
 'recall': 0.9553753044875819,
 'support': 39821}
weighted
{'f1-score': 0.9522170029106551,
 'precision': 0.9509721715212338,
 'recall': 0.9553753044875819,
 'support': 39821}
### report_full
macro
{'f1-score': 0.4768604603389933,
 'precision': 0.5803477297515931,
 'recall': 0.4275075975017045,
 'support': 50027}
micro
{'f1-score': 0.8468524619357135,
 'precision': 0.9553753044875819,
 'recall': 0.7604693465528615,
 'support': 50027}
weighted
{'f1-score': 0.82339345983927,
 'precision': 0.9101731971739562,
 'recall': 0.7604693465528615,
 'support': 50027}
## test
PPCR: 0.756768
### report
macro
{'f1-score': 0.5285523971461058,
 'precision': 0.5396952201064971,
 'recall': 0.5384213824073144,
 'support': 7548}
micro
{'f1-score': 0.9523052464228935,
 'precision': 0.9523052464228935,
 'recall': 0.9523052464228935,
 'support': 7548}
weighted
{'f1-score': 0.9500710522829753,
 'precision': 0.9505441843457035,
 'recall': 0.9523052464228935,
 'support': 7548}
### report_full
macro
{'f1-score': 0.4055855853915359,
 'precision': 0.5396952201064971,
 'recall': 0.3579625512537189,
 'support': 9974}
micro
{'f1-score': 0.8204542860404063,
 'precision': 0.9523052464228935,
 'recall': 0.7206737517545618,
 'support': 9974}
weighted
{'f1-score': 0.7891910801257467,
 'precision': 0.9105702731244484,
 'recall': 0.7206737517545618,
 'support': 9974}
```

## javascript
### Summary
32 rules, avg.len. 8.6

| | |
|-|-|
|Min support|91|
|Max support|7156|
|Min confidence|0.8009259104728699|
|Max confidence|0.9996168613433838|

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
                     'min_samples_leaf': 91,
                     'min_samples_split': 239,
                     'n_estimators': 10,
                     'prune_attributes': True,
                     'prune_branches_algorithms': ['reduced-error'],
                     'prune_dataset_ratio': 0.2,
                     'top_down_greedy_budget': [False, 0.5]}}
```

### Rules

| rule number | description |
|----:|:-----|
| 1 | `  +1.reserved not in {)}<br>	∧ ^1.internal_type = MemberExpression<br>⇒ y = ∅<br>Confidence: 0.995. Support: 7156.` |
| 2 | `  -1.roles not in {STRING}<br>	∧ -4.diff_offset ≥ 6<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {)}<br>	∧ ^1.internal_type not in {MemberExpression}<br>	∧ ^1.roles in {OPERATOR}<br>⇒ y = ␣<br>Confidence: 0.960. Support: 3777.` |
| 3 | `  -1.roles not in {STRING}<br>	∧ -3.roles in {EXPRESSION}<br>	∧ -4.diff_offset ≤ 5<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {)}<br>	∧ ^1.internal_type not in {MemberExpression}<br>	∧ ^1.roles in {OPERATOR}<br>⇒ y = ␣<br>Confidence: 0.989. Support: 218.` |
| 4 | `  -1.roles not in {STRING}<br>	∧ -3.roles not in {EXPRESSION}<br>	∧ -4.diff_offset ≤ 5<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {)}<br>	∧ +2.roles in {EXPRESSION}<br>	∧ ^1.internal_type not in {MemberExpression}<br>	∧ ^1.roles in {OPERATOR}<br>⇒ y = ␣<br>Confidence: 0.890. Support: 187.` |
| 5 | `  -1.reserved = {<br>	∧ +1.length ≥ 2<br>	∧ +2.length ≤ 14<br>	∧ ^1.internal_type not in {MemberExpression}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ⏎⇥⁺<br>Confidence: 0.827. Support: 1174.` |
| 6 | `  -1.reserved not in {{}<br>	∧ +1.roles in {COMMENT}<br>	∧ +1.length ≥ 2<br>	∧ ^1.internal_type not in {MemberExpression}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.999. Support: 1043.` |
| 7 | `  -1.internal_type = CommentLine<br>	∧ -1.reserved not in {{}<br>	∧ +1.roles not in {COMMENT}<br>	∧ +1.length ≥ 2<br>	∧ ^1.internal_type not in {MemberExpression}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ⏎<br>Confidence: 0.987. Support: 883.` |
| 8 | `  -1.internal_type not in {CommentLine}<br>	∧ -1.reserved = ;<br>	∧ +1.roles not in {COMMENT}<br>	∧ +1.length ≥ 2<br>	∧ +2.roles in {COMMENT}<br>	∧ ^1.internal_type not in {MemberExpression}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ⏎⏎<br>Confidence: 0.905. Support: 300.` |
| 9 | `  -1.internal_type not in {CommentLine}<br>	∧ -1.reserved = (<br>	∧ -4.roles in {IDENTIFIER}<br>	∧ +1.roles in {STRING} and not in {COMMENT}<br>	∧ +1.length ≥ 2<br>	∧ ^1.internal_type not in {MemberExpression}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ␣'<br>Confidence: 0.880. Support: 420.` |
| 10 | `  -1.internal_type not in {CommentLine}<br>	∧ -1.reserved = }<br>	∧ +1.reserved not in {else}<br>	∧ +1.roles not in {COMMENT, STRING}<br>	∧ +1.length ≥ 2<br>	∧ ^1.internal_type not in {MemberExpression}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ⏎⏎<br>Confidence: 0.832. Support: 580.` |
| 11 | `  -1.internal_type not in {CommentLine}<br>	∧ -1.reserved not in {;, {, }}<br>	∧ +1.roles not in {COMMENT, STRING}<br>	∧ +1.length ≥ 2<br>	∧ ^1.internal_type = ObjectExpression<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ⏎<br>Confidence: 0.801. Support: 324.` |
| 12 | `  -1.internal_type not in {CommentLine}<br>	∧ -1.reserved not in {;, {, }}<br>	∧ -3.reserved = if<br>	∧ +1.roles not in {COMMENT, STRING}<br>	∧ +1.length ≥ 2<br>	∧ ^1.internal_type not in {ObjectExpression}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.986. Support: 246.` |
| 13 | `  -1.internal_type not in {CommentLine}<br>	∧ -1.reserved not in {;, {, }}<br>	∧ -2.roles in {COMMENT}<br>	∧ -3.diff_line ≥ 1<br>	∧ +1.roles not in {COMMENT, STRING}<br>	∧ +1.length ≥ 2<br>	∧ ^1.internal_type not in {MemberExpression}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ⏎<br>Confidence: 0.989. Support: 136.` |
| 14 | `  -1.internal_type not in {CommentLine}<br>	∧ -1.reserved not in {;, {, }}<br>	∧ -2.roles not in {COMMENT}<br>	∧ -3.diff_line ≥ 1<br>	∧ +1.roles not in {STRING}<br>	∧ +1.length ≥ 2<br>	∧ ^1.internal_type not in {ObjectExpression}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.935. Support: 238.` |
| 15 | `  -1.internal_type not in {CommentLine}<br>	∧ -1.reserved not in {;, {, }}<br>	∧ -3.diff_line = 0<br>	∧ -3.reserved not in {if}<br>	∧ +1.roles not in {COMMENT, STRING}<br>	∧ +1.length ≥ 2<br>	∧ +2.internal_type = CommentLine<br>	∧ ^1.internal_type not in {MemberExpression, ObjectExpression}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ⏎⏎<br>Confidence: 0.868. Support: 133.` |
| 16 | `  -1.internal_type not in {CommentLine}<br>	∧ -1.reserved = (<br>	∧ -2.reserved = if<br>	∧ -3.diff_line = 0<br>	∧ -3.reserved not in {if}<br>	∧ +1.roles not in {COMMENT, STRING}<br>	∧ +1.length ≥ 2<br>	∧ +2.internal_type not in {CommentLine}<br>	∧ ^1.internal_type not in {MemberExpression, ObjectExpression}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ␣<br>Confidence: 0.998. Support: 277.` |
| 17 | `  -1.internal_type not in {CommentLine}<br>	∧ -1.reserved = (<br>	∧ -2.reserved not in {if}<br>	∧ -2.length ≥ 7<br>	∧ -4.diff_offset ≤ 12<br>	∧ +1.roles not in {COMMENT, STRING}<br>	∧ +1.length ≥ 2<br>	∧ +2.internal_type not in {CommentLine}<br>	∧ ^1.internal_type not in {MemberExpression, ObjectExpression}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ␣<br>Confidence: 0.904. Support: 214.` |
| 18 | `  -1.internal_type not in {CommentLine}<br>	∧ -1.reserved not in {(, ;, {, }}<br>	∧ -3.diff_line = 0<br>	∧ -3.diff_offset ≥ 4<br>	∧ -3.reserved not in {if}<br>	∧ +1.roles not in {COMMENT, STRING}<br>	∧ +1.length ≥ 2<br>	∧ +2.internal_type not in {CommentLine}<br>	∧ ^1.internal_type not in {MemberExpression, ObjectExpression}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ␣<br>Confidence: 0.923. Support: 2770.` |
| 19 | `  +1.reserved = }<br>	∧ +1.length ≤ 1<br>	∧ +3.roles in {BLOCK}<br>	∧ ^1.internal_type not in {MemberExpression}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ⏎⏎⇥⁻<br>Confidence: 0.887. Support: 111.` |
| 20 | `  +1.reserved = }<br>	∧ +1.length ≤ 1<br>	∧ +3.roles not in {BLOCK}<br>	∧ ^1.internal_type not in {MemberExpression}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ⏎⇥⁻<br>Confidence: 0.850. Support: 1267.` |
| 21 | `  +1.reserved = {<br>	∧ +1.length ≤ 1<br>	∧ ^1.internal_type not in {MemberExpression}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ␣<br>Confidence: 0.975. Support: 1399.` |
| 22 | `  -1.roles in {STRING}<br>	∧ -4.reserved = .<br>	∧ +1.reserved not in {), {, }}<br>	∧ +1.length ≤ 1<br>	∧ ^1.internal_type not in {MemberExpression}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = '<br>Confidence: 0.932. Support: 139.` |
| 23 | `  -1.roles not in {STRING}<br>	∧ +1.reserved not in {{, }}<br>	∧ +1.length ≤ 1<br>	∧ ^1.internal_type = VariableDeclarator<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ␣<br>Confidence: 0.955. Support: 561.` |
| 24 | `  -1.roles not in {STRING}<br>	∧ -2.label in {<space>}<br>	∧ -3.reserved = (<br>	∧ +1.reserved = )<br>	∧ +1.roles not in {EXPRESSION}<br>	∧ +1.length ≤ 1<br>	∧ ^1.internal_type not in {MemberExpression}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ␣<br>Confidence: 0.998. Support: 260.` |
| 25 | `  -1.roles in {IDENTIFIER} and not in {STRING}<br>	∧ -2.label in {<space>}<br>	∧ -3.reserved not in {(}<br>	∧ -5.reserved = (<br>	∧ +1.reserved = )<br>	∧ +1.roles not in {EXPRESSION}<br>	∧ +1.length ≤ 1<br>	∧ ^1.internal_type not in {MemberExpression, VariableDeclarator}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.995. Support: 93.` |
| 26 | `  -1.roles not in {STRING}<br>	∧ -2.label not in {<space>}<br>	∧ -2.reserved = .<br>	∧ -4.label not in {<space>}<br>	∧ +1.reserved = )<br>	∧ +1.roles not in {EXPRESSION}<br>	∧ +1.length ≤ 1<br>	∧ ^1.internal_type not in {MemberExpression, VariableDeclarator}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.808. Support: 91.` |
| 27 | `  -1.roles not in {STRING}<br>	∧ -2.label not in {<space>}<br>	∧ -2.reserved not in {.}<br>	∧ +1.reserved = )<br>	∧ +1.roles not in {EXPRESSION}<br>	∧ +1.length ≤ 1<br>	∧ ^1.internal_type not in {MemberExpression, VariableDeclarator}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.894. Support: 1094.` |
| 28 | `  -1.roles not in {STRING}<br>	∧ +1.reserved not in {), {, }}<br>	∧ +1.roles not in {EXPRESSION}<br>	∧ +1.length ≤ 1<br>	∧ ^1.internal_type = ConditionalExpression<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ␣<br>Confidence: 0.861. Support: 184.` |
| 29 | `  •••start_col ≥ 12<br>	∧ -1.reserved not in {function}<br>	∧ -1.roles in {EXPRESSION} and not in {STRING}<br>	∧ +1.reserved not in {), {, }}<br>	∧ +1.roles not in {EXPRESSION}<br>	∧ +1.length ≤ 1<br>	∧ ^1.internal_type not in {ConditionalExpression, MemberExpression, VariableDeclarator}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.996. Support: 4422.` |
| 30 | `  •••start_col ≥ 12<br>	∧ -1.reserved not in {function}<br>	∧ -1.roles not in {EXPRESSION, STRING}<br>	∧ +1.reserved = ;<br>	∧ +1.roles not in {EXPRESSION}<br>	∧ +1.length ≤ 1<br>	∧ ^1.internal_type not in {ConditionalExpression, MemberExpression, VariableDeclarator}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 1.000. Support: 1305.` |
| 31 | `  •••start_col ≥ 12<br>	∧ -1.reserved not in {function}<br>	∧ -1.roles not in {EXPRESSION, STRING}<br>	∧ +1.reserved = ,<br>	∧ +1.roles not in {EXPRESSION}<br>	∧ +1.length ≤ 1<br>	∧ ^1.internal_type not in {ConditionalExpression, VariableDeclarator}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.991. Support: 166.` |
| 32 | `  •••start_col ≤ 11<br>	∧ -1.roles not in {STRING}<br>	∧ +1.reserved not in {(, ), {, }}<br>	∧ +1.roles not in {EXPRESSION}<br>	∧ +1.length ≤ 1<br>	∧ ^1.internal_type not in {ConditionalExpression, MemberExpression, VariableDeclarator}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.927. Support: 705.` |

<details>
    <summary>Machine-readable report</summary>
```json
{"javascript": {"avg_rule_len": 8.5625, "max_conf": 0.9996168613433838, "max_support": 7156, "min_conf": 0.8009259104728699, "min_support": 91, "num_rules": 32}}
```
</details>
