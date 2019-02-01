# Model report for file:///tmp/top-repos-quality-repos-bz0i_ahe/react-native HEAD 1850906e5e557beb2234a1708cfc5fe8e7b4f0bf

### Dump

```json
{'created_at': datetime.datetime(2019, 1, 31, 16, 14, 28, 949514),
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
 'uuid': 'b02f195f-d090-4001-8514-18ba901cc5bf',
 'version': [1]}
style.format.analyzer.FormatAnalyzer/[1] file:///tmp/top-repos-quality-repos-bz0i_ahe/react-native 1850906e5e557beb2234a1708cfc5fe8e7b4f0bf

# javascript
168 rules, avg.len. 11.0
## train
PPCR: 0.940824
### report
macro
{'f1-score': 0.5687973426033368,
 'precision': 0.6113034973486,
 'recall': 0.5434607459213134,
 'support': 443862}
micro
{'f1-score': 0.9603210006713798,
 'precision': 0.9603210006713798,
 'recall': 0.9603210006713798,
 'support': 443862}
weighted
{'f1-score': 0.95778434673862,
 'precision': 0.9566707370667771,
 'recall': 0.9603210006713798,
 'support': 443862}
### report_full
macro
{'f1-score': 0.5367953476719303,
 'precision': 0.6113034973486,
 'recall': 0.4923072598551307,
 'support': 471780}
micro
{'f1-score': 0.9310407342607699,
 'precision': 0.9603210006713798,
 'recall': 0.903493153588537,
 'support': 471780}
weighted
{'f1-score': 0.9245726238837066,
 'precision': 0.952642664288292,
 'recall': 0.903493153588537,
 'support': 471780}
## test
PPCR: 0.930524
### report
macro
{'f1-score': 0.543632920283398,
 'precision': 0.6123094819055703,
 'recall': 0.5180167075455591,
 'support': 83321}
micro
{'f1-score': 0.9517408576469317,
 'precision': 0.9517408576469317,
 'recall': 0.9517408576469317,
 'support': 83321}
weighted
{'f1-score': 0.9484636867648648,
 'precision': 0.947419637047283,
 'recall': 0.9517408576469317,
 'support': 83321}
### report_full
macro
{'f1-score': 0.5031212552262099,
 'precision': 0.6123094819055703,
 'recall': 0.4541890529959519,
 'support': 89542}
micro
{'f1-score': 0.9174895726673724,
 'precision': 0.9517408576469317,
 'recall': 0.8856179223157847,
 'support': 89542}
weighted
{'f1-score': 0.9090747590549623,
 'precision': 0.9421825975756143,
 'recall': 0.8856179223157847,
 'support': 89542}
```

## javascript
### Summary
168 rules, avg.len. 11.0

| | |
|-|-|
|Min support|94|
|Max support|36628|
|Min confidence|0.8046594858169556|
|Max confidence|0.9998785257339478|

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
| 1 | `  -1.reserved = (<br>	∧ ^1.roles in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.928. Support: 1166.` |
| 2 | `  -1.reserved not in {(}<br>	∧ -3.diff_line ≥ 1<br>	∧ +1.roles in {LEFT, STRING}<br>	∧ ^1.roles in {OPERATOR}<br>⇒ y = "<br>Confidence: 0.984. Support: 95.` |
| 3 | `  -1.reserved not in {(}<br>	∧ -3.diff_line ≥ 1<br>	∧ +1.roles in {STRING} and not in {LEFT}<br>	∧ ^1.roles in {OPERATOR}<br>⇒ y = '<br>Confidence: 0.915. Support: 242.` |
| 4 | `  -1.reserved not in {(}<br>	∧ -3.diff_line = 0<br>	∧ -3.roles not in {LITERAL}<br>	∧ +1.roles in {RIGHT, STRING}<br>	∧ ^1.roles in {OPERATOR}<br>⇒ y = ␣'<br>Confidence: 0.858. Support: 822.` |
| 5 | `  -1.reserved not in {(}<br>	∧ -1.roles in {STRING}<br>	∧ +1.roles not in {STRING}<br>	∧ +1.length ≥ 3<br>	∧ ^1.roles in {OPERATOR}<br>⇒ y = "␣<br>Confidence: 0.999. Support: 423.` |
| 6 | `  -1.reserved not in {(}<br>	∧ -1.roles in {STRING}<br>	∧ +1.roles not in {STRING}<br>	∧ +1.length ≤ 2<br>	∧ +2.roles in {EXPRESSION}<br>	∧ ^1.roles in {OPERATOR}<br>⇒ y = '␣<br>Confidence: 0.902. Support: 513.` |
| 7 | `  -1.reserved not in {(}<br>	∧ -1.roles not in {STRING}<br>	∧ +1.reserved = )<br>	∧ +1.roles not in {STRING}<br>	∧ ^1.roles in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.963. Support: 470.` |
| 8 | `  -1.reserved not in {(}<br>	∧ -1.roles not in {STRING}<br>	∧ -2.diff_offset ≥ 6<br>	∧ +1.reserved = (<br>	∧ +1.roles not in {STRING}<br>	∧ ^1.roles in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.979. Support: 165.` |
| 9 | `  -1.reserved not in {(}<br>	∧ -1.roles not in {STRING}<br>	∧ -2.diff_offset ≤ 5<br>	∧ +1.reserved = (<br>	∧ +1.roles not in {STRING}<br>	∧ ^1.internal_type not in {LogicalExpression}<br>	∧ ^1.roles in {OPERATOR}<br>⇒ y = ␣<br>Confidence: 0.836. Support: 247.` |
| 10 | `  -1.reserved not in {(, {}<br>	∧ -1.roles not in {STRING}<br>	∧ -2.reserved = .<br>	∧ +1.reserved not in {(, )}<br>	∧ +1.roles not in {STRING}<br>	∧ +2.reserved = =<br>	∧ ^1.roles in {OPERATOR}<br>⇒ y = ␣<br>Confidence: 0.920. Support: 131.` |
| 11 | `  -1.reserved not in {(, {}<br>	∧ -1.roles not in {STRING}<br>	∧ -3.reserved = =<br>	∧ -4.diff_offset ≥ 5<br>	∧ +1.reserved not in {(, )}<br>	∧ +1.roles not in {STRING}<br>	∧ +2.reserved not in {=}<br>	∧ ^1.roles in {OPERATOR}<br>⇒ y = ␣<br>Confidence: 0.999. Support: 373.` |
| 12 | `  -1.reserved not in {(, {, ||}<br>	∧ -1.roles in {NUMBER} and not in {STRING}<br>	∧ -3.reserved not in {=}<br>	∧ -4.diff_offset ≥ 7<br>	∧ +1.reserved not in {(, )}<br>	∧ +1.roles not in {LEFT, STRING}<br>	∧ +2.reserved not in {=}<br>	∧ +2.roles in {EXPRESSION}<br>	∧ ^1.roles in {OPERATOR}<br>⇒ y = ␣<br>Confidence: 0.998. Support: 290.` |
| 13 | `  -1.reserved not in {(, {, ||}<br>	∧ -1.roles not in {NUMBER, STRING}<br>	∧ -3.reserved not in {=}<br>	∧ -4.diff_offset ≥ 7<br>	∧ +1.reserved not in {(, )}<br>	∧ +1.roles not in {LEFT, STRING}<br>	∧ +2.reserved not in {=}<br>	∧ ^1.roles in {OPERATOR}<br>⇒ y = ␣<br>Confidence: 0.982. Support: 20957.` |
| 14 | `  -1.reserved not in {(, {, ||}<br>	∧ -1.roles not in {STRING}<br>	∧ -4.diff_offset ≤ 6<br>	∧ +1.reserved not in {(, )}<br>	∧ +1.roles in {EXPRESSION} and not in {LEFT, STRING}<br>	∧ +2.reserved not in {=}<br>	∧ ^1.internal_type = LogicalExpression<br>	∧ ^1.roles in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.996. Support: 117.` |
| 15 | `  -1.reserved not in {(, {, ||}<br>	∧ -1.roles not in {STRING}<br>	∧ -3.reserved not in {=}<br>	∧ -4.diff_offset ≤ 6<br>	∧ +1.reserved not in {(, )}<br>	∧ +1.roles not in {EXPRESSION, LEFT, STRING}<br>	∧ +2.reserved not in {=}<br>	∧ ^1.internal_type = LogicalExpression<br>	∧ ^1.roles in {OPERATOR}<br>⇒ y = ␣<br>Confidence: 0.902. Support: 127.` |
| 16 | `  -1.reserved not in {(, {, ||}<br>	∧ -1.roles in {LITERAL} and not in {STRING}<br>	∧ -3.reserved not in {=}<br>	∧ -4.diff_offset ≤ 6<br>	∧ +1.reserved not in {(, )}<br>	∧ +1.roles not in {LEFT, STRING}<br>	∧ +2.reserved not in {=}<br>	∧ +2.roles in {EXPRESSION}<br>	∧ ^1.roles in {OPERATOR}<br>⇒ y = ␣<br>Confidence: 0.998. Support: 206.` |
| 17 | `  -1.reserved not in {(, {, ||}<br>	∧ -1.roles not in {LITERAL, STRING}<br>	∧ -3.reserved not in {=}<br>	∧ -4.diff_offset ≤ 6<br>	∧ +1.reserved not in {(, )}<br>	∧ +1.roles not in {LEFT, STRING}<br>	∧ +2.reserved not in {=}<br>	∧ ^1.roles in {OPERATOR}<br>⇒ y = ␣<br>Confidence: 0.936. Support: 2358.` |
| 18 | `  -1.reserved = ,<br>	∧ +1.reserved = }<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ⏎␣⁻␣⁻<br>Confidence: 0.983. Support: 2556.` |
| 19 | `  -1.reserved = ,<br>	∧ -3.internal_type = StringLiteral<br>	∧ -3.roles in {CALL}<br>	∧ +1.reserved = )<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ⏎␣⁻␣⁻<br>Confidence: 0.996. Support: 133.` |
| 20 | `  -1.reserved = ,<br>	∧ -3.internal_type not in {StringLiteral}<br>	∧ +1.reserved = )<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ⏎␣⁻␣⁻<br>Confidence: 0.959. Support: 966.` |
| 21 | `  -1.reserved = ,<br>	∧ -4.label in {<newline>}<br>	∧ +1.reserved not in {), }}<br>	∧ +2.reserved = )<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ⏎<br>Confidence: 0.906. Support: 421.` |
| 22 | `  -1.reserved = ,<br>	∧ -4.label not in {<newline>}<br>	∧ +1.reserved not in {), }}<br>	∧ +1.roles in {STRING}<br>	∧ +1.length ≥ 43<br>	∧ +2.reserved = )<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ⏎<br>Confidence: 0.884. Support: 125.` |
| 23 | `  -1.reserved = ,<br>	∧ -4.label not in {<newline>}<br>	∧ +1.reserved not in {), }}<br>	∧ +1.roles in {STRING}<br>	∧ +1.length ≤ 42<br>	∧ +2.reserved = )<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ␣'<br>Confidence: 0.924. Support: 270.` |
| 24 | `  •••start_col ≥ 42<br>	∧ -1.reserved = ,<br>	∧ -3.reserved = .<br>	∧ -3.roles not in {IDENTIFIER}<br>	∧ -4.label not in {<newline>}<br>	∧ +1.reserved not in {), }}<br>	∧ +1.roles not in {STRING}<br>	∧ +2.reserved = )<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ␣<br>Confidence: 0.824. Support: 105.` |
| 25 | `  -1.reserved = ,<br>	∧ -3.reserved not in {.}<br>	∧ -3.roles not in {IDENTIFIER}<br>	∧ -4.label not in {<newline>}<br>	∧ +1.reserved not in {), }}<br>	∧ +1.roles not in {STRING}<br>	∧ +2.reserved = )<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ␣<br>Confidence: 0.962. Support: 2517.` |
| 26 | `  -1.reserved = ,<br>	∧ -4.reserved = ,<br>	∧ +1.reserved not in {), }}<br>	∧ +2.reserved not in {)}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ␣<br>Confidence: 0.991. Support: 811.` |
| 27 | `  -1.reserved = ,<br>	∧ -2.roles in {ARGUMENT}<br>	∧ -4.reserved not in {,}<br>	∧ -4.roles in {IDENTIFIER}<br>	∧ +1.reserved not in {), }}<br>	∧ +2.reserved not in {)}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ␣<br>Confidence: 0.952. Support: 845.` |
| 28 | `  -1.reserved = ,<br>	∧ -2.roles not in {ARGUMENT}<br>	∧ -4.reserved not in {,}<br>	∧ -4.roles in {IDENTIFIER}<br>	∧ +1.reserved not in {}}<br>	∧ +2.reserved = :<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ⏎<br>Confidence: 0.908. Support: 420.` |
| 29 | `  -1.reserved = ,<br>	∧ -2.roles not in {ARGUMENT}<br>	∧ -4.reserved not in {,}<br>	∧ -4.roles in {IDENTIFIER}<br>	∧ +1.reserved not in {}}<br>	∧ +2.reserved = ><br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ␣<br>Confidence: 0.968. Support: 110.` |
| 30 | `  -1.reserved = ,<br>	∧ -2.internal_type = StringLiteral<br>	∧ -4.reserved not in {,}<br>	∧ -4.roles not in {IDENTIFIER}<br>	∧ +1.reserved not in {), }}<br>	∧ +2.reserved not in {)}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ␣'<br>Confidence: 0.988. Support: 208.` |
| 31 | `  •••start_col ≥ 31<br>	∧ •••start_line ≥ 17<br>	∧ -1.reserved = ,<br>	∧ -2.internal_type not in {StringLiteral}<br>	∧ -3.length ≥ 2<br>	∧ -4.reserved not in {,}<br>	∧ -4.roles not in {IDENTIFIER}<br>	∧ +1.reserved not in {), }}<br>	∧ +1.length ≥ 3<br>	∧ +2.reserved not in {), =}<br>	∧ +4.reserved = ,<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ⏎<br>Confidence: 0.954. Support: 701.` |
| 32 | `  •••start_col ≥ 31<br>	∧ •••start_line ≥ 17<br>	∧ -1.reserved = ,<br>	∧ -2.internal_type not in {StringLiteral}<br>	∧ -3.length ≤ 1<br>	∧ -4.reserved not in {,}<br>	∧ -4.roles not in {IDENTIFIER}<br>	∧ -5.diff_offset ≥ 19<br>	∧ +1.reserved not in {), }}<br>	∧ +1.length ≥ 3<br>	∧ +2.reserved not in {), =}<br>	∧ +4.reserved = ,<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ⏎<br>Confidence: 0.959. Support: 256.` |
| 33 | `  •••start_col ≤ 30<br>	∧ •••start_line ≥ 17<br>	∧ -1.reserved = ,<br>	∧ -2.internal_type not in {StringLiteral}<br>	∧ -4.reserved not in {,}<br>	∧ -4.roles not in {IDENTIFIER}<br>	∧ +1.reserved not in {), }}<br>	∧ +1.length ≥ 3<br>	∧ +2.reserved not in {), =}<br>	∧ +4.reserved = ,<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ⏎<br>Confidence: 0.976. Support: 3247.` |
| 34 | `  •••start_line ≤ 245<br>	∧ -1.reserved = ,<br>	∧ -2.internal_type not in {StringLiteral}<br>	∧ -4.reserved not in {,}<br>	∧ -4.roles not in {IDENTIFIER}<br>	∧ +1.reserved not in {), }}<br>	∧ +1.length ≥ 3<br>	∧ +2.reserved not in {)}<br>	∧ +3.roles in {VALUE}<br>	∧ +4.reserved not in {,}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ␣<br>Confidence: 0.998. Support: 206.` |
| 35 | `  -1.reserved = ,<br>	∧ -2.internal_type not in {StringLiteral}<br>	∧ -4.reserved not in {,}<br>	∧ -4.roles not in {IDENTIFIER}<br>	∧ +1.reserved not in {}}<br>	∧ +1.length ≥ 3<br>	∧ +2.reserved = }<br>	∧ +3.roles not in {VALUE}<br>	∧ +4.reserved not in {,}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ␣<br>Confidence: 0.995. Support: 110.` |
| 36 | `  -1.reserved = ,<br>	∧ -2.internal_type not in {StringLiteral}<br>	∧ -3.roles in {ARGUMENT}<br>	∧ -4.reserved not in {,}<br>	∧ -4.roles not in {IDENTIFIER}<br>	∧ +1.reserved not in {), }}<br>	∧ +1.length ≥ 3<br>	∧ +2.reserved not in {), }}<br>	∧ +3.reserved = ,<br>	∧ +4.reserved not in {,}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ⏎<br>Confidence: 0.919. Support: 267.` |
| 37 | `  -1.reserved = ,<br>	∧ -2.internal_type not in {StringLiteral}<br>	∧ -3.roles not in {ARGUMENT}<br>	∧ -4.reserved not in {,}<br>	∧ -4.roles not in {IDENTIFIER}<br>	∧ -5.reserved = ,<br>	∧ +1.reserved not in {), }}<br>	∧ +1.length ≥ 3<br>	∧ +2.reserved not in {), }}<br>	∧ +4.reserved not in {,}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ⏎<br>Confidence: 0.977. Support: 1026.` |
| 38 | `  •••start_col ≥ 6<br>	∧ -1.reserved = ,<br>	∧ -2.internal_type not in {StringLiteral}<br>	∧ -3.roles not in {ARGUMENT}<br>	∧ -4.reserved not in {,}<br>	∧ -4.roles not in {IDENTIFIER}<br>	∧ -5.reserved not in {,}<br>	∧ +1.reserved not in {), }}<br>	∧ +1.length ≥ 6<br>	∧ +2.reserved not in {), }}<br>	∧ +4.reserved not in {,}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ⏎<br>Confidence: 0.835. Support: 2288.` |
| 39 | `  •••start_col ≤ 5<br>	∧ -1.reserved = ,<br>	∧ -2.internal_type not in {StringLiteral}<br>	∧ -3.roles not in {ARGUMENT}<br>	∧ -4.reserved not in {,}<br>	∧ -4.roles not in {IDENTIFIER}<br>	∧ -5.reserved not in {,}<br>	∧ +1.reserved not in {), }}<br>	∧ +1.length ≥ 3<br>	∧ +2.reserved not in {), }}<br>	∧ +4.reserved not in {,}<br>	∧ ^1.roles not in {OPERATOR}<br>	∧ ^2.roles in {EXPRESSION}<br>⇒ y = ⏎⏎<br>Confidence: 0.883. Support: 98.` |
| 40 | `  -1.reserved = ,<br>	∧ -2.internal_type not in {StringLiteral}<br>	∧ -4.label in {<-space>}<br>	∧ -4.reserved not in {,}<br>	∧ -4.roles not in {IDENTIFIER}<br>	∧ +1.reserved not in {), }}<br>	∧ +1.length ≤ 2<br>	∧ +2.reserved not in {)}<br>	∧ +2.roles in {BLOCK}<br>	∧ +2.length ≥ 3<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ⏎⏎<br>Confidence: 0.934. Support: 99.` |
| 41 | `  -1.reserved = ,<br>	∧ -2.internal_type not in {StringLiteral}<br>	∧ -3.length ≥ 2<br>	∧ -4.reserved not in {,}<br>	∧ -4.roles not in {IDENTIFIER}<br>	∧ +1.reserved not in {), }}<br>	∧ +1.roles in {EXPRESSION}<br>	∧ +1.length ≤ 2<br>	∧ +2.reserved not in {)}<br>	∧ +2.length ≤ 2<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ⏎<br>Confidence: 0.819. Support: 152.` |
| 42 | `  •••start_col ≥ 20<br>	∧ -1.reserved = ,<br>	∧ -2.internal_type not in {StringLiteral}<br>	∧ -3.length ≤ 1<br>	∧ -4.reserved not in {,}<br>	∧ -4.roles not in {IDENTIFIER}<br>	∧ +1.reserved not in {), }}<br>	∧ +1.roles in {EXPRESSION}<br>	∧ +1.length ≤ 2<br>	∧ +2.reserved not in {)}<br>	∧ +2.length ≤ 2<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ␣<br>Confidence: 0.907. Support: 221.` |
| 43 | `  -1.reserved = ,<br>	∧ -2.internal_type not in {StringLiteral}<br>	∧ -4.reserved not in {,}<br>	∧ -4.roles not in {IDENTIFIER}<br>	∧ +1.reserved not in {)}<br>	∧ +1.roles not in {EXPRESSION}<br>	∧ +1.length ≤ 2<br>	∧ +2.reserved not in {)}<br>	∧ +2.length ≤ 2<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ⏎␣⁻␣⁻<br>Confidence: 0.823. Support: 507.` |
| 44 | `  •••start_line ≤ 238<br>	∧ -1.reserved = ;<br>	∧ -3.reserved = )<br>	∧ +1.reserved = }<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ⏎␣⁻␣⁻<br>Confidence: 0.934. Support: 114.` |
| 45 | `  -1.reserved = ;<br>	∧ -3.reserved not in {)}<br>	∧ +1.reserved = }<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ⏎␣⁻␣⁻<br>Confidence: 0.954. Support: 5664.` |
| 46 | `  •••start_col ≥ 15<br>	∧ -1.reserved = ;<br>	∧ +1.reserved not in {}}<br>	∧ +1.length ≤ 3<br>	∧ ^1.roles not in {OPERATOR}<br>	∧ ^2.roles in {SCOPE}<br>⇒ y = ␣<br>Confidence: 0.911. Support: 310.` |
| 47 | `  •••start_col ≥ 15<br>	∧ -1.reserved = ;<br>	∧ -3.roles not in {ASSIGNMENT}<br>	∧ +1.reserved not in {}}<br>	∧ ^1.roles in {BLOCK} and not in {OPERATOR}<br>	∧ ^2.roles not in {SCOPE}<br>⇒ y = ⏎<br>Confidence: 0.824. Support: 6337.` |
| 48 | `  •••start_col ≥ 15<br>	∧ -1.reserved = ;<br>	∧ +1.reserved not in {}}<br>	∧ +4.roles in {CALLEE}<br>	∧ ^1.roles not in {BLOCK, OPERATOR}<br>	∧ ^2.roles not in {SCOPE}<br>⇒ y = ⏎<br>Confidence: 0.851. Support: 1067.` |
| 49 | `  •••start_col ≥ 15<br>	∧ -1.reserved = ;<br>	∧ -5.reserved = '<br>	∧ +1.reserved not in {}}<br>	∧ +2.length ≥ 4<br>	∧ +4.roles not in {CALLEE}<br>	∧ ^1.roles not in {BLOCK, OPERATOR}<br>	∧ ^2.roles not in {SCOPE}<br>⇒ y = ⏎⏎<br>Confidence: 0.907. Support: 371.` |
| 50 | `  •••start_col ≥ 15<br>	∧ •••start_line ≥ 25<br>	∧ -1.reserved = ;<br>	∧ +1.reserved not in {}}<br>	∧ +1.length ≥ 1<br>	∧ +2.length ≤ 3<br>	∧ ^1.roles in {SWITCH} and not in {BLOCK, OPERATOR, STATEMENT}<br>	∧ ^2.roles not in {SCOPE}<br>⇒ y = ⏎<br>Confidence: 0.866. Support: 183.` |
| 51 | `  •••start_col ≥ 15<br>	∧ •••start_line ≤ 24<br>	∧ -1.reserved = ;<br>	∧ +1.reserved not in {}}<br>	∧ +1.length ≥ 1<br>	∧ +2.reserved = {<br>	∧ +2.length ≤ 3<br>	∧ ^1.roles not in {BLOCK, OPERATOR, STATEMENT}<br>	∧ ^2.roles not in {SCOPE}<br>⇒ y = ⏎<br>Confidence: 0.826. Support: 112.` |
| 52 | `  •••start_col ≥ 15<br>	∧ -1.reserved = ;<br>	∧ +1.reserved not in {}}<br>	∧ +1.length = 0<br>	∧ +2.length ≤ 3<br>	∧ +4.roles not in {CALLEE}<br>	∧ ^1.roles not in {BLOCK, OPERATOR}<br>	∧ ^2.roles not in {SCOPE}<br>⇒ y = ⏎<br>Confidence: 0.996. Support: 346.` |
| 53 | `  •••start_col ≤ 14<br>	∧ -1.reserved = ;<br>	∧ +1.reserved not in {}}<br>	∧ ^1.internal_type = Program<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ⏎⏎<br>Confidence: 0.894. Support: 1210.` |
| 54 | `  •••start_col ≤ 14<br>	∧ -1.reserved = ;<br>	∧ +1.reserved not in {}}<br>	∧ +3.internal_type = StringLiteral<br>	∧ ^1.internal_type not in {Program}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ⏎⏎<br>Confidence: 0.886. Support: 391.` |
| 55 | `  •••start_col ≤ 14<br>	∧ -1.reserved = ;<br>	∧ +1.reserved not in {}}<br>	∧ +3.internal_type not in {StringLiteral}<br>	∧ ^1.internal_type not in {Program}<br>	∧ ^1.roles not in {OPERATOR}<br>	∧ ^2.internal_type = ClassDeclaration<br>⇒ y = ⏎⏎<br>Confidence: 0.920. Support: 294.` |
| 56 | `  •••start_col ≤ 14<br>	∧ -1.reserved = ;<br>	∧ +1.reserved not in {}}<br>	∧ +3.internal_type not in {StringLiteral}<br>	∧ ^1.internal_type not in {Program}<br>	∧ ^1.roles in {FILE} and not in {OPERATOR}<br>	∧ ^2.internal_type not in {ClassDeclaration}<br>⇒ y = ⏎⏎<br>Confidence: 0.885. Support: 221.` |
| 57 | `  •••start_col ≤ 14<br>	∧ -1.reserved = ;<br>	∧ +1.reserved not in {}}<br>	∧ +3.internal_type not in {StringLiteral}<br>	∧ ^1.internal_type not in {Program}<br>	∧ ^1.roles not in {FILE, FUNCTION, OPERATOR}<br>	∧ ^2.internal_type not in {ClassDeclaration}<br>	∧ ^2.roles not in {BODY}<br>⇒ y = ⏎<br>Confidence: 0.851. Support: 473.` |
| 58 | `  -1.reserved not in {,, ;}<br>	∧ -4.diff_offset ≥ 5<br>	∧ +2.reserved = =<br>	∧ ^1.internal_type = VariableDeclarator<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.908. Support: 322.` |
| 59 | `  •••start_col ≥ 9<br>	∧ -1.reserved not in {,, ;}<br>	∧ -3.label in {<space>}<br>	∧ -4.diff_offset ≥ 5<br>	∧ -5.diff_offset ≥ 13<br>	∧ +1.roles not in {STRING}<br>	∧ +2.reserved not in {=}<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.internal_type = VariableDeclarator<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ␣<br>Confidence: 0.997. Support: 161.` |
| 60 | `  •••start_col ≥ 9<br>	∧ -1.reserved not in {,, ;}<br>	∧ -3.label not in {<space>}<br>	∧ -4.diff_offset ≥ 5<br>	∧ +1.roles not in {STRING}<br>	∧ +2.reserved not in {=}<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.internal_type = VariableDeclarator<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ␣<br>Confidence: 0.987. Support: 11472.` |
| 61 | `  -1.reserved = =<br>	∧ -4.diff_offset ≤ 4<br>	∧ ^1.internal_type = VariableDeclarator<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ␣<br>Confidence: 0.932. Support: 184.` |
| 62 | `  -1.reserved = :<br>	∧ +1.internal_type = StringLiteral<br>	∧ +1.length ≤ 43<br>	∧ ^1.internal_type not in {VariableDeclarator}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ␣'<br>Confidence: 0.961. Support: 2139.` |
| 63 | `  -1.reserved = :<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +2.roles not in {LITERAL}<br>	∧ ^1.internal_type not in {VariableDeclarator}<br>	∧ ^1.roles in {SWITCH} and not in {OPERATOR}<br>⇒ y = ⏎␣⁺␣⁺<br>Confidence: 0.999. Support: 362.` |
| 64 | `  -1.reserved = :<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ ^1.internal_type not in {VariableDeclarator}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ␣<br>Confidence: 0.994. Support: 10533.` |
| 65 | `  -1.internal_type = StringLiteral<br>	∧ -1.reserved not in {,, :, ;}<br>	∧ -3.internal_type = JSXIdentifier<br>	∧ +1.reserved = ><br>	∧ ^1.internal_type not in {VariableDeclarator}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = "<br>Confidence: 0.995. Support: 100.` |
| 66 | `  -1.internal_type = StringLiteral<br>	∧ -1.reserved not in {,, :, ;}<br>	∧ -3.internal_type = JSXIdentifier<br>	∧ -5.label in {<newline>}<br>	∧ +1.reserved not in {>}<br>	∧ ^1.internal_type not in {VariableDeclarator}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = "⏎<br>Confidence: 0.838. Support: 207.` |
| 67 | `  •••start_line = 255<br>	∧ -1.internal_type = StringLiteral<br>	∧ -1.reserved not in {,, :, ;}<br>	∧ -2.diff_col ≤ 101<br>	∧ +1.reserved = ,<br>	∧ ^1.internal_type not in {VariableDeclarator}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = '<br>Confidence: 0.922. Support: 924.` |
| 68 | `  •••start_line ≤ 254<br>	∧ -1.internal_type = StringLiteral<br>	∧ -1.reserved not in {,, :, ;}<br>	∧ -2.diff_col ≤ 101<br>	∧ ^1.internal_type not in {VariableDeclarator}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = '<br>Confidence: 0.987. Support: 6909.` |
| 69 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = {<br>	∧ ^1.internal_type not in {VariableDeclarator}<br>	∧ ^1.roles in {UNANNOTATED} and not in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.952. Support: 2958.` |
| 70 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = {<br>	∧ +1.roles in {VALUE}<br>	∧ ^1.internal_type not in {VariableDeclarator}<br>	∧ ^1.roles not in {OPERATOR, UNANNOTATED}<br>⇒ y = ∅<br>Confidence: 0.817. Support: 408.` |
| 71 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = {<br>	∧ +1.roles in {IMPORT} and not in {VALUE}<br>	∧ ^1.internal_type not in {VariableDeclarator}<br>	∧ ^1.roles not in {OPERATOR, UNANNOTATED}<br>⇒ y = ∅<br>Confidence: 0.951. Support: 175.` |
| 72 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = {<br>	∧ -2.label in {<space>}<br>	∧ +1.roles not in {IMPORT}<br>	∧ +2.reserved = :<br>	∧ ^1.internal_type not in {VariableDeclarator}<br>	∧ ^1.roles not in {INITIALIZATION, OPERATOR, UNANNOTATED}<br>⇒ y = ⏎␣⁺␣⁺<br>Confidence: 0.839. Support: 1343.` |
| 73 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = {<br>	∧ -2.label in {<space>}<br>	∧ +1.roles not in {IMPORT}<br>	∧ ^1.internal_type not in {VariableDeclarator}<br>	∧ ^1.roles not in {OPERATOR, UNANNOTATED}<br>⇒ y = ⏎␣⁺␣⁺<br>Confidence: 0.991. Support: 7317.` |
| 74 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = {<br>	∧ -2.label not in {<space>}<br>	∧ -3.diff_col ≤ 9<br>	∧ +1.roles not in {IMPORT}<br>	∧ ^1.internal_type not in {VariableDeclarator}<br>	∧ ^1.roles in {INITIALIZATION} and not in {OPERATOR, UNANNOTATED}<br>⇒ y = ⏎␣⁺␣⁺<br>Confidence: 0.961. Support: 498.` |
| 75 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = {<br>	∧ -2.label not in {<space>}<br>	∧ +1.roles not in {IMPORT, VALUE}<br>	∧ +3.roles in {VALUE}<br>	∧ ^1.internal_type not in {VariableDeclarator}<br>	∧ ^1.roles not in {INITIALIZATION, OPERATOR, UNANNOTATED}<br>⇒ y = ∅<br>Confidence: 0.992. Support: 194.` |
| 76 | `  •••start_col ≥ 34<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = {<br>	∧ -2.label not in {<space>}<br>	∧ +1.roles in {MAP} and not in {IMPORT}<br>	∧ ^1.internal_type not in {VariableDeclarator}<br>	∧ ^1.roles not in {INITIALIZATION, OPERATOR, UNANNOTATED}<br>⇒ y = ⏎␣⁺␣⁺<br>Confidence: 0.878. Support: 94.` |
| 77 | `  •••start_line = 255<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, :, ;, {}<br>	∧ +1.internal_type = StringLiteral<br>	∧ +1.length ≥ 90<br>	∧ ^1.internal_type not in {VariableDeclarator}<br>	∧ ^1.roles in {EXPRESSION} and not in {OPERATOR}<br>⇒ y = "<br>Confidence: 0.952. Support: 115.` |
| 78 | `  •••start_line = 255<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, :, ;, {}<br>	∧ +1.internal_type = StringLiteral<br>	∧ +1.length ≤ 89<br>	∧ ^1.internal_type not in {VariableDeclarator}<br>	∧ ^1.roles in {EXPRESSION} and not in {OPERATOR}<br>	∧ ^2.roles in {IF}<br>⇒ y = "<br>Confidence: 0.828. Support: 96.` |
| 79 | `  •••start_line = 255<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, :, ;, {}<br>	∧ +1.internal_type = StringLiteral<br>	∧ +1.roles not in {BINARY}<br>	∧ ^1.internal_type not in {VariableDeclarator}<br>	∧ ^1.roles in {EXPRESSION} and not in {OPERATOR}<br>	∧ ^2.roles not in {IF}<br>⇒ y = '<br>Confidence: 0.812. Support: 392.` |
| 80 | `  •••start_line ≤ 254<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, :, ;, {}<br>	∧ +1.internal_type = StringLiteral<br>	∧ ^1.internal_type not in {VariableDeclarator}<br>	∧ ^1.roles in {EXPRESSION} and not in {OPERATOR}<br>⇒ y = '<br>Confidence: 0.945. Support: 4096.` |
| 81 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, :, ;, {}<br>	∧ +1.internal_type = StringLiteral<br>	∧ +2.reserved = ;<br>	∧ ^1.internal_type not in {VariableDeclarator}<br>	∧ ^1.roles not in {EXPRESSION, OPERATOR}<br>⇒ y = ␣'<br>Confidence: 0.893. Support: 284.` |
| 82 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, :, ;, {}<br>	∧ +1.internal_type = StringLiteral<br>	∧ +2.reserved not in {;}<br>	∧ ^1.internal_type not in {VariableDeclarator}<br>	∧ ^1.roles not in {EXPRESSION, OPERATOR}<br>	∧ ^2.internal_type = ObjectExpression<br>⇒ y = '<br>Confidence: 0.889. Support: 166.` |
| 83 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, :, ;, {}<br>	∧ -2.roles in {UNANNOTATED}<br>	∧ +1.internal_type = StringLiteral<br>	∧ +2.reserved not in {;}<br>	∧ ^1.internal_type not in {VariableDeclarator}<br>	∧ ^1.roles not in {EXPRESSION, OPERATOR}<br>	∧ ^2.internal_type not in {ObjectExpression}<br>⇒ y = "<br>Confidence: 0.999. Support: 471.` |
| 84 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, :, ;, {}<br>	∧ -2.label in {<space>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = {<br>	∧ ^1.roles in {UNANNOTATED} and not in {OPERATOR}<br>⇒ y = ␣<br>Confidence: 0.978. Support: 113.` |
| 85 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, :, ;, {}<br>	∧ -2.label not in {<space>}<br>	∧ -2.roles in {UNANNOTATED}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = {<br>	∧ ^1.internal_type not in {VariableDeclarator}<br>	∧ ^1.roles in {UNANNOTATED} and not in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 1.000. Support: 2390.` |
| 86 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, :, ;, {}<br>	∧ -2.label not in {<space>}<br>	∧ -2.roles not in {UNANNOTATED}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = {<br>	∧ ^1.internal_type = JSXElement<br>	∧ ^1.roles in {UNANNOTATED} and not in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.998. Support: 269.` |
| 87 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, :, ;, {}<br>	∧ -2.label not in {<space>}<br>	∧ -2.roles not in {UNANNOTATED}<br>	∧ -2.length ≥ 5<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = {<br>	∧ ^1.internal_type not in {VariableDeclarator}<br>	∧ ^1.roles in {UNANNOTATED} and not in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.901. Support: 147.` |
| 88 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = (<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = {<br>	∧ ^1.internal_type not in {VariableDeclarator}<br>	∧ ^1.roles not in {OPERATOR, UNANNOTATED}<br>⇒ y = ∅<br>Confidence: 0.948. Support: 935.` |
| 89 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {(, ,, :, ;, {}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = {<br>	∧ ^1.roles not in {OPERATOR, UNANNOTATED}<br>⇒ y = ␣<br>Confidence: 0.976. Support: 8024.` |
| 90 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = const<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ␣<br>Confidence: 1.000. Support: 4056.` |
| 91 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = if<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ␣<br>Confidence: 1.000. Support: 2352.` |
| 92 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = return<br>	∧ -3.length ≥ 4<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ ^1.internal_type not in {VariableDeclarator}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.856. Support: 94.` |
| 93 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = return<br>	∧ -3.length ≤ 3<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ␣<br>Confidence: 0.993. Support: 2086.` |
| 94 | `  -1.diff_col ≥ 9<br>	∧ -1.diff_offset ≥ 96<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, :, ;, const, if, return, {}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {{}<br>	∧ +1.length ≥ 2<br>	∧ ^1.internal_type not in {VariableDeclarator}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.986. Support: 1648.` |
| 95 | `  -1.diff_col ≥ 9<br>	∧ -1.diff_offset ≤ 95<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, :, ;, const, {}<br>	∧ -5.reserved = ,<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {{}<br>	∧ +1.length ≥ 2<br>	∧ ^1.internal_type = File<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.855. Support: 287.` |
| 96 | `  -1.diff_col ≥ 9<br>	∧ -1.diff_offset ≤ 95<br>	∧ -1.internal_type = CommentLine<br>	∧ -1.reserved not in {,, :, ;, return, {}<br>	∧ -3.label in {<newline>}<br>	∧ -5.reserved not in {,}<br>	∧ +1.reserved not in {{}<br>	∧ +1.length ≥ 2<br>	∧ ^1.internal_type = File<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ⏎<br>Confidence: 0.834. Support: 226.` |
| 97 | `  -1.diff_col ≥ 9<br>	∧ -1.diff_offset ≤ 95<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, :, ;, const, {}<br>	∧ -3.label in {<newline>}<br>	∧ -5.reserved not in {,}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {{}<br>	∧ +1.length ≥ 2<br>	∧ ^1.internal_type = File<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.996. Support: 127.` |
| 98 | `  -1.diff_col ≥ 9<br>	∧ -1.diff_offset ≤ 95<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, :, ;, return, {}<br>	∧ -3.label not in {<newline>}<br>	∧ -5.reserved not in {,}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {{}<br>	∧ +1.length ≥ 2<br>	∧ ^1.internal_type = File<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ⏎<br>Confidence: 0.923. Support: 2457.` |
| 99 | `  -1.diff_col ≥ 9<br>	∧ -1.diff_offset ≤ 95<br>	∧ -1.reserved not in {,, :, ;, {}<br>	∧ +1.internal_type = DirectiveLiteral<br>	∧ +1.reserved not in {{}<br>	∧ +1.length ≥ 2<br>	∧ ^1.internal_type not in {File, VariableDeclarator}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ⏎⏎'<br>Confidence: 0.856. Support: 518.` |
| 100 | `  -1.diff_col ≥ 9<br>	∧ -1.diff_offset ≤ 95<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, :, ;, const, {}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {{}<br>	∧ +1.length ≥ 2<br>	∧ ^1.internal_type not in {File, VariableDeclarator}<br>	∧ ^1.roles in {FUNCTION} and not in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.998. Support: 305.` |
| 101 | `  -1.diff_col ≥ 9<br>	∧ -1.diff_offset ≤ 95<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, ;, {}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.length ≥ 3<br>	∧ ^1.internal_type not in {File}<br>	∧ ^1.roles in {TYPE} and not in {FUNCTION, OPERATOR}<br>⇒ y = ␣<br>Confidence: 0.998. Support: 234.` |
| 102 | `  -1.diff_col ≥ 9<br>	∧ -1.diff_offset ≤ 95<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, :, ;, const, {}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {{}<br>	∧ +1.length ≤ 2<br>	∧ ^1.internal_type not in {File, VariableDeclarator}<br>	∧ ^1.roles not in {FUNCTION, OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.888. Support: 227.` |
| 103 | `  -1.diff_col ≤ 8<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, :, ;, const, if, return, {}<br>	∧ -1.roles in {ARGUMENT}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {{}<br>	∧ +1.length ≥ 2<br>	∧ ^1.internal_type not in {VariableDeclarator}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.995. Support: 753.` |
| 104 | `  -1.diff_col ≤ 8<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, :, ;, const, return, {}<br>	∧ -1.roles not in {ARGUMENT}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {{}<br>	∧ +1.length ≥ 2<br>	∧ +2.reserved = ,<br>	∧ ^1.internal_type not in {VariableDeclarator}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.851. Support: 124.` |
| 105 | `  •••start_col ≥ 13<br>	∧ -1.diff_col ≤ 8<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, ;, return, {}<br>	∧ -1.roles not in {ARGUMENT}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.length ≥ 5<br>	∧ +2.reserved not in {,}<br>	∧ +3.roles in {EXPRESSION}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ␣<br>Confidence: 0.939. Support: 746.` |
| 106 | `  •••start_col ≤ 27<br>	∧ -1.diff_col ≤ 8<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, ;, return, {}<br>	∧ -1.roles not in {ARGUMENT}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.length ≤ 4<br>	∧ +2.reserved not in {,}<br>	∧ ^1.roles not in {OPERATOR}<br>	∧ ^2.internal_type = BlockStatement<br>⇒ y = ␣<br>Confidence: 0.931. Support: 195.` |
| 107 | `  •••start_col ≤ 12<br>	∧ -1.diff_col ≤ 8<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, ;, return, {}<br>	∧ -1.roles not in {ARGUMENT}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.length ≥ 2<br>	∧ +2.reserved not in {,}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ␣<br>Confidence: 0.974. Support: 3760.` |
| 108 | `  •••start_col ≥ 5<br>	∧ -1.diff_col ≤ 2<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = }<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.length ≥ 2<br>	∧ ^1.roles not in {OPERATOR}<br>	∧ ^2.roles in {SCOPE}<br>⇒ y = ␣<br>Confidence: 0.954. Support: 316.` |
| 109 | `  •••start_col ≥ 5<br>	∧ -1.diff_col ≤ 2<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = }<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.length ≥ 2<br>	∧ ^1.internal_type = JSXElement<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.998. Support: 224.` |
| 110 | `  •••start_col ≥ 5<br>	∧ -1.diff_col ≤ 2<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = }<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.length ≥ 2<br>	∧ ^1.internal_type not in {VariableDeclarator}<br>	∧ ^1.roles in {LITERAL} and not in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.997. Support: 153.` |
| 111 | `  •••start_col ≥ 5<br>	∧ -1.diff_col ≤ 2<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = }<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.length ≥ 2<br>	∧ +2.internal_type = StringLiteral<br>	∧ ^1.internal_type not in {JSXElement}<br>	∧ ^1.roles not in {LITERAL, OPERATOR}<br>	∧ ^2.roles not in {SCOPE}<br>⇒ y = ␣<br>Confidence: 0.997. Support: 158.` |
| 112 | `  •••start_col ≥ 5<br>	∧ -1.diff_col ≤ 2<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = }<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.length ≥ 6<br>	∧ +2.internal_type not in {StringLiteral}<br>	∧ ^1.internal_type not in {JSXElement, VariableDeclarator}<br>	∧ ^1.roles not in {LITERAL, OPERATOR}<br>	∧ ^2.roles not in {DECLARATION}<br>⇒ y = ⏎<br>Confidence: 0.902. Support: 1048.` |
| 113 | `  •••start_col ≥ 5<br>	∧ -1.diff_col ≤ 2<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = }<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.length ≤ 5<br>	∧ +2.internal_type not in {StringLiteral}<br>	∧ ^1.internal_type not in {JSXElement, VariableDeclarator}<br>	∧ ^1.roles in {BLOCK} and not in {LITERAL, OPERATOR}<br>	∧ ^2.roles not in {DECLARATION}<br>⇒ y = ⏎<br>Confidence: 0.899. Support: 174.` |
| 114 | `  •••start_col ≤ 5<br>	∧ •••start_line = 255<br>	∧ -1.diff_col ≤ 2<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = }<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.length ≥ 2<br>	∧ +2.roles in {EXPRESSION}<br>	∧ ^1.internal_type not in {VariableDeclarator}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ⏎<br>Confidence: 0.823. Support: 558.` |
| 115 | `  •••start_col ≤ 5<br>	∧ •••start_line = 255<br>	∧ -1.diff_col ≤ 2<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = }<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.length ≥ 2<br>	∧ ^1.internal_type not in {VariableDeclarator}<br>	∧ ^1.roles not in {OPERATOR, STATEMENT}<br>⇒ y = ⏎⏎<br>Confidence: 0.906. Support: 196.` |
| 116 | `  •••start_col ≤ 5<br>	∧ •••start_line ≤ 254<br>	∧ -1.diff_col ≤ 2<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = }<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.length ≥ 2<br>	∧ ^1.internal_type not in {VariableDeclarator}<br>	∧ ^1.roles not in {OPERATOR, STATEMENT}<br>⇒ y = ⏎⏎<br>Confidence: 0.937. Support: 1050.` |
| 117 | `  -1.diff_col ≤ 2<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, ;, return, {, }}<br>	∧ -2.reserved = =<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.length ≥ 2<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ␣<br>Confidence: 0.828. Support: 754.` |
| 118 | `  -1.diff_col ≤ 2<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = =<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.length ≥ 2<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ␣<br>Confidence: 0.962. Support: 412.` |
| 119 | `  -1.diff_col ≤ 2<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = )<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.length ≥ 2<br>	∧ +2.reserved = (<br>	∧ ^1.internal_type not in {VariableDeclarator}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ⏎␣⁺␣⁺<br>Confidence: 0.942. Support: 182.` |
| 120 | `  -1.diff_col ≤ 2<br>	∧ -1.reserved = |<br>	∧ +1.internal_type = StringLiteralTypeAnnotation<br>	∧ +1.length ≥ 2<br>	∧ ^1.internal_type not in {VariableDeclarator}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ␣'<br>Confidence: 0.998. Support: 275.` |
| 121 | `  -1.diff_col ≤ 2<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = |<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.length ≥ 2<br>	∧ ^1.internal_type = UnionTypeAnnotation<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ␣<br>Confidence: 0.988. Support: 129.` |
| 122 | `  -1.diff_col ≤ 2<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = |<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.length ≥ 2<br>	∧ ^1.internal_type not in {VariableDeclarator}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ⏎␣⁺␣⁺<br>Confidence: 0.862. Support: 177.` |
| 123 | `  -1.diff_col ≤ 2<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {), ,, ;, if, {, |, }}<br>	∧ -3.diff_col ≥ 4<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.length ≥ 2<br>	∧ ^1.internal_type = ConditionalExpression<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ␣<br>Confidence: 0.963. Support: 558.` |
| 124 | `  -1.diff_col ≤ 2<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {), ,, :, ;, =, const, {, |, }}<br>	∧ -2.reserved not in {=}<br>	∧ -3.diff_col ≤ 3<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {{}<br>	∧ +1.length ≥ 2<br>	∧ ^1.internal_type = ConditionalExpression<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.900. Support: 417.` |
| 125 | `  •••start_col ≤ 68<br>	∧ -1.diff_col ≤ 2<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = (<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.length ≥ 2<br>	∧ +2.reserved = )<br>	∧ ^1.internal_type not in {ConditionalExpression, VariableDeclarator}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.998. Support: 3168.` |
| 126 | `  •••start_col ≤ 29<br>	∧ -1.diff_col ≤ 2<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = (<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.length ≥ 2<br>	∧ +5.roles in {EXPRESSION}<br>	∧ ^1.internal_type not in {ConditionalExpression, VariableDeclarator}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.892. Support: 2327.` |
| 127 | `  •••start_col ≤ 53<br>	∧ -1.diff_col ≤ 2<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = (<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.length ≥ 2<br>	∧ +5.roles not in {EXPRESSION}<br>	∧ ^1.internal_type not in {ConditionalExpression, VariableDeclarator}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.942. Support: 4638.` |
| 128 | `  -1.diff_col ≤ 2<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {(, ), ,, ;, return, {, |, }}<br>	∧ -3.diff_col ≥ 6<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.length ≥ 2<br>	∧ ^1.roles not in {OPERATOR}<br>	∧ ^2.internal_type = BlockStatement<br>⇒ y = ␣<br>Confidence: 0.824. Support: 111.` |
| 129 | `  -1.diff_col ≤ 2<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {(, ), ,, :, ;, =, const, {, |, }}<br>	∧ -2.reserved not in {=}<br>	∧ -3.diff_col ≤ 5<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {{}<br>	∧ +1.length ≥ 2<br>	∧ ^1.internal_type not in {ConditionalExpression, VariableDeclarator}<br>	∧ ^1.roles not in {OPERATOR}<br>	∧ ^2.internal_type = BlockStatement<br>⇒ y = ∅<br>Confidence: 0.926. Support: 141.` |
| 130 | `  -1.diff_col ≤ 2<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {(, ), ,, :, ;, =, const, {, |, }}<br>	∧ -2.reserved not in {=}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {{}<br>	∧ +1.length ≥ 2<br>	∧ ^1.internal_type not in {ConditionalExpression, VariableDeclarator}<br>	∧ ^1.roles not in {OPERATOR}<br>	∧ ^2.internal_type not in {BlockStatement}<br>⇒ y = ∅<br>Confidence: 0.991. Support: 36628.` |
| 131 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, :, ;, const, if, return}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = }<br>	∧ +1.length ≤ 1<br>	∧ +2.reserved = ><br>	∧ ^1.internal_type not in {VariableDeclarator}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.973. Support: 872.` |
| 132 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, ;, return, {}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = =<br>	∧ +1.length ≤ 1<br>	∧ +2.reserved = ><br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ␣<br>Confidence: 1.000. Support: 2198.` |
| 133 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, ;, return, {}<br>	∧ -2.reserved = <<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = /<br>	∧ +1.length ≤ 1<br>	∧ +2.reserved = ><br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ␣<br>Confidence: 0.996. Support: 117.` |
| 134 | `  •••start_col ≤ 47<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, :, ;, return}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = /<br>	∧ +1.length ≤ 1<br>	∧ +2.reserved = ><br>	∧ ^1.internal_type not in {VariableDeclarator}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ⏎␣⁻␣⁻<br>Confidence: 0.828. Support: 189.` |
| 135 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, :, ;, const, return, {}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {=, {, }}<br>	∧ +1.length ≤ 1<br>	∧ +2.reserved = ><br>	∧ ^1.internal_type not in {VariableDeclarator}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.986. Support: 106.` |
| 136 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, :, ;, const, if, return}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = )<br>	∧ +1.length ≤ 1<br>	∧ +2.reserved not in {>}<br>	∧ ^1.internal_type = ConditionalExpression<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.879. Support: 647.` |
| 137 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, :, ;, const, return}<br>	∧ -5.diff_col ≥ 18<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = :<br>	∧ +1.length ≤ 1<br>	∧ ^1.internal_type = ConditionalExpression<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ⏎<br>Confidence: 0.835. Support: 191.` |
| 138 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, ;, return, {}<br>	∧ -3.diff_offset ≤ 16<br>	∧ -3.label in {<+space>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {), :}<br>	∧ +1.length ≤ 1<br>	∧ ^1.internal_type = ConditionalExpression<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ␣<br>Confidence: 0.998. Support: 208.` |
| 139 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, ;, return, {}<br>	∧ -3.diff_offset ≤ 16<br>	∧ -3.label not in {<+space>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {), :}<br>	∧ +1.length ≤ 1<br>	∧ +2.roles in {EXPRESSION}<br>	∧ +2.length ≤ 8<br>	∧ ^1.internal_type = ConditionalExpression<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ␣<br>Confidence: 0.805. Support: 279.` |
| 140 | `  -1.internal_type = DirectiveLiteral<br>	∧ -1.reserved not in {,, :, ;, return, {}<br>	∧ +1.reserved not in {{}<br>	∧ +1.length ≤ 1<br>	∧ +2.reserved not in {>}<br>	∧ ^1.internal_type not in {VariableDeclarator}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = '<br>Confidence: 0.993. Support: 532.` |
| 141 | `  -1.internal_type not in {DirectiveLiteral, StringLiteral}<br>	∧ -1.reserved not in {,, :, ;, return}<br>	∧ -3.label in {<newline>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = }<br>	∧ +1.length ≤ 1<br>	∧ ^1.internal_type not in {VariableDeclarator}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ⏎␣⁻␣⁻<br>Confidence: 0.908. Support: 1440.` |
| 142 | `  -1.internal_type not in {DirectiveLiteral, StringLiteral}<br>	∧ -1.reserved not in {,, :, ;, const}<br>	∧ -3.label not in {<newline>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = }<br>	∧ +1.length ≤ 1<br>	∧ +2.reserved not in {>}<br>	∧ ^1.internal_type not in {ConditionalExpression, VariableDeclarator}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.928. Support: 3418.` |
| 143 | `  -1.internal_type not in {DirectiveLiteral, StringLiteral}<br>	∧ -1.reserved not in {,, ;, {}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = =<br>	∧ +1.length ≤ 1<br>	∧ ^1.roles in {VARIABLE} and not in {OPERATOR}<br>⇒ y = ␣<br>Confidence: 0.999. Support: 563.` |
| 144 | `  -1.internal_type not in {DirectiveLiteral, StringLiteral}<br>	∧ -1.reserved not in {,, ;, {}<br>	∧ -3.roles in {KEY}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {}}<br>	∧ +1.length ≤ 1<br>	∧ ^1.roles in {VARIABLE} and not in {OPERATOR}<br>⇒ y = ␣<br>Confidence: 0.998. Support: 228.` |
| 145 | `  -1.internal_type not in {DirectiveLiteral, StringLiteral}<br>	∧ -1.reserved not in {,, :, ;, const, {}<br>	∧ -3.roles not in {KEY}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {=, {, }}<br>	∧ +1.length ≤ 1<br>	∧ +2.reserved not in {>}<br>	∧ ^1.internal_type not in {ConditionalExpression, VariableDeclarator}<br>	∧ ^1.roles in {VARIABLE} and not in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.980. Support: 1144.` |
| 146 | `  -1.internal_type = StringLiteralTypeAnnotation<br>	∧ -1.reserved not in {,, :, ;, return}<br>	∧ -4.internal_type = StringLiteralTypeAnnotation<br>	∧ +1.reserved = |<br>	∧ +1.length ≤ 1<br>	∧ ^1.internal_type not in {VariableDeclarator}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = '⏎<br>Confidence: 0.996. Support: 126.` |
| 147 | `  -1.internal_type = StringLiteralTypeAnnotation<br>	∧ -1.reserved not in {,, :, ;}<br>	∧ +1.reserved = |<br>	∧ +1.length ≤ 1<br>	∧ ^1.internal_type not in {VariableDeclarator}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = '␣<br>Confidence: 0.813. Support: 131.` |
| 148 | `  -1.internal_type not in {DirectiveLiteral, StringLiteral}<br>	∧ -1.reserved not in {,, ;, return, {}<br>	∧ -2.reserved = =<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {}}<br>	∧ +1.length ≤ 1<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ␣<br>Confidence: 0.985. Support: 168.` |
| 149 | `  -1.internal_type not in {DirectiveLiteral, StringLiteral}<br>	∧ -1.reserved not in {,, :, ;, {}<br>	∧ -1.length ≥ 3<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {}}<br>	∧ +1.roles in {EXPRESSION}<br>	∧ +1.length ≤ 1<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ␣<br>Confidence: 0.985. Support: 421.` |
| 150 | `  -1.internal_type not in {DirectiveLiteral, StringLiteral}<br>	∧ -1.reserved not in {,, :, ;, const, {}<br>	∧ -1.length ≤ 2<br>	∧ -2.reserved not in {=}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {{, }}<br>	∧ +1.roles in {EXPRESSION}<br>	∧ +1.length ≤ 1<br>	∧ +2.reserved not in {>}<br>	∧ ^1.internal_type not in {ConditionalExpression, VariableDeclarator}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.950. Support: 2314.` |
| 151 | `  -1.internal_type not in {DirectiveLiteral, StringLiteral}<br>	∧ -1.reserved = (<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {{, }}<br>	∧ +1.length ≤ 1<br>	∧ +2.internal_type = JSXIdentifier<br>	∧ ^1.internal_type not in {VariableDeclarator}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ⏎␣⁺␣⁺<br>Confidence: 0.964. Support: 398.` |
| 152 | `  -1.internal_type not in {DirectiveLiteral, StringLiteral}<br>	∧ -1.reserved = (<br>	∧ -2.reserved not in {=}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = )<br>	∧ +1.roles not in {EXPRESSION}<br>	∧ +1.length ≤ 1<br>	∧ +2.reserved not in {>}<br>	∧ ^1.internal_type not in {ConditionalExpression, VariableDeclarator}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 1.000. Support: 4116.` |
| 153 | `  -1.internal_type not in {DirectiveLiteral, StringLiteral}<br>	∧ -1.reserved = (<br>	∧ -2.reserved not in {=}<br>	∧ -3.length ≤ 5<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {), }}<br>	∧ +1.roles not in {EXPRESSION}<br>	∧ +1.length ≤ 1<br>	∧ +2.reserved not in {>}<br>	∧ ^1.internal_type not in {ConditionalExpression, VariableDeclarator}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.854. Support: 1287.` |
| 154 | `  -1.internal_type not in {DirectiveLiteral, StringLiteral}<br>	∧ -1.reserved not in {(, ,, :, ;, {}<br>	∧ -4.label in {<newline>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = =<br>	∧ +1.roles not in {EXPRESSION}<br>	∧ +1.length ≤ 1<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ␣<br>Confidence: 0.998. Support: 237.` |
| 155 | `  -1.internal_type not in {DirectiveLiteral, StringLiteral}<br>	∧ -1.reserved not in {(, ,, :, ;, const, {}<br>	∧ -2.reserved not in {=}<br>	∧ -4.label in {<newline>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {=, {, }}<br>	∧ +1.roles not in {EXPRESSION}<br>	∧ +1.length ≤ 1<br>	∧ +2.reserved not in {>}<br>	∧ ^1.internal_type not in {ConditionalExpression, VariableDeclarator}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.982. Support: 353.` |
| 156 | `  -1.internal_type not in {DirectiveLiteral, StringLiteral}<br>	∧ -1.reserved not in {(, ,, :, ;, return}<br>	∧ -1.roles in {ARGUMENT}<br>	∧ -4.reserved = ,<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = )<br>	∧ +1.roles not in {EXPRESSION}<br>	∧ +1.length ≤ 1<br>	∧ ^1.internal_type not in {VariableDeclarator}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ⏎␣⁻␣⁻<br>Confidence: 0.992. Support: 456.` |
| 157 | `  -1.internal_type not in {DirectiveLiteral, StringLiteral}<br>	∧ -1.reserved not in {(, ,, :, ;, const}<br>	∧ -1.roles not in {ARGUMENT}<br>	∧ -2.reserved not in {=}<br>	∧ -4.label not in {<newline>}<br>	∧ -4.reserved = ,<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = )<br>	∧ +1.roles not in {EXPRESSION}<br>	∧ +1.length ≤ 1<br>	∧ +2.reserved not in {>}<br>	∧ ^1.internal_type not in {ConditionalExpression, VariableDeclarator}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.921. Support: 1037.` |
| 158 | `  -1.internal_type not in {DirectiveLiteral, StringLiteral}<br>	∧ -1.reserved not in {(, ,, :, ;, const}<br>	∧ -1.roles in {ARGUMENT}<br>	∧ -2.reserved not in {=}<br>	∧ -4.label not in {<newline>}<br>	∧ -4.reserved not in {,}<br>	∧ -5.label in {<newline>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = )<br>	∧ +1.roles not in {EXPRESSION}<br>	∧ +1.length ≤ 1<br>	∧ +2.reserved not in {>}<br>	∧ ^1.internal_type not in {ConditionalExpression, VariableDeclarator}<br>	∧ ^1.roles not in {OPERATOR}<br>	∧ ^2.roles in {STATEMENT}<br>⇒ y = ∅<br>Confidence: 0.997. Support: 161.` |
| 159 | `  -1.internal_type not in {DirectiveLiteral, StringLiteral}<br>	∧ -1.reserved not in {(, ,, :, ;, const}<br>	∧ -2.reserved not in {=}<br>	∧ -4.label not in {<newline>}<br>	∧ -4.reserved not in {,}<br>	∧ -5.label not in {<newline>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = )<br>	∧ +1.roles not in {EXPRESSION}<br>	∧ +1.length ≤ 1<br>	∧ +2.reserved not in {>}<br>	∧ ^1.internal_type not in {ConditionalExpression, VariableDeclarator}<br>	∧ ^1.roles not in {OPERATOR}<br>	∧ ^2.roles in {STATEMENT}<br>⇒ y = ∅<br>Confidence: 0.957. Support: 5664.` |
| 160 | `  -1.internal_type not in {DirectiveLiteral, StringLiteral}<br>	∧ -1.reserved not in {(, ,, :, ;, const}<br>	∧ -2.reserved not in {=}<br>	∧ -4.label not in {<newline>}<br>	∧ -4.reserved not in {,}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = )<br>	∧ +1.roles not in {EXPRESSION}<br>	∧ +1.length ≤ 1<br>	∧ +2.reserved not in {>}<br>	∧ ^1.internal_type not in {ConditionalExpression, VariableDeclarator}<br>	∧ ^1.roles not in {OPERATOR}<br>	∧ ^2.roles not in {STATEMENT}<br>⇒ y = ∅<br>Confidence: 0.989. Support: 8565.` |
| 161 | `  -1.internal_type not in {DirectiveLiteral, StringLiteral}<br>	∧ -1.reserved not in {(, ,, :, ;, const}<br>	∧ -1.roles in {EXPRESSION}<br>	∧ -2.reserved not in {=}<br>	∧ -4.label not in {<newline>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = (<br>	∧ +1.roles not in {EXPRESSION}<br>	∧ +1.length ≤ 1<br>	∧ +2.reserved not in {>}<br>	∧ ^1.internal_type not in {ConditionalExpression, VariableDeclarator}<br>	∧ ^1.roles not in {OPERATOR}<br>	∧ ^2.roles in {BLOCK}<br>⇒ y = ∅<br>Confidence: 0.997. Support: 165.` |
| 162 | `  -1.internal_type not in {DirectiveLiteral, StringLiteral}<br>	∧ -1.reserved not in {(, ,, :, ;, const, {}<br>	∧ -2.reserved not in {=}<br>	∧ -4.label not in {<newline>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {(, ), {, }}<br>	∧ +1.roles not in {EXPRESSION}<br>	∧ +1.length ≤ 1<br>	∧ +2.reserved not in {>}<br>	∧ ^1.internal_type not in {ConditionalExpression, VariableDeclarator}<br>	∧ ^1.roles not in {OPERATOR}<br>	∧ ^2.roles in {BLOCK}<br>⇒ y = ∅<br>Confidence: 0.994. Support: 1323.` |
| 163 | `  -1.internal_type not in {DirectiveLiteral, StringLiteral}<br>	∧ -1.reserved not in {(, ,, ;, if, {}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = =<br>	∧ +1.roles not in {EXPRESSION}<br>	∧ +1.length ≤ 1<br>	∧ ^1.roles not in {OPERATOR}<br>	∧ ^2.roles in {MODULE} and not in {BLOCK}<br>⇒ y = ␣<br>Confidence: 0.997. Support: 171.` |
| 164 | `  -1.internal_type not in {DirectiveLiteral, StringLiteral}<br>	∧ -1.reserved not in {(, ,, :, ;, const, {}<br>	∧ -2.reserved not in {=}<br>	∧ -4.label not in {<newline>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {), =, {, }}<br>	∧ +1.roles not in {EXPRESSION}<br>	∧ +1.length ≤ 1<br>	∧ +2.reserved not in {>}<br>	∧ ^1.internal_type not in {ConditionalExpression, VariableDeclarator}<br>	∧ ^1.roles not in {OPERATOR}<br>	∧ ^2.roles in {MODULE} and not in {BLOCK}<br>⇒ y = ∅<br>Confidence: 0.963. Support: 1939.` |
| 165 | `  -1.internal_type not in {DirectiveLiteral, StringLiteral}<br>	∧ -1.reserved not in {(, ,, :, ;, const, {}<br>	∧ -2.reserved = |<br>	∧ -4.label not in {<newline>}<br>	∧ -5.diff_offset ≤ 11<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {), {, }}<br>	∧ +1.roles not in {EXPRESSION}<br>	∧ +1.length ≤ 1<br>	∧ +2.reserved not in {>}<br>	∧ ^1.internal_type not in {ConditionalExpression, VariableDeclarator}<br>	∧ ^1.roles not in {OPERATOR}<br>	∧ ^2.roles not in {BLOCK, MODULE}<br>⇒ y = ∅<br>Confidence: 0.996. Support: 139.` |
| 166 | `  -1.internal_type not in {DirectiveLiteral, StringLiteral}<br>	∧ -1.reserved not in {(, ,, :, ;, {}<br>	∧ -3.label not in {<-space>}<br>	∧ -4.reserved = ;<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {), }}<br>	∧ +1.roles not in {EXPRESSION}<br>	∧ +1.length ≤ 1<br>	∧ ^1.internal_type = BlockStatement<br>	∧ ^1.roles not in {OPERATOR}<br>	∧ ^2.roles not in {BLOCK, MODULE}<br>⇒ y = ␣<br>Confidence: 0.857. Support: 221.` |
| 167 | `  -1.internal_type not in {DirectiveLiteral, StringLiteral}<br>	∧ -1.reserved not in {(, ,, :, ;, const, {}<br>	∧ -2.reserved not in {=}<br>	∧ -4.label not in {<newline>}<br>	∧ -4.reserved = ;<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {), {, }}<br>	∧ +1.roles not in {EXPRESSION}<br>	∧ +1.length ≤ 1<br>	∧ +2.reserved not in {>}<br>	∧ ^1.internal_type not in {BlockStatement, ConditionalExpression, VariableDeclarator}<br>	∧ ^1.roles not in {OPERATOR}<br>	∧ ^2.roles not in {BLOCK, MODULE}<br>⇒ y = ∅<br>Confidence: 0.998. Support: 5054.` |
| 168 | `  -1.internal_type not in {DirectiveLiteral, StringLiteral}<br>	∧ -1.reserved not in {(, ,, :, ;, const, {}<br>	∧ -2.reserved not in {=}<br>	∧ -4.label not in {<newline>}<br>	∧ -4.reserved not in {;}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {), {, }}<br>	∧ +1.roles not in {EXPRESSION}<br>	∧ +1.length ≤ 1<br>	∧ +2.reserved not in {>}<br>	∧ ^1.internal_type not in {ConditionalExpression, VariableDeclarator}<br>	∧ ^1.roles not in {OPERATOR}<br>	∧ ^2.roles not in {BLOCK, MODULE}<br>⇒ y = ∅<br>Confidence: 0.994. Support: 35741.` |

<details>
    <summary>Machine-readable report</summary>
```json
{"javascript": {"avg_rule_len": 11.005952380952381, "max_conf": 0.9998785257339478, "max_support": 36628, "min_conf": 0.8046594858169556, "min_support": 94, "num_rules": 168}}
```
</details>
