# Model report for file:///tmp/top-repos-quality-repos-0qyyera1/jquery HEAD dae5f3ce3d2df27873d01f0d9682f6a91ad66b87

### Dump

```json
{'created_at': datetime.datetime(2019, 1, 31, 12, 31, 43, 634973),
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
 'uuid': 'efa18240-2972-45ba-8edf-90d843e93b0e',
 'version': [1]}
style.format.analyzer.FormatAnalyzer/[1] file:///tmp/top-repos-quality-repos-0qyyera1/jquery dae5f3ce3d2df27873d01f0d9682f6a91ad66b87

# javascript
346 rules, avg.len. 9.3
## train
PPCR: 0.962217
### report
macro
{'f1-score': 0.5935705344460303,
 'precision': 0.6149490388860678,
 'recall': 0.5827445576205211,
 'support': 157157}
micro
{'f1-score': 0.9641123208002189,
 'precision': 0.9641123208002189,
 'recall': 0.9641123208002189,
 'support': 157157}
weighted
{'f1-score': 0.9607482462652023,
 'precision': 0.9585740155380525,
 'recall': 0.9641123208002189,
 'support': 157157}
### report_full
macro
{'f1-score': 0.5651865045534894,
 'precision': 0.6149490388860678,
 'recall': 0.5376346668653246,
 'support': 163328}
micro
{'f1-score': 0.9455481535797308,
 'precision': 0.9641123208002189,
 'recall': 0.9276853938087775,
 'support': 163328}
weighted
{'f1-score': 0.9360556755880663,
 'precision': 0.9537703241331674,
 'recall': 0.9276853938087775,
 'support': 163328}
## test
PPCR: 0.960380
### report
macro
{'f1-score': 0.5901673954160161,
 'precision': 0.6110897843840531,
 'recall': 0.5783621677060018,
 'support': 39899}
micro
{'f1-score': 0.9726058297200431,
 'precision': 0.9726058297200431,
 'recall': 0.9726058297200431,
 'support': 39899}
weighted
{'f1-score': 0.9693435489864289,
 'precision': 0.9672623671295342,
 'recall': 0.9726058297200431,
 'support': 39899}
### report_full
macro
{'f1-score': 0.5562506273025661,
 'precision': 0.6110897843840531,
 'recall': 0.529540455061859,
 'support': 41545}
micro
{'f1-score': 0.9529492657531554,
 'precision': 0.9726058297200431,
 'recall': 0.9340714887471416,
 'support': 41545}
weighted
{'f1-score': 0.941114159238579,
 'precision': 0.9619547710855728,
 'recall': 0.9340714887471416,
 'support': 41545}
```

## javascript
### Summary
346 rules, avg.len. 9.3

| | |
|-|-|
|Min support|129|
|Max support|24572|
|Min confidence|0.8004807829856873|
|Max confidence|0.9998798370361328|

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
                     'base_model_name': 'sklearn.ensemble.RandomForestClassifier',
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
| 1 | `  -2.label in {<newline>}<br>	∧ +1.internal_type = StringLiteral<br>⇒ y = "<br>Confidence: 1.000. Support: 1134.` |
| 2 | `  -2.label not in {<newline>}<br>	∧ -4.label in {"}<br>	∧ +1.internal_type = StringLiteral<br>⇒ y = ⏎<br>Confidence: 0.831. Support: 582.` |
| 3 | `  -2.label not in {<newline>}<br>	∧ -4.label not in {"}<br>	∧ -5.label in {"}<br>	∧ +1.internal_type = StringLiteral<br>⇒ y = ⏎<br>Confidence: 0.975. Support: 224.` |
| 4 | `  -2.label not in {<newline>}<br>	∧ -4.label not in {"}<br>	∧ -5.label not in {"}<br>	∧ +1.internal_type = StringLiteral<br>⇒ y = ␣"<br>Confidence: 0.958. Support: 9443.` |
| 5 | `  -1.internal_type = StringLiteral<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = )<br>⇒ y = "␣<br>Confidence: 0.983. Support: 5073.` |
| 6 | `  -1.internal_type = StringLiteral<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {)}<br>	∧ ^1.roles in {OPERATOR}<br>⇒ y = "␣<br>Confidence: 0.999. Support: 552.` |
| 7 | `  -1.internal_type = StringLiteral<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {), ], }}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = "<br>Confidence: 0.982. Support: 4241.` |
| 8 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = [<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ ^1.internal_type = MemberExpression<br>⇒ y = ␣<br>Confidence: 0.996. Support: 714.` |
| 9 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {[}<br>	∧ -2.roles in {IDENTIFIER}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = )<br>	∧ ^1.internal_type = MemberExpression<br>⇒ y = ∅<br>Confidence: 0.999. Support: 395.` |
| 10 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {[}<br>	∧ -2.roles not in {IDENTIFIER}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = )<br>	∧ ^1.internal_type = MemberExpression<br>⇒ y = ␣<br>Confidence: 0.993. Support: 744.` |
| 11 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {[}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = ]<br>	∧ ^1.internal_type = MemberExpression<br>⇒ y = ␣<br>Confidence: 0.998. Support: 226.` |
| 12 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {[}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = }<br>	∧ ^1.internal_type = MemberExpression<br>⇒ y = ⏎⇥⁻<br>Confidence: 0.833. Support: 129.` |
| 13 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {[}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {), ], }}<br>	∧ ^1.internal_type = MemberExpression<br>⇒ y = ∅<br>Confidence: 0.989. Support: 23182.` |
| 14 | `  •••start_col ≤ 6<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = ;<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.roles in {EXPRESSION}<br>	∧ ^1.internal_type not in {MemberExpression}<br>⇒ y = ⏎⏎<br>Confidence: 0.819. Support: 795.` |
| 15 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = {<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.roles in {EXPRESSION}<br>	∧ +3.roles not in {VALUE}<br>	∧ ^1.internal_type not in {MemberExpression}<br>⇒ y = ⏎⇥⁺<br>Confidence: 0.901. Support: 1941.` |
| 16 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {;, {}<br>	∧ -1.roles in {COMMENT}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.roles in {EXPRESSION}<br>	∧ ^1.internal_type not in {MemberExpression}<br>⇒ y = ⏎<br>Confidence: 0.966. Support: 634.` |
| 17 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {;, {}<br>	∧ -1.roles not in {COMMENT}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.roles in {EXPRESSION} and not in {KEY}<br>	∧ +2.reserved not in {=}<br>	∧ ^1.internal_type = VariableDeclaration<br>⇒ y = ␣<br>Confidence: 0.950. Support: 309.` |
| 18 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {;, {}<br>	∧ -1.roles not in {COMMENT}<br>	∧ -3.diff_offset ≥ 4<br>	∧ -5.diff_offset ≥ 7<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.roles in {EXPRESSION} and not in {KEY}<br>	∧ ^1.internal_type not in {MemberExpression, VariableDeclaration}<br>⇒ y = ␣<br>Confidence: 0.965. Support: 14313.` |
| 19 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = ,<br>	∧ -1.roles not in {COMMENT}<br>	∧ -3.diff_offset ≤ 3<br>	∧ -5.diff_offset ≥ 7<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.roles in {EXPRESSION} and not in {KEY}<br>	∧ ^1.internal_type not in {MemberExpression, VariableDeclaration}<br>⇒ y = ␣<br>Confidence: 0.950. Support: 553.` |
| 20 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, ;, {}<br>	∧ -1.roles not in {COMMENT}<br>	∧ -3.diff_offset ≤ 3<br>	∧ -5.diff_offset ≥ 7<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.roles in {EXPRESSION} and not in {KEY}<br>	∧ +3.roles in {EXPRESSION}<br>	∧ ^1.internal_type not in {MemberExpression, VariableDeclaration}<br>⇒ y = ␣<br>Confidence: 0.824. Support: 349.` |
| 21 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {;, {}<br>	∧ -1.roles not in {COMMENT}<br>	∧ -3.reserved = (<br>	∧ -5.diff_offset ≤ 6<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.roles in {EXPRESSION} and not in {KEY}<br>	∧ ^1.internal_type not in {MemberExpression, VariableDeclaration}<br>⇒ y = ∅<br>Confidence: 0.927. Support: 239.` |
| 22 | `  •••start_col ≥ 4<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {;, {}<br>	∧ -1.roles not in {COMMENT}<br>	∧ -3.reserved not in {(}<br>	∧ -5.diff_offset ≤ 6<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.roles in {EXPRESSION} and not in {KEY}<br>	∧ ^1.internal_type not in {MemberExpression, VariableDeclaration}<br>⇒ y = ␣<br>Confidence: 0.830. Support: 840.` |
| 23 | `  -1.internal_type not in {StringLiteral}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.roles not in {EXPRESSION}<br>	∧ +2.roles in {ARGUMENT}<br>	∧ ^1.internal_type not in {MemberExpression}<br>⇒ y = ∅<br>Confidence: 0.993. Support: 11644.` |
| 24 | `  -1.internal_type not in {StringLiteral}<br>	∧ -2.diff_offset ≤ 13<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = ;<br>	∧ +1.roles not in {EXPRESSION}<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.internal_type not in {MemberExpression}<br>⇒ y = ∅<br>Confidence: 0.997. Support: 7879.` |
| 25 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.roles in {NUMBER}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = }<br>	∧ +1.roles not in {EXPRESSION}<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.internal_type not in {MemberExpression}<br>⇒ y = ␣<br>Confidence: 0.904. Support: 162.` |
| 26 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.roles not in {NUMBER}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = }<br>	∧ +1.roles not in {EXPRESSION}<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.internal_type not in {MemberExpression}<br>⇒ y = ⏎⇥⁻<br>Confidence: 0.883. Support: 2929.` |
| 27 | `  -1.internal_type not in {StringLiteral}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = ,<br>	∧ +1.roles not in {EXPRESSION}<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.internal_type not in {MemberExpression}<br>⇒ y = ∅<br>Confidence: 1.000. Support: 3355.` |
| 28 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = if<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = (<br>	∧ +1.roles not in {EXPRESSION}<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.internal_type not in {MemberExpression}<br>⇒ y = ␣<br>Confidence: 0.999. Support: 562.` |
| 29 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {if}<br>	∧ -1.roles in {IDENTIFIER}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = (<br>	∧ +1.roles not in {EXPRESSION}<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.internal_type not in {MemberExpression}<br>⇒ y = ∅<br>Confidence: 1.000. Support: 4143.` |
| 30 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {if}<br>	∧ -1.roles not in {IDENTIFIER}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = (<br>	∧ +1.roles not in {EXPRESSION}<br>	∧ +2.reserved = )<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.internal_type not in {MemberExpression}<br>⇒ y = ∅<br>Confidence: 0.999. Support: 530.` |
| 31 | `  -1.internal_type not in {StringLiteral}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {,, ;, }}<br>	∧ +1.roles in {COMMENT} and not in {EXPRESSION}<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.internal_type not in {MemberExpression}<br>⇒ y = ∅<br>Confidence: 0.990. Support: 1479.` |
| 32 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.roles in {KEY}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {,, ;, }}<br>	∧ +1.roles not in {COMMENT, EXPRESSION}<br>	∧ ^1.internal_type not in {MemberExpression}<br>⇒ y = ∅<br>Confidence: 1.000. Support: 1310.` |
| 33 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = ;<br>	∧ -1.roles not in {KEY}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {(, ,, ;, return, }}<br>	∧ +1.roles not in {EXPRESSION}<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ +2.length ≥ 2<br>	∧ ^1.internal_type not in {MemberExpression}<br>⇒ y = ⏎⏎<br>Confidence: 0.819. Support: 899.` |
| 34 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = ;<br>	∧ -1.roles not in {KEY}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {(, ,, ;, }}<br>	∧ +1.roles not in {COMMENT, EXPRESSION}<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ +2.length ≤ 1<br>	∧ ^1.internal_type not in {MemberExpression}<br>	∧ ^1.roles not in {FUNCTION}<br>⇒ y = ⏎<br>Confidence: 0.838. Support: 226.` |
| 35 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = (<br>	∧ -1.roles not in {KEY}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = )<br>	∧ +1.roles not in {EXPRESSION}<br>	∧ ^1.internal_type not in {MemberExpression}<br>⇒ y = ∅<br>Confidence: 0.998. Support: 1630.` |
| 36 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = (<br>	∧ -1.roles not in {KEY}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {(, ), ,, }}<br>	∧ +1.roles not in {COMMENT, EXPRESSION}<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.internal_type not in {MemberExpression}<br>⇒ y = ␣<br>Confidence: 0.979. Support: 996.` |
| 37 | `  -1.internal_type = CommentLine<br>	∧ -1.reserved not in {(, ;}<br>	∧ -1.roles not in {KEY}<br>	∧ +1.reserved not in {(, ,, ;, }}<br>	∧ +1.roles not in {COMMENT, EXPRESSION}<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.internal_type not in {MemberExpression}<br>⇒ y = ⏎<br>Confidence: 0.973. Support: 824.` |
| 38 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved = {<br>	∧ -1.roles not in {KEY}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {,, }}<br>	∧ +1.roles not in {COMMENT, EXPRESSION}<br>	∧ +2.roles in {COMMENT} and not in {ARGUMENT}<br>	∧ ^1.internal_type not in {MemberExpression}<br>⇒ y = ⏎⏎⇥⁺<br>Confidence: 0.916. Support: 185.` |
| 39 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved = {<br>	∧ -1.roles not in {KEY}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {,, }}<br>	∧ +1.roles not in {COMMENT, EXPRESSION}<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.internal_type not in {MemberExpression}<br>⇒ y = ⏎⇥⁺<br>Confidence: 0.909. Support: 774.` |
| 40 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {(, ;, {}<br>	∧ -1.roles not in {KEY}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {(, ,, ;, }}<br>	∧ +1.roles not in {COMMENT, EXPRESSION}<br>	∧ +2.roles in {COMMENT} and not in {ARGUMENT}<br>	∧ ^1.internal_type not in {MemberExpression}<br>⇒ y = ⏎⏎<br>Confidence: 0.856. Support: 296.` |
| 41 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {(, ;, {}<br>	∧ -1.roles not in {KEY}<br>	∧ -2.diff_offset ≤ 19<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {(, ,, ;, }}<br>	∧ +1.roles not in {COMMENT, EXPRESSION}<br>	∧ +1.length ≥ 2<br>	∧ +2.roles not in {ARGUMENT, COMMENT}<br>	∧ ^1.internal_type not in {BlockStatement, MemberExpression}<br>	∧ ^1.roles not in {LIST}<br>⇒ y = ␣<br>Confidence: 0.942. Support: 2871.` |
| 42 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {(, ;, {}<br>	∧ -1.roles not in {KEY}<br>	∧ -2.diff_offset ≤ 19<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {(, ,, ;, }}<br>	∧ +1.roles not in {COMMENT, EXPRESSION}<br>	∧ +1.length ≤ 1<br>	∧ +2.roles not in {ARGUMENT, COMMENT}<br>	∧ ^1.internal_type not in {MemberExpression}<br>	∧ ^1.roles not in {LIST}<br>⇒ y = ␣<br>Confidence: 0.979. Support: 13673.` |
| 43 | `  -2.label in {<newline>}<br>	∧ +1.roles in {STRING}<br>⇒ y = "<br>Confidence: 1.000. Support: 1116.` |
| 44 | `  -2.label not in {<newline>}<br>	∧ -4.label in {"}<br>	∧ +1.roles in {STRING}<br>⇒ y = ⏎<br>Confidence: 0.811. Support: 590.` |
| 45 | `  -2.label not in {<newline>}<br>	∧ -4.label not in {"}<br>	∧ -5.label in {"}<br>	∧ +1.roles in {STRING}<br>⇒ y = ⏎<br>Confidence: 0.958. Support: 252.` |
| 46 | `  -2.label not in {<newline>}<br>	∧ -4.label not in {"}<br>	∧ -5.label not in {"}<br>	∧ +1.roles in {STRING}<br>⇒ y = ␣"<br>Confidence: 0.960. Support: 9101.` |
| 47 | `  -1.roles in {STRING}<br>	∧ +1.reserved = )<br>	∧ +1.roles not in {STRING}<br>⇒ y = "␣<br>Confidence: 0.980. Support: 4971.` |
| 48 | `  -1.roles in {STRING}<br>	∧ +1.reserved not in {)}<br>	∧ +1.roles not in {STRING}<br>	∧ ^1.roles in {OPERATOR}<br>⇒ y = "␣<br>Confidence: 0.999. Support: 568.` |
| 49 | `  -1.roles in {STRING}<br>	∧ +1.reserved = ]<br>	∧ +1.roles not in {STRING}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = "␣<br>Confidence: 0.818. Support: 370.` |
| 50 | `  -1.roles in {STRING}<br>	∧ +1.reserved not in {), ], }}<br>	∧ +1.roles not in {STRING}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = "<br>Confidence: 0.982. Support: 4225.` |
| 51 | `  -1.reserved = [<br>	∧ -1.roles not in {STRING}<br>	∧ +1.roles not in {STRING}<br>	∧ ^1.roles in {IDENTIFIER}<br>⇒ y = ␣<br>Confidence: 0.996. Support: 667.` |
| 52 | `  -1.reserved not in {[}<br>	∧ -1.roles not in {STRING}<br>	∧ -2.roles in {IDENTIFIER}<br>	∧ +1.reserved = )<br>	∧ +1.roles not in {STRING}<br>	∧ ^1.roles in {IDENTIFIER}<br>⇒ y = ∅<br>Confidence: 0.999. Support: 387.` |
| 53 | `  -1.reserved not in {[}<br>	∧ -1.roles not in {STRING}<br>	∧ -2.roles not in {IDENTIFIER}<br>	∧ +1.reserved = )<br>	∧ +1.roles not in {STRING}<br>	∧ ^1.roles in {IDENTIFIER}<br>⇒ y = ␣<br>Confidence: 0.991. Support: 725.` |
| 54 | `  -1.reserved not in {[}<br>	∧ -1.roles not in {STRING}<br>	∧ +1.reserved = ]<br>	∧ +1.roles not in {STRING}<br>	∧ ^1.roles in {IDENTIFIER}<br>⇒ y = ␣<br>Confidence: 0.997. Support: 197.` |
| 55 | `  -1.reserved = )<br>	∧ -1.roles not in {STRING}<br>	∧ +1.reserved not in {), ]}<br>	∧ +1.roles not in {STRING}<br>	∧ ^1.roles in {IDENTIFIER}<br>⇒ y = ∅<br>Confidence: 0.891. Support: 2237.` |
| 56 | `  -1.reserved not in {), [}<br>	∧ -1.roles not in {STRING}<br>	∧ -2.diff_offset ≥ 4<br>	∧ +1.reserved not in {), ]}<br>	∧ +1.roles not in {STRING}<br>	∧ +2.reserved = )<br>	∧ ^1.roles in {IDENTIFIER}<br>⇒ y = ∅<br>Confidence: 0.944. Support: 685.` |
| 57 | `  -1.reserved not in {), [}<br>	∧ -1.roles not in {STRING}<br>	∧ +1.reserved not in {), ]}<br>	∧ +1.roles not in {STRING}<br>	∧ +2.reserved not in {)}<br>	∧ ^1.roles in {IDENTIFIER}<br>⇒ y = ∅<br>Confidence: 0.999. Support: 20445.` |
| 58 | `  •••start_col ≤ 6<br>	∧ -1.reserved = ;<br>	∧ -1.roles not in {STRING}<br>	∧ +1.roles in {EXPRESSION} and not in {STRING}<br>	∧ ^1.roles not in {IDENTIFIER}<br>⇒ y = ⏎⏎<br>Confidence: 0.828. Support: 749.` |
| 59 | `  -1.reserved = {<br>	∧ -1.roles not in {STRING}<br>	∧ +1.roles in {EXPRESSION} and not in {STRING}<br>	∧ +3.roles not in {VALUE}<br>	∧ ^1.roles not in {IDENTIFIER}<br>⇒ y = ⏎⇥⁺<br>Confidence: 0.886. Support: 1997.` |
| 60 | `  -1.diff_col ≥ 9<br>	∧ -1.reserved not in {;, {}<br>	∧ -1.roles not in {STRING}<br>	∧ +1.roles in {EXPRESSION} and not in {STRING}<br>	∧ ^1.roles not in {IDENTIFIER}<br>⇒ y = ⏎<br>Confidence: 0.977. Support: 704.` |
| 61 | `  -1.diff_col ≤ 8<br>	∧ -1.reserved not in {;, {}<br>	∧ -1.roles not in {STRING}<br>	∧ +1.roles in {EXPRESSION} and not in {KEY, STRING}<br>	∧ +2.reserved not in {=}<br>	∧ ^1.internal_type = VariableDeclaration<br>	∧ ^1.roles not in {IDENTIFIER}<br>⇒ y = ␣<br>Confidence: 0.917. Support: 309.` |
| 62 | `  -1.diff_col ≤ 8<br>	∧ -1.reserved not in {;, {}<br>	∧ -1.roles not in {STRING}<br>	∧ -3.diff_offset ≥ 4<br>	∧ -5.diff_offset ≥ 7<br>	∧ +1.roles in {EXPRESSION} and not in {KEY, STRING}<br>	∧ ^1.internal_type not in {VariableDeclaration}<br>	∧ ^1.roles not in {IDENTIFIER}<br>⇒ y = ␣<br>Confidence: 0.963. Support: 14325.` |
| 63 | `  -1.diff_col ≤ 8<br>	∧ -1.reserved = ,<br>	∧ -1.roles not in {STRING}<br>	∧ -3.diff_offset ≤ 3<br>	∧ -5.diff_offset ≥ 7<br>	∧ +1.roles in {EXPRESSION} and not in {KEY, STRING}<br>	∧ ^1.internal_type not in {VariableDeclaration}<br>	∧ ^1.roles not in {IDENTIFIER}<br>⇒ y = ␣<br>Confidence: 0.962. Support: 568.` |
| 64 | `  -1.diff_col ≤ 8<br>	∧ -1.reserved not in {,, ;, {}<br>	∧ -1.roles not in {STRING}<br>	∧ -3.diff_offset ≤ 3<br>	∧ -5.diff_offset ≥ 7<br>	∧ +1.roles in {EXPRESSION} and not in {KEY, STRING}<br>	∧ +3.roles in {EXPRESSION}<br>	∧ ^1.internal_type not in {VariableDeclaration}<br>	∧ ^1.roles not in {IDENTIFIER}<br>⇒ y = ␣<br>Confidence: 0.832. Support: 349.` |
| 65 | `  -1.diff_col ≤ 8<br>	∧ -1.reserved not in {;, {}<br>	∧ -1.roles not in {STRING}<br>	∧ -3.reserved = (<br>	∧ -5.diff_offset ≤ 6<br>	∧ +1.roles in {EXPRESSION} and not in {KEY, STRING}<br>	∧ ^1.internal_type not in {VariableDeclaration}<br>	∧ ^1.roles not in {IDENTIFIER}<br>⇒ y = ∅<br>Confidence: 0.936. Support: 243.` |
| 66 | `  •••start_col ≥ 4<br>	∧ -1.diff_col ≤ 8<br>	∧ -1.reserved not in {;, {}<br>	∧ -1.roles not in {STRING}<br>	∧ -3.reserved not in {(}<br>	∧ -5.diff_offset ≤ 6<br>	∧ +1.roles in {EXPRESSION} and not in {KEY, STRING}<br>	∧ ^1.internal_type not in {VariableDeclaration}<br>	∧ ^1.roles not in {IDENTIFIER}<br>⇒ y = ␣<br>Confidence: 0.827. Support: 817.` |
| 67 | `  -1.roles not in {STRING}<br>	∧ +1.roles not in {EXPRESSION, STRING}<br>	∧ +2.roles in {ARGUMENT}<br>	∧ ^1.roles not in {IDENTIFIER}<br>⇒ y = ∅<br>Confidence: 0.993. Support: 11519.` |
| 68 | `  -1.roles not in {STRING}<br>	∧ +1.reserved = ;<br>	∧ +1.roles not in {EXPRESSION, STRING}<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.roles not in {IDENTIFIER}<br>⇒ y = ∅<br>Confidence: 0.987. Support: 8151.` |
| 69 | `  •••start_col ≥ 42<br>	∧ -1.roles in {EXPRESSION} and not in {STRING}<br>	∧ +1.reserved = }<br>	∧ +1.roles not in {EXPRESSION, STRING}<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.roles not in {IDENTIFIER}<br>⇒ y = ␣<br>Confidence: 0.971. Support: 153.` |
| 70 | `  -1.roles not in {EXPRESSION, STRING}<br>	∧ +1.reserved = }<br>	∧ +1.roles not in {EXPRESSION, STRING}<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.roles not in {IDENTIFIER}<br>⇒ y = ⏎⇥⁻<br>Confidence: 0.893. Support: 2898.` |
| 71 | `  -1.roles not in {STRING}<br>	∧ +1.reserved = ,<br>	∧ +1.roles not in {EXPRESSION, STRING}<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.roles not in {IDENTIFIER}<br>⇒ y = ∅<br>Confidence: 1.000. Support: 3346.` |
| 72 | `  -1.reserved = if<br>	∧ -1.roles not in {STRING}<br>	∧ +1.reserved = (<br>	∧ +1.roles not in {EXPRESSION, STRING}<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.roles not in {IDENTIFIER}<br>⇒ y = ␣<br>Confidence: 0.999. Support: 570.` |
| 73 | `  -1.reserved not in {if}<br>	∧ -1.roles in {IDENTIFIER} and not in {STRING}<br>	∧ +1.reserved = (<br>	∧ +1.roles not in {EXPRESSION, STRING}<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.roles not in {IDENTIFIER}<br>⇒ y = ∅<br>Confidence: 1.000. Support: 4140.` |
| 74 | `  -1.reserved not in {if}<br>	∧ -1.roles not in {IDENTIFIER, STRING}<br>	∧ +1.reserved = (<br>	∧ +1.roles not in {EXPRESSION, STRING}<br>	∧ +2.reserved = )<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.roles not in {IDENTIFIER}<br>⇒ y = ∅<br>Confidence: 0.999. Support: 555.` |
| 75 | `  -1.roles not in {STRING}<br>	∧ +1.reserved not in {,, ;, }}<br>	∧ +1.roles in {COMMENT} and not in {EXPRESSION, STRING}<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.roles not in {IDENTIFIER}<br>⇒ y = ∅<br>Confidence: 0.992. Support: 1535.` |
| 76 | `  -1.roles in {KEY} and not in {STRING}<br>	∧ +1.reserved not in {,, ;, }}<br>	∧ +1.roles not in {COMMENT, EXPRESSION, STRING}<br>	∧ ^1.roles not in {IDENTIFIER}<br>⇒ y = ∅<br>Confidence: 1.000. Support: 1352.` |
| 77 | `  -1.reserved = ;<br>	∧ -1.roles not in {KEY, STRING}<br>	∧ +1.reserved not in {(, ,, ;, return, }}<br>	∧ +1.roles not in {EXPRESSION, STRING}<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ +2.length ≥ 2<br>	∧ ^1.roles not in {IDENTIFIER}<br>⇒ y = ⏎⏎<br>Confidence: 0.806. Support: 990.` |
| 78 | `  -1.reserved = ;<br>	∧ -1.roles not in {KEY, STRING}<br>	∧ +1.reserved not in {(, ,, ;, }}<br>	∧ +1.roles not in {COMMENT, EXPRESSION, STRING}<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ +2.length ≤ 1<br>	∧ ^1.roles not in {IDENTIFIER}<br>	∧ ^2.roles not in {FUNCTION}<br>⇒ y = ⏎<br>Confidence: 0.830. Support: 203.` |
| 79 | `  -1.reserved = (<br>	∧ -1.roles not in {KEY, STRING}<br>	∧ +1.reserved = )<br>	∧ +1.roles not in {EXPRESSION, STRING}<br>	∧ ^1.roles not in {IDENTIFIER}<br>⇒ y = ∅<br>Confidence: 1.000. Support: 1676.` |
| 80 | `  -1.reserved = (<br>	∧ -1.roles not in {KEY, STRING}<br>	∧ +1.reserved not in {(, ), ,, }}<br>	∧ +1.roles not in {COMMENT, EXPRESSION, STRING}<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.roles not in {IDENTIFIER}<br>⇒ y = ␣<br>Confidence: 0.978. Support: 1025.` |
| 81 | `  -1.internal_type = CommentLine<br>	∧ -1.reserved not in {(, ;}<br>	∧ -1.roles not in {KEY, STRING}<br>	∧ +1.reserved not in {(, ,, ;, }}<br>	∧ +1.roles not in {COMMENT, EXPRESSION, STRING}<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.roles not in {IDENTIFIER}<br>⇒ y = ⏎<br>Confidence: 0.974. Support: 812.` |
| 82 | `  -1.internal_type not in {CommentLine}<br>	∧ -1.reserved = {<br>	∧ -1.roles not in {KEY, STRING}<br>	∧ +1.reserved not in {,, }}<br>	∧ +1.roles not in {COMMENT, EXPRESSION, STRING}<br>	∧ +2.internal_type = CommentLine<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.roles not in {IDENTIFIER}<br>⇒ y = ⏎⏎⇥⁺<br>Confidence: 0.926. Support: 195.` |
| 83 | `  -1.internal_type not in {CommentLine}<br>	∧ -1.reserved = {<br>	∧ -1.roles not in {KEY, STRING}<br>	∧ +1.reserved not in {,, }}<br>	∧ +1.roles not in {COMMENT, EXPRESSION, STRING}<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.roles not in {IDENTIFIER}<br>⇒ y = ⏎⇥⁺<br>Confidence: 0.922. Support: 785.` |
| 84 | `  -1.internal_type not in {CommentLine}<br>	∧ -1.reserved not in {(, ;, {}<br>	∧ -1.roles not in {KEY, STRING}<br>	∧ +1.reserved not in {(, ,, ;, }}<br>	∧ +1.roles not in {COMMENT, EXPRESSION, STRING}<br>	∧ +2.roles in {COMMENT} and not in {ARGUMENT}<br>	∧ ^1.roles not in {IDENTIFIER}<br>⇒ y = ⏎⏎<br>Confidence: 0.838. Support: 305.` |
| 85 | `  -1.internal_type not in {CommentLine}<br>	∧ -1.reserved not in {(, ;, {}<br>	∧ -1.roles not in {KEY, STRING}<br>	∧ -2.diff_offset ≤ 19<br>	∧ -3.label in {<-tab>}<br>	∧ +1.reserved not in {(, ,, ;, }}<br>	∧ +1.roles not in {COMMENT, EXPRESSION, STRING}<br>	∧ +1.length ≥ 2<br>	∧ +2.roles not in {ARGUMENT, COMMENT}<br>	∧ ^1.roles not in {IDENTIFIER, LIST, SCOPE}<br>⇒ y = ␣<br>Confidence: 0.993. Support: 210.` |
| 86 | `  -1.internal_type not in {CommentLine}<br>	∧ -1.reserved not in {(, ;, {}<br>	∧ -1.roles not in {KEY, STRING}<br>	∧ -2.diff_offset ≤ 19<br>	∧ -3.label in {<-tab>}<br>	∧ +1.reserved not in {(, ,, ;, }}<br>	∧ +1.roles not in {COMMENT, EXPRESSION, STRING}<br>	∧ +1.length ≤ 1<br>	∧ +2.roles not in {ARGUMENT, COMMENT}<br>	∧ ^1.roles not in {IDENTIFIER, LIST}<br>⇒ y = ␣<br>Confidence: 0.906. Support: 699.` |
| 87 | `  -1.internal_type not in {CommentLine}<br>	∧ -1.reserved not in {(, ;, {}<br>	∧ -1.roles not in {KEY, STRING}<br>	∧ -2.diff_offset ≤ 19<br>	∧ -3.label in {<space>} and not in {<-tab>}<br>	∧ +1.reserved not in {(, ,, ;, }}<br>	∧ +1.roles not in {COMMENT, EXPRESSION, STRING}<br>	∧ +2.roles not in {ARGUMENT, COMMENT}<br>	∧ ^1.roles not in {IDENTIFIER, LIST}<br>⇒ y = ␣<br>Confidence: 0.802. Support: 577.` |
| 88 | `  -1.internal_type not in {CommentLine}<br>	∧ -1.reserved not in {(, ;, {}<br>	∧ -1.roles not in {KEY, STRING}<br>	∧ -2.diff_offset ≤ 19<br>	∧ -3.label not in {<-tab>, <space>}<br>	∧ +1.reserved not in {(, ,, ;, }}<br>	∧ +1.roles not in {COMMENT, EXPRESSION, STRING}<br>	∧ +1.length ≥ 2<br>	∧ +2.roles not in {ARGUMENT, COMMENT}<br>	∧ ^1.roles not in {IDENTIFIER, LIST}<br>	∧ ^2.roles not in {FUNCTION}<br>⇒ y = ␣<br>Confidence: 0.953. Support: 2233.` |
| 89 | `  -1.internal_type not in {CommentLine}<br>	∧ -1.reserved not in {(, ;, {}<br>	∧ -1.roles not in {KEY, STRING}<br>	∧ -2.diff_offset ≤ 19<br>	∧ -3.label not in {<-tab>, <space>}<br>	∧ +1.reserved not in {(, ,, ;, }}<br>	∧ +1.roles not in {COMMENT, EXPRESSION, STRING}<br>	∧ +1.length ≤ 1<br>	∧ +2.roles not in {ARGUMENT, COMMENT}<br>	∧ ^1.roles not in {IDENTIFIER, LIST}<br>⇒ y = ␣<br>Confidence: 0.985. Support: 12461.` |
| 90 | `  -1.roles in {STRING}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = )<br>⇒ y = "␣<br>Confidence: 0.979. Support: 4997.` |
| 91 | `  -1.roles in {STRING}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {)}<br>	∧ ^1.roles in {OPERATOR}<br>⇒ y = "␣<br>Confidence: 0.999. Support: 550.` |
| 92 | `  -1.roles in {STRING}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = ]<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = "␣<br>Confidence: 0.811. Support: 373.` |
| 93 | `  -1.roles in {STRING}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {), ], }}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = "<br>Confidence: 0.985. Support: 4220.` |
| 94 | `  -1.reserved = [<br>	∧ -1.roles not in {STRING}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ ^1.roles in {QUALIFIED}<br>⇒ y = ␣<br>Confidence: 0.999. Support: 731.` |
| 95 | `  -1.reserved not in {[}<br>	∧ -1.roles not in {STRING}<br>	∧ -2.roles in {IDENTIFIER}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = )<br>	∧ ^1.roles in {QUALIFIED}<br>⇒ y = ∅<br>Confidence: 0.999. Support: 378.` |
| 96 | `  -1.reserved not in {[}<br>	∧ -1.roles not in {STRING}<br>	∧ -2.roles not in {IDENTIFIER}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = )<br>	∧ ^1.roles in {QUALIFIED}<br>⇒ y = ␣<br>Confidence: 0.991. Support: 738.` |
| 97 | `  -1.reserved not in {[}<br>	∧ -1.roles not in {STRING}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = ]<br>	∧ ^1.roles in {QUALIFIED}<br>⇒ y = ␣<br>Confidence: 0.997. Support: 190.` |
| 98 | `  -1.reserved = )<br>	∧ -1.roles not in {STRING}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {), ]}<br>	∧ ^1.roles in {QUALIFIED}<br>⇒ y = ∅<br>Confidence: 0.890. Support: 2220.` |
| 99 | `  -1.reserved not in {), [}<br>	∧ -1.roles not in {STRING}<br>	∧ -2.diff_col ≥ 4<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {), ]}<br>	∧ +2.reserved = )<br>	∧ ^1.roles in {QUALIFIED}<br>⇒ y = ∅<br>Confidence: 0.947. Support: 668.` |
| 100 | `  -1.reserved not in {), [}<br>	∧ -1.roles not in {STRING}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {), ]}<br>	∧ +2.reserved not in {)}<br>	∧ ^1.roles in {QUALIFIED}<br>⇒ y = ∅<br>Confidence: 0.999. Support: 20617.` |
| 101 | `  •••start_col ≥ 7<br>	∧ -1.reserved = ;<br>	∧ -1.roles not in {STRING}<br>	∧ -4.length ≥ 9<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.roles in {EXPRESSION, LEFT}<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ⏎⏎<br>Confidence: 0.835. Support: 173.` |
| 102 | `  -1.reserved = {<br>	∧ -1.roles not in {STRING}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.roles in {EXPRESSION}<br>	∧ +3.roles not in {VALUE}<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ⏎⇥⁺<br>Confidence: 0.875. Support: 1996.` |
| 103 | `  -1.diff_col ≥ 9<br>	∧ -1.reserved not in {;, {}<br>	∧ -1.roles not in {STRING}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.roles in {EXPRESSION}<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ⏎<br>Confidence: 0.969. Support: 651.` |
| 104 | `  -1.diff_col ≤ 8<br>	∧ -1.reserved not in {;, {}<br>	∧ -1.roles not in {STRING}<br>	∧ -2.internal_type = Identifier<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.roles in {EXPRESSION}<br>	∧ +2.reserved = =<br>	∧ ^1.internal_type = VariableDeclaration<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ⏎⇥⁺<br>Confidence: 0.804. Support: 196.` |
| 105 | `  -1.diff_col ≤ 8<br>	∧ -1.reserved not in {;, {}<br>	∧ -1.roles not in {STRING}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.roles in {EXPRESSION} and not in {KEY}<br>	∧ +2.reserved not in {=}<br>	∧ ^1.internal_type = VariableDeclaration<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ␣<br>Confidence: 0.947. Support: 333.` |
| 106 | `  -1.diff_col ≤ 8<br>	∧ -1.reserved not in {;, {}<br>	∧ -1.roles not in {STRING}<br>	∧ -3.diff_offset ≥ 4<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.roles in {EXPRESSION} and not in {KEY}<br>	∧ ^1.internal_type not in {VariableDeclaration}<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ␣<br>Confidence: 0.967. Support: 14310.` |
| 107 | `  -1.diff_col ≤ 8<br>	∧ -1.reserved = ,<br>	∧ -1.roles not in {STRING}<br>	∧ -3.diff_offset ≤ 3<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.roles in {EXPRESSION} and not in {KEY}<br>	∧ ^1.internal_type not in {VariableDeclaration}<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ␣<br>Confidence: 0.964. Support: 825.` |
| 108 | `  -1.diff_col ≤ 8<br>	∧ -1.reserved not in {,, ;, {}<br>	∧ -1.roles not in {STRING}<br>	∧ -3.diff_offset ≤ 3<br>	∧ -3.reserved = (<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.roles in {EXPRESSION} and not in {KEY}<br>	∧ ^1.internal_type not in {VariableDeclaration}<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ∅<br>Confidence: 0.849. Support: 322.` |
| 109 | `  -1.roles not in {STRING}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.roles not in {EXPRESSION}<br>	∧ +2.roles in {ARGUMENT}<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ∅<br>Confidence: 0.994. Support: 11791.` |
| 110 | `  -1.roles not in {STRING}<br>	∧ -3.diff_line = 0<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = ;<br>	∧ +1.roles not in {EXPRESSION}<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ∅<br>Confidence: 0.997. Support: 7829.` |
| 111 | `  •••start_col ≥ 37<br>	∧ -1.roles in {EXPRESSION} and not in {STRING}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = }<br>	∧ +1.roles not in {EXPRESSION}<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ␣<br>Confidence: 0.973. Support: 166.` |
| 112 | `  -1.roles not in {EXPRESSION, STRING}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = }<br>	∧ +1.roles not in {EXPRESSION}<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ⏎⇥⁻<br>Confidence: 0.895. Support: 2797.` |
| 113 | `  -1.roles not in {STRING}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = ,<br>	∧ +1.roles not in {EXPRESSION}<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ∅<br>Confidence: 1.000. Support: 3319.` |
| 114 | `  -1.reserved = if<br>	∧ -1.roles not in {STRING}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = (<br>	∧ +1.roles not in {EXPRESSION}<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ␣<br>Confidence: 0.999. Support: 570.` |
| 115 | `  -1.reserved not in {if}<br>	∧ -1.roles in {EXPRESSION} and not in {STRING}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = (<br>	∧ +1.roles not in {EXPRESSION}<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ∅<br>Confidence: 1.000. Support: 4148.` |
| 116 | `  -1.reserved not in {if}<br>	∧ -1.roles not in {EXPRESSION, STRING}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = (<br>	∧ +1.roles not in {EXPRESSION}<br>	∧ +2.reserved = )<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ∅<br>Confidence: 0.999. Support: 523.` |
| 117 | `  -1.roles not in {STRING}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {,, ;, }}<br>	∧ +1.roles in {COMMENT} and not in {EXPRESSION}<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ∅<br>Confidence: 0.994. Support: 1479.` |
| 118 | `  -1.roles in {KEY} and not in {STRING}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {,, ;, }}<br>	∧ +1.roles not in {COMMENT, EXPRESSION}<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ∅<br>Confidence: 1.000. Support: 1322.` |
| 119 | `  -1.reserved = (<br>	∧ -1.roles not in {KEY, STRING}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = )<br>	∧ +1.roles not in {EXPRESSION}<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ∅<br>Confidence: 0.999. Support: 1707.` |
| 120 | `  -1.reserved = (<br>	∧ -1.roles not in {KEY, STRING}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {(, ), ,, ;, }}<br>	∧ +1.roles not in {COMMENT, EXPRESSION}<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ␣<br>Confidence: 0.975. Support: 952.` |
| 121 | `  -1.reserved = {<br>	∧ -1.roles not in {KEY, STRING}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {,, ;, }}<br>	∧ +1.roles not in {COMMENT, EXPRESSION}<br>	∧ +1.length ≥ 2<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.roles in {EXPRESSION} and not in {QUALIFIED}<br>⇒ y = ⏎⇥⁺<br>Confidence: 0.920. Support: 421.` |
| 122 | `  -1.reserved not in {(, {}<br>	∧ -1.roles not in {KEY, STRING}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {(, ,, ;, }}<br>	∧ +1.roles not in {COMMENT, EXPRESSION}<br>	∧ +1.length ≥ 2<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.roles in {EXPRESSION} and not in {QUALIFIED}<br>⇒ y = ␣<br>Confidence: 0.944. Support: 1981.` |
| 123 | `  -1.internal_type = CommentLine<br>	∧ -1.reserved not in {(}<br>	∧ -1.roles not in {KEY, STRING}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {(, ,, ;, }}<br>	∧ +1.roles not in {COMMENT, EXPRESSION}<br>	∧ +1.length ≥ 2<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.roles not in {EXPRESSION, QUALIFIED}<br>⇒ y = ⏎<br>Confidence: 0.981. Support: 754.` |
| 124 | `  -1.internal_type not in {CommentLine}<br>	∧ -1.reserved not in {(}<br>	∧ -1.roles not in {KEY, STRING}<br>	∧ -2.label in {<space>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {(, ,, ;, }}<br>	∧ +1.roles not in {COMMENT, EXPRESSION}<br>	∧ +1.length ≥ 2<br>	∧ +2.roles in {COMMENT} and not in {ARGUMENT}<br>	∧ +2.length ≥ 3<br>	∧ ^1.roles not in {EXPRESSION, QUALIFIED}<br>⇒ y = ⏎⏎⇥⁺<br>Confidence: 0.856. Support: 205.` |
| 125 | `  -1.internal_type not in {CommentLine}<br>	∧ -1.reserved not in {(}<br>	∧ -1.roles not in {KEY, STRING}<br>	∧ -2.label not in {<space>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {(, ,, ;, }}<br>	∧ +1.roles not in {COMMENT, EXPRESSION}<br>	∧ +1.length ≤ 5<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ +2.length ≥ 3<br>	∧ ^1.roles not in {EXPRESSION, QUALIFIED}<br>⇒ y = ⏎⏎<br>Confidence: 0.804. Support: 1197.` |
| 126 | `  -1.internal_type not in {CommentLine}<br>	∧ -1.reserved not in {(}<br>	∧ -1.roles not in {KEY, STRING}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {(, ,, ;, }}<br>	∧ +1.roles not in {COMMENT, EXPRESSION}<br>	∧ +1.length ≥ 4<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ +2.length ≤ 2<br>	∧ ^1.roles not in {EXPRESSION, QUALIFIED}<br>⇒ y = ␣<br>Confidence: 0.836. Support: 752.` |
| 127 | `  -1.reserved not in {(}<br>	∧ -1.roles not in {KEY, STRING}<br>	∧ -3.label not in {<space>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {(, ,, ;, }}<br>	∧ +1.roles not in {COMMENT, EXPRESSION}<br>	∧ +1.length ≤ 1<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ␣<br>Confidence: 0.977. Support: 13605.` |
| 128 | `  -1.reserved = [<br>	∧ -1.roles not in {STRING}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ ^1.internal_type = MemberExpression<br>⇒ y = ␣<br>Confidence: 0.998. Support: 708.` |
| 129 | `  -1.reserved not in {[}<br>	∧ -1.roles not in {STRING}<br>	∧ -2.roles in {IDENTIFIER}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = )<br>	∧ ^1.internal_type = MemberExpression<br>⇒ y = ∅<br>Confidence: 0.999. Support: 362.` |
| 130 | `  -1.reserved not in {[}<br>	∧ -1.roles not in {STRING}<br>	∧ -2.roles not in {IDENTIFIER}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = )<br>	∧ ^1.internal_type = MemberExpression<br>⇒ y = ␣<br>Confidence: 0.991. Support: 754.` |
| 131 | `  -1.reserved not in {[}<br>	∧ -1.roles not in {STRING}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = ]<br>	∧ ^1.internal_type = MemberExpression<br>⇒ y = ␣<br>Confidence: 0.998. Support: 224.` |
| 132 | `  -1.reserved = )<br>	∧ -1.roles not in {STRING}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {), ]}<br>	∧ ^1.internal_type = MemberExpression<br>⇒ y = ∅<br>Confidence: 0.881. Support: 2174.` |
| 133 | `  -1.reserved not in {), [}<br>	∧ -1.roles not in {STRING}<br>	∧ -2.diff_col ≥ 4<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {), ]}<br>	∧ +2.reserved = )<br>	∧ ^1.internal_type = MemberExpression<br>⇒ y = ∅<br>Confidence: 0.952. Support: 683.` |
| 134 | `  -1.reserved not in {), [}<br>	∧ -1.roles not in {STRING}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {), ]}<br>	∧ +2.reserved not in {)}<br>	∧ ^1.internal_type = MemberExpression<br>⇒ y = ∅<br>Confidence: 0.998. Support: 20585.` |
| 135 | `  •••start_col ≤ 6<br>	∧ -1.reserved = ;<br>	∧ -1.roles not in {STRING}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.roles in {EXPRESSION}<br>	∧ ^1.internal_type not in {MemberExpression}<br>⇒ y = ⏎⏎<br>Confidence: 0.803. Support: 720.` |
| 136 | `  -1.reserved = {<br>	∧ -1.roles not in {STRING}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.roles in {EXPRESSION}<br>	∧ ^1.internal_type not in {MemberExpression}<br>⇒ y = ⏎⇥⁺<br>Confidence: 0.849. Support: 2197.` |
| 137 | `  -1.reserved not in {;, {}<br>	∧ -1.roles in {COMMENT} and not in {STRING}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.roles in {EXPRESSION}<br>	∧ ^1.internal_type not in {MemberExpression}<br>⇒ y = ⏎<br>Confidence: 0.977. Support: 627.` |
| 138 | `  -1.reserved not in {;, {}<br>	∧ -1.roles not in {COMMENT, STRING}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.roles in {EXPRESSION, KEY}<br>	∧ +1.length ≥ 4<br>	∧ ^1.internal_type not in {MemberExpression}<br>⇒ y = ⏎<br>Confidence: 0.846. Support: 732.` |
| 139 | `  -1.reserved not in {;, {}<br>	∧ -1.roles not in {COMMENT, STRING}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.roles in {EXPRESSION} and not in {KEY}<br>	∧ +2.reserved not in {=}<br>	∧ ^1.internal_type = VariableDeclaration<br>⇒ y = ␣<br>Confidence: 0.914. Support: 296.` |
| 140 | `  -1.reserved not in {;, {}<br>	∧ -1.roles not in {COMMENT, STRING}<br>	∧ -3.diff_offset ≥ 4<br>	∧ -5.diff_offset ≥ 7<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.roles in {EXPRESSION} and not in {KEY}<br>	∧ ^1.internal_type not in {MemberExpression, VariableDeclaration}<br>⇒ y = ␣<br>Confidence: 0.964. Support: 14413.` |
| 141 | `  -1.reserved = ,<br>	∧ -1.roles not in {COMMENT, STRING}<br>	∧ -3.diff_offset ≤ 3<br>	∧ -5.diff_offset ≥ 7<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.roles in {EXPRESSION} and not in {KEY}<br>	∧ ^1.internal_type not in {MemberExpression, VariableDeclaration}<br>⇒ y = ␣<br>Confidence: 0.948. Support: 563.` |
| 142 | `  -1.reserved not in {;, {}<br>	∧ -1.roles not in {COMMENT, STRING}<br>	∧ -3.reserved = (<br>	∧ -5.diff_offset ≤ 6<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.roles in {EXPRESSION} and not in {KEY}<br>	∧ ^1.internal_type not in {MemberExpression, VariableDeclaration}<br>⇒ y = ∅<br>Confidence: 0.953. Support: 247.` |
| 143 | `  •••start_col ≥ 4<br>	∧ -1.reserved not in {;, {}<br>	∧ -1.roles not in {COMMENT, STRING}<br>	∧ -3.reserved not in {(}<br>	∧ -5.diff_offset ≤ 6<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.roles in {EXPRESSION} and not in {KEY}<br>	∧ ^1.internal_type not in {MemberExpression, VariableDeclaration}<br>⇒ y = ␣<br>Confidence: 0.869. Support: 784.` |
| 144 | `  -1.roles not in {STRING}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.roles not in {EXPRESSION}<br>	∧ +2.roles in {ARGUMENT}<br>	∧ ^1.internal_type not in {MemberExpression}<br>⇒ y = ∅<br>Confidence: 0.995. Support: 11806.` |
| 145 | `  -1.roles not in {STRING}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = ;<br>	∧ +1.roles not in {EXPRESSION}<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.internal_type not in {MemberExpression}<br>⇒ y = ∅<br>Confidence: 0.987. Support: 7950.` |
| 146 | `  -1.roles not in {EXPRESSION, STRING}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = }<br>	∧ +1.roles not in {EXPRESSION}<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.internal_type not in {MemberExpression}<br>⇒ y = ⏎⇥⁻<br>Confidence: 0.893. Support: 2771.` |
| 147 | `  -1.roles not in {STRING}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = ,<br>	∧ +1.roles not in {EXPRESSION}<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.internal_type not in {MemberExpression}<br>⇒ y = ∅<br>Confidence: 1.000. Support: 3344.` |
| 148 | `  -1.reserved = if<br>	∧ -1.roles not in {STRING}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = (<br>	∧ +1.roles not in {EXPRESSION}<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.internal_type not in {MemberExpression}<br>⇒ y = ␣<br>Confidence: 0.999. Support: 588.` |
| 149 | `  -1.reserved not in {if}<br>	∧ -1.roles in {EXPRESSION} and not in {STRING}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = (<br>	∧ +1.roles not in {EXPRESSION}<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.internal_type not in {MemberExpression}<br>⇒ y = ∅<br>Confidence: 1.000. Support: 4050.` |
| 150 | `  -1.reserved not in {if}<br>	∧ -1.roles not in {EXPRESSION, STRING}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = (<br>	∧ +1.roles not in {EXPRESSION}<br>	∧ +2.reserved = )<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.internal_type not in {MemberExpression}<br>⇒ y = ∅<br>Confidence: 0.999. Support: 549.` |
| 151 | `  -1.roles not in {STRING}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {,, ;, }}<br>	∧ +1.roles in {COMMENT} and not in {EXPRESSION}<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.internal_type not in {MemberExpression}<br>⇒ y = ∅<br>Confidence: 0.992. Support: 1552.` |
| 152 | `  -1.roles in {KEY} and not in {STRING}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {,, ;, }}<br>	∧ +1.roles not in {COMMENT, EXPRESSION}<br>	∧ ^1.internal_type not in {MemberExpression}<br>⇒ y = ∅<br>Confidence: 1.000. Support: 1369.` |
| 153 | `  -1.reserved = ;<br>	∧ -1.roles not in {KEY, STRING}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {(, ,, ;, }}<br>	∧ +1.roles not in {COMMENT, EXPRESSION}<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ +2.length ≤ 1<br>	∧ ^1.internal_type not in {MemberExpression}<br>	∧ ^2.roles not in {FUNCTION}<br>⇒ y = ⏎<br>Confidence: 0.800. Support: 208.` |
| 154 | `  -1.reserved = (<br>	∧ -1.roles not in {KEY, STRING}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = )<br>	∧ +1.roles not in {EXPRESSION}<br>	∧ ^1.internal_type not in {MemberExpression}<br>⇒ y = ∅<br>Confidence: 0.999. Support: 1616.` |
| 155 | `  -1.reserved = (<br>	∧ -1.roles not in {KEY, STRING}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {(, ), ,, }}<br>	∧ +1.roles not in {COMMENT, EXPRESSION}<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.internal_type not in {MemberExpression}<br>⇒ y = ␣<br>Confidence: 0.982. Support: 992.` |
| 156 | `  -1.internal_type = CommentLine<br>	∧ -1.reserved not in {(, ;}<br>	∧ -1.roles not in {KEY, STRING}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {(, ,, ;, }}<br>	∧ +1.roles not in {COMMENT, EXPRESSION}<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.internal_type not in {MemberExpression}<br>⇒ y = ⏎<br>Confidence: 0.964. Support: 798.` |
| 157 | `  -1.internal_type not in {CommentLine}<br>	∧ -1.reserved = {<br>	∧ -1.roles not in {KEY, STRING}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {,, }}<br>	∧ +1.roles not in {COMMENT, EXPRESSION}<br>	∧ +2.roles in {COMMENT} and not in {ARGUMENT}<br>	∧ ^1.internal_type not in {MemberExpression}<br>⇒ y = ⏎⏎⇥⁺<br>Confidence: 0.939. Support: 187.` |
| 158 | `  -1.internal_type not in {CommentLine}<br>	∧ -1.reserved = {<br>	∧ -1.roles not in {KEY, STRING}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {,, }}<br>	∧ +1.roles not in {COMMENT, EXPRESSION}<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.internal_type not in {MemberExpression}<br>⇒ y = ⏎⇥⁺<br>Confidence: 0.894. Support: 724.` |
| 159 | `  -1.internal_type not in {CommentLine}<br>	∧ -1.reserved not in {(, ;, {}<br>	∧ -1.roles not in {KEY, STRING}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {(, ,, ;, }}<br>	∧ +1.roles not in {COMMENT, EXPRESSION}<br>	∧ +2.roles in {COMMENT} and not in {ARGUMENT}<br>	∧ ^1.internal_type not in {MemberExpression}<br>⇒ y = ⏎⏎<br>Confidence: 0.826. Support: 331.` |
| 160 | `  -1.internal_type not in {CommentLine}<br>	∧ -1.reserved not in {(, ;, {}<br>	∧ -1.roles not in {KEY, STRING}<br>	∧ -3.label not in {<space>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {(, ,, ;, }}<br>	∧ +1.roles not in {COMMENT, EXPRESSION}<br>	∧ +1.length ≥ 2<br>	∧ +2.roles not in {ARGUMENT, COMMENT}<br>	∧ ^1.internal_type not in {MemberExpression}<br>	∧ ^1.roles not in {SCOPE}<br>⇒ y = ␣<br>Confidence: 0.935. Support: 2505.` |
| 161 | `  -1.internal_type not in {CommentLine}<br>	∧ -1.reserved not in {(, ;, {}<br>	∧ -1.roles not in {KEY, STRING}<br>	∧ -3.label not in {<space>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {(, ,, ;, }}<br>	∧ +1.roles not in {COMMENT, EXPRESSION}<br>	∧ +1.length ≤ 1<br>	∧ +2.roles not in {ARGUMENT, COMMENT}<br>	∧ ^1.internal_type not in {MemberExpression}<br>⇒ y = ␣<br>Confidence: 0.979. Support: 13448.` |
| 162 | `  -1.reserved = [<br>	∧ -1.roles not in {STRING}<br>	∧ +1.roles not in {STRING}<br>	∧ ^1.roles in {QUALIFIED}<br>⇒ y = ␣<br>Confidence: 0.999. Support: 699.` |
| 163 | `  -1.reserved not in {[}<br>	∧ -1.roles not in {STRING}<br>	∧ -2.roles in {IDENTIFIER}<br>	∧ +1.reserved = )<br>	∧ +1.roles not in {STRING}<br>	∧ ^1.roles in {QUALIFIED}<br>⇒ y = ∅<br>Confidence: 0.999. Support: 374.` |
| 164 | `  -1.reserved not in {[}<br>	∧ -1.roles not in {STRING}<br>	∧ -2.roles not in {IDENTIFIER}<br>	∧ +1.reserved = )<br>	∧ +1.roles not in {STRING}<br>	∧ ^1.roles in {QUALIFIED}<br>⇒ y = ␣<br>Confidence: 0.991. Support: 755.` |
| 165 | `  -1.reserved not in {[}<br>	∧ -1.roles not in {STRING}<br>	∧ +1.reserved = ]<br>	∧ +1.roles not in {STRING}<br>	∧ ^1.roles in {QUALIFIED}<br>⇒ y = ␣<br>Confidence: 0.998. Support: 207.` |
| 166 | `  -1.reserved not in {[}<br>	∧ -1.roles not in {STRING}<br>	∧ -2.diff_col ≥ 4<br>	∧ +1.reserved not in {), ]}<br>	∧ +1.roles not in {STRING}<br>	∧ +2.reserved = )<br>	∧ ^1.roles in {QUALIFIED}<br>⇒ y = ∅<br>Confidence: 0.949. Support: 697.` |
| 167 | `  -1.reserved not in {[}<br>	∧ -1.roles not in {STRING}<br>	∧ +1.reserved not in {), ]}<br>	∧ +1.roles not in {STRING}<br>	∧ +2.reserved not in {)}<br>	∧ ^1.roles in {QUALIFIED}<br>⇒ y = ∅<br>Confidence: 0.988. Support: 22640.` |
| 168 | `  •••start_col ≥ 74<br>	∧ -1.reserved = ;<br>	∧ -1.roles not in {STRING}<br>	∧ +1.roles in {BINARY, EXPRESSION} and not in {STRING}<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ⏎⏎<br>Confidence: 0.830. Support: 144.` |
| 169 | `  •••start_col ≤ 6<br>	∧ -1.reserved = ;<br>	∧ -1.roles not in {STRING}<br>	∧ +1.roles in {EXPRESSION} and not in {STRING}<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ⏎⏎<br>Confidence: 0.802. Support: 774.` |
| 170 | `  -1.reserved = {<br>	∧ -1.roles not in {STRING}<br>	∧ +1.roles in {EXPRESSION} and not in {STRING}<br>	∧ +3.roles not in {VALUE}<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ⏎⇥⁺<br>Confidence: 0.893. Support: 1927.` |
| 171 | `  -1.reserved not in {;, {}<br>	∧ -1.roles in {COMMENT} and not in {STRING}<br>	∧ +1.roles in {EXPRESSION} and not in {STRING}<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ⏎<br>Confidence: 0.969. Support: 663.` |
| 172 | `  -1.reserved not in {;, {}<br>	∧ -1.roles not in {COMMENT, STRING}<br>	∧ +1.roles in {EXPRESSION} and not in {KEY, STRING}<br>	∧ +2.reserved not in {=}<br>	∧ ^1.internal_type = VariableDeclaration<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ␣<br>Confidence: 0.955. Support: 345.` |
| 173 | `  -1.reserved not in {;, {}<br>	∧ -1.roles not in {COMMENT, STRING}<br>	∧ -3.diff_offset ≥ 4<br>	∧ -5.diff_offset ≥ 7<br>	∧ +1.roles in {EXPRESSION} and not in {KEY, STRING}<br>	∧ ^1.internal_type not in {VariableDeclaration}<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ␣<br>Confidence: 0.965. Support: 14378.` |
| 174 | `  -1.reserved = ,<br>	∧ -1.roles not in {COMMENT, STRING}<br>	∧ -3.diff_offset ≤ 3<br>	∧ -5.diff_offset ≥ 7<br>	∧ +1.roles in {EXPRESSION} and not in {KEY, STRING}<br>	∧ ^1.internal_type not in {VariableDeclaration}<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ␣<br>Confidence: 0.956. Support: 575.` |
| 175 | `  -1.reserved not in {,, ;, {}<br>	∧ -1.roles not in {COMMENT, STRING}<br>	∧ -3.diff_offset ≤ 3<br>	∧ -5.diff_offset ≥ 7<br>	∧ +1.roles in {EXPRESSION} and not in {KEY, STRING}<br>	∧ +3.roles in {EXPRESSION}<br>	∧ ^1.internal_type not in {VariableDeclaration}<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ␣<br>Confidence: 0.809. Support: 363.` |
| 176 | `  -1.reserved not in {;, {}<br>	∧ -1.roles not in {COMMENT, STRING}<br>	∧ -3.reserved = (<br>	∧ -5.diff_offset ≤ 6<br>	∧ +1.roles in {EXPRESSION} and not in {KEY, STRING}<br>	∧ ^1.internal_type not in {VariableDeclaration}<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ∅<br>Confidence: 0.925. Support: 234.` |
| 177 | `  •••start_col ≥ 4<br>	∧ -1.reserved not in {;, {}<br>	∧ -1.roles not in {COMMENT, STRING}<br>	∧ -3.reserved not in {(}<br>	∧ -5.diff_offset ≤ 6<br>	∧ +1.roles in {EXPRESSION} and not in {KEY, STRING}<br>	∧ ^1.internal_type not in {VariableDeclaration}<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ␣<br>Confidence: 0.871. Support: 787.` |
| 178 | `  -1.roles not in {STRING}<br>	∧ +1.roles not in {EXPRESSION, STRING}<br>	∧ +2.roles in {ARGUMENT}<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ∅<br>Confidence: 0.994. Support: 11605.` |
| 179 | `  -1.roles not in {STRING}<br>	∧ +1.reserved = ;<br>	∧ +1.roles not in {EXPRESSION, STRING}<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ∅<br>Confidence: 0.989. Support: 8002.` |
| 180 | `  •••start_col ≥ 36<br>	∧ -1.roles not in {STRING}<br>	∧ -2.label in {<space>}<br>	∧ +1.reserved = }<br>	∧ +1.roles not in {EXPRESSION, STRING}<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ␣<br>Confidence: 0.866. Support: 198.` |
| 181 | `  -1.roles not in {STRING}<br>	∧ -2.label not in {<space>}<br>	∧ +1.reserved = }<br>	∧ +1.roles not in {EXPRESSION, STRING}<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ⏎⇥⁻<br>Confidence: 0.907. Support: 2777.` |
| 182 | `  -1.roles not in {STRING}<br>	∧ +1.reserved = ,<br>	∧ +1.roles not in {EXPRESSION, STRING}<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ∅<br>Confidence: 1.000. Support: 3298.` |
| 183 | `  -1.reserved = if<br>	∧ -1.roles not in {STRING}<br>	∧ +1.reserved = (<br>	∧ +1.roles not in {EXPRESSION, STRING}<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ␣<br>Confidence: 0.999. Support: 568.` |
| 184 | `  -1.reserved not in {if}<br>	∧ -1.roles in {IDENTIFIER} and not in {STRING}<br>	∧ +1.reserved = (<br>	∧ +1.roles not in {EXPRESSION, STRING}<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ∅<br>Confidence: 1.000. Support: 4067.` |
| 185 | `  -1.reserved not in {if}<br>	∧ -1.roles not in {IDENTIFIER, STRING}<br>	∧ +1.reserved = (<br>	∧ +1.roles not in {EXPRESSION, STRING}<br>	∧ +2.reserved = )<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ∅<br>Confidence: 0.999. Support: 537.` |
| 186 | `  -1.roles not in {STRING}<br>	∧ +1.reserved not in {,, ;, }}<br>	∧ +1.roles in {COMMENT} and not in {EXPRESSION, STRING}<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ∅<br>Confidence: 0.992. Support: 1467.` |
| 187 | `  -1.roles in {KEY} and not in {STRING}<br>	∧ +1.reserved not in {,, ;, }}<br>	∧ +1.roles not in {COMMENT, EXPRESSION, STRING}<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ∅<br>Confidence: 1.000. Support: 1312.` |
| 188 | `  -1.reserved = ;<br>	∧ -1.roles not in {KEY, STRING}<br>	∧ +1.reserved not in {(, ,, ;, return, }}<br>	∧ +1.roles not in {EXPRESSION, STRING}<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ +2.length ≥ 3<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ⏎⏎<br>Confidence: 0.801. Support: 959.` |
| 189 | `  -1.reserved = ;<br>	∧ -1.roles not in {KEY, STRING}<br>	∧ +1.reserved not in {(, ,, ;, }}<br>	∧ +1.roles not in {COMMENT, EXPRESSION, STRING}<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ +2.length ≤ 2<br>	∧ ^1.roles not in {QUALIFIED}<br>	∧ ^2.roles not in {DECLARATION}<br>⇒ y = ⏎<br>Confidence: 0.825. Support: 191.` |
| 190 | `  -1.reserved = (<br>	∧ -1.roles not in {KEY, STRING}<br>	∧ +1.reserved = )<br>	∧ +1.roles not in {EXPRESSION, STRING}<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ∅<br>Confidence: 1.000. Support: 1743.` |
| 191 | `  -1.reserved = (<br>	∧ -1.roles not in {KEY, STRING}<br>	∧ +1.reserved not in {(, ), ,, }}<br>	∧ +1.roles not in {COMMENT, EXPRESSION, STRING}<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ␣<br>Confidence: 0.973. Support: 962.` |
| 192 | `  -1.internal_type = CommentLine<br>	∧ -1.reserved not in {(, ;}<br>	∧ -1.roles not in {KEY, STRING}<br>	∧ +1.reserved not in {(, ,, ;, }}<br>	∧ +1.roles not in {COMMENT, EXPRESSION, STRING}<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ⏎<br>Confidence: 0.981. Support: 825.` |
| 193 | `  -1.internal_type not in {CommentLine}<br>	∧ -1.reserved = {<br>	∧ -1.roles not in {KEY, STRING}<br>	∧ +1.reserved not in {,, }}<br>	∧ +1.roles not in {COMMENT, EXPRESSION, STRING}<br>	∧ +2.internal_type = CommentLine<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ⏎⏎⇥⁺<br>Confidence: 0.916. Support: 173.` |
| 194 | `  -1.internal_type not in {CommentLine}<br>	∧ -1.reserved = {<br>	∧ -1.roles not in {KEY, STRING}<br>	∧ +1.reserved not in {,, }}<br>	∧ +1.roles not in {COMMENT, EXPRESSION, STRING}<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ⏎⇥⁺<br>Confidence: 0.934. Support: 705.` |
| 195 | `  -1.internal_type not in {CommentLine}<br>	∧ -1.reserved not in {(, ;, {}<br>	∧ -1.roles not in {KEY, STRING}<br>	∧ +1.reserved not in {(, ,, ;, }}<br>	∧ +1.roles not in {COMMENT, EXPRESSION, STRING}<br>	∧ +2.roles in {COMMENT} and not in {ARGUMENT}<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ⏎⏎<br>Confidence: 0.825. Support: 295.` |
| 196 | `  -1.internal_type not in {CommentLine}<br>	∧ -1.reserved not in {(, ;, {}<br>	∧ -1.roles not in {KEY, STRING}<br>	∧ +1.reserved not in {(, ,, ;, }}<br>	∧ +1.roles not in {COMMENT, EXPRESSION, STRING}<br>	∧ +1.length ≥ 2<br>	∧ +2.roles not in {ARGUMENT, COMMENT}<br>	∧ ^1.roles not in {QUALIFIED, SCOPE}<br>⇒ y = ␣<br>Confidence: 0.922. Support: 2901.` |
| 197 | `  -1.internal_type not in {CommentLine}<br>	∧ -1.reserved not in {(, ;, {}<br>	∧ -1.roles not in {KEY, STRING}<br>	∧ +1.reserved not in {(, ,, ;, }}<br>	∧ +1.roles not in {COMMENT, EXPRESSION, STRING}<br>	∧ +1.length ≤ 1<br>	∧ +2.roles not in {ARGUMENT, COMMENT}<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ␣<br>Confidence: 0.970. Support: 13832.` |
| 198 | `  -2.label not in {<newline>}<br>	∧ -4.label in {"}<br>	∧ +1.internal_type = StringLiteral<br>	∧ +1.length ≤ 35<br>⇒ y = ⏎<br>Confidence: 0.908. Support: 438.` |
| 199 | `  -1.reserved not in {[}<br>	∧ -1.roles not in {STRING}<br>	∧ -2.internal_type = Identifier<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = )<br>	∧ ^1.internal_type = MemberExpression<br>⇒ y = ∅<br>Confidence: 0.999. Support: 390.` |
| 200 | `  -1.reserved not in {[}<br>	∧ -1.roles not in {STRING}<br>	∧ -2.internal_type not in {Identifier}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = )<br>	∧ ^1.internal_type = MemberExpression<br>⇒ y = ␣<br>Confidence: 0.988. Support: 729.` |
| 201 | `  -1.reserved not in {), [}<br>	∧ -1.roles not in {STRING}<br>	∧ -2.diff_offset ≥ 3<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {), ]}<br>	∧ +2.reserved = )<br>	∧ ^1.internal_type = MemberExpression<br>⇒ y = ∅<br>Confidence: 0.949. Support: 672.` |
| 202 | `  -1.internal_type = CommentLine<br>	∧ -1.reserved not in {;, {}<br>	∧ -1.roles not in {STRING}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.roles in {EXPRESSION}<br>	∧ ^1.internal_type not in {MemberExpression}<br>⇒ y = ⏎<br>Confidence: 0.974. Support: 640.` |
| 203 | `  -1.internal_type not in {CommentLine}<br>	∧ -1.reserved not in {;, {}<br>	∧ -1.roles not in {STRING}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.roles in {EXPRESSION} and not in {KEY}<br>	∧ +2.reserved not in {=}<br>	∧ ^1.internal_type = VariableDeclaration<br>⇒ y = ␣<br>Confidence: 0.960. Support: 313.` |
| 204 | `  -1.internal_type not in {CommentLine}<br>	∧ -1.reserved not in {;, {}<br>	∧ -1.roles not in {STRING}<br>	∧ -3.diff_offset ≥ 4<br>	∧ -5.diff_offset ≥ 7<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.roles in {EXPRESSION} and not in {KEY}<br>	∧ ^1.internal_type not in {MemberExpression, VariableDeclaration}<br>⇒ y = ␣<br>Confidence: 0.967. Support: 14208.` |
| 205 | `  -1.internal_type not in {CommentLine}<br>	∧ -1.reserved = ,<br>	∧ -1.roles not in {STRING}<br>	∧ -3.diff_offset ≤ 3<br>	∧ -5.diff_offset ≥ 7<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.roles in {EXPRESSION} and not in {KEY}<br>	∧ ^1.internal_type not in {MemberExpression, VariableDeclaration}<br>⇒ y = ␣<br>Confidence: 0.943. Support: 569.` |
| 206 | `  -1.internal_type not in {CommentLine}<br>	∧ -1.reserved not in {,, ;, {}<br>	∧ -1.roles not in {STRING}<br>	∧ -3.diff_offset ≤ 3<br>	∧ -5.diff_offset ≥ 7<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.roles in {EXPRESSION} and not in {KEY}<br>	∧ ^1.internal_type not in {MemberExpression, VariableDeclaration}<br>⇒ y = ␣<br>Confidence: 0.861. Support: 320.` |
| 207 | `  -1.internal_type not in {CommentLine}<br>	∧ -1.reserved not in {;, {}<br>	∧ -1.roles not in {STRING}<br>	∧ -3.reserved = (<br>	∧ -5.diff_offset ≤ 6<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.roles in {EXPRESSION} and not in {KEY}<br>	∧ ^1.internal_type not in {MemberExpression, VariableDeclaration}<br>⇒ y = ∅<br>Confidence: 0.964. Support: 237.` |
| 208 | `  •••start_col ≥ 4<br>	∧ -1.internal_type not in {CommentLine}<br>	∧ -1.reserved not in {;, {}<br>	∧ -1.roles not in {STRING}<br>	∧ -3.reserved not in {(}<br>	∧ -5.diff_offset ≤ 6<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.roles in {EXPRESSION} and not in {KEY}<br>	∧ ^1.internal_type not in {MemberExpression, VariableDeclaration}<br>⇒ y = ␣<br>Confidence: 0.830. Support: 843.` |
| 209 | `  •••start_col ≥ 42<br>	∧ -1.roles in {EXPRESSION} and not in {STRING}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = }<br>	∧ +1.roles not in {EXPRESSION}<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.internal_type not in {MemberExpression}<br>⇒ y = ␣<br>Confidence: 0.983. Support: 146.` |
| 210 | `  -1.reserved not in {if}<br>	∧ -1.roles in {IDENTIFIER} and not in {STRING}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = (<br>	∧ +1.roles not in {EXPRESSION}<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.internal_type not in {MemberExpression}<br>⇒ y = ∅<br>Confidence: 1.000. Support: 4085.` |
| 211 | `  -1.reserved not in {if}<br>	∧ -1.roles not in {IDENTIFIER, STRING}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = (<br>	∧ +1.roles not in {EXPRESSION}<br>	∧ +2.reserved = )<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.internal_type not in {MemberExpression}<br>⇒ y = ∅<br>Confidence: 0.999. Support: 592.` |
| 212 | `  -1.reserved = ;<br>	∧ -1.roles not in {KEY, STRING}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {(, ,, ;, }}<br>	∧ +1.roles not in {COMMENT, EXPRESSION}<br>	∧ +1.length ≤ 4<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ +2.length ≥ 2<br>	∧ ^1.internal_type not in {MemberExpression}<br>⇒ y = ⏎⏎<br>Confidence: 0.814. Support: 856.` |
| 213 | `  -1.internal_type not in {CommentLine}<br>	∧ -1.reserved = {<br>	∧ -1.roles not in {KEY, STRING}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {,, }}<br>	∧ +1.roles not in {COMMENT, EXPRESSION}<br>	∧ +2.internal_type = CommentLine<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.internal_type not in {MemberExpression}<br>⇒ y = ⏎⏎⇥⁺<br>Confidence: 0.934. Support: 175.` |
| 214 | `  -1.internal_type not in {CommentLine}<br>	∧ -1.reserved not in {(, ;, {}<br>	∧ -1.roles not in {KEY, STRING}<br>	∧ -2.diff_offset ≤ 19<br>	∧ -3.label in {<-tab>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {(, ,, ;, }}<br>	∧ +1.roles not in {COMMENT, EXPRESSION}<br>	∧ +2.roles not in {ARGUMENT, COMMENT, EXPRESSION}<br>	∧ ^1.internal_type not in {ArrayExpression, MemberExpression}<br>⇒ y = ␣<br>Confidence: 0.870. Support: 971.` |
| 215 | `  -1.internal_type not in {CommentLine}<br>	∧ -1.reserved not in {(, ;, {}<br>	∧ -1.roles not in {KEY, STRING}<br>	∧ -2.diff_offset ≤ 19<br>	∧ -3.label in {<space>} and not in {<-tab>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {(, ,, ;, }}<br>	∧ +1.roles not in {COMMENT, EXPRESSION}<br>	∧ +2.roles not in {ARGUMENT, COMMENT}<br>	∧ ^1.internal_type not in {ArrayExpression, MemberExpression}<br>⇒ y = ␣<br>Confidence: 0.808. Support: 627.` |
| 216 | `  -1.internal_type not in {CommentLine}<br>	∧ -1.reserved not in {(, ;, {}<br>	∧ -1.roles not in {KEY, STRING}<br>	∧ -2.diff_offset ≤ 19<br>	∧ -3.label not in {<-tab>, <space>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {(, ,, ;, }}<br>	∧ +1.roles not in {COMMENT, EXPRESSION}<br>	∧ +1.length ≥ 2<br>	∧ +2.roles not in {ARGUMENT, COMMENT}<br>	∧ ^1.internal_type not in {ArrayExpression, MemberExpression}<br>	∧ ^1.roles not in {STATEMENT}<br>⇒ y = ␣<br>Confidence: 0.955. Support: 2258.` |
| 217 | `  -1.internal_type not in {CommentLine}<br>	∧ -1.reserved not in {(, ;, {}<br>	∧ -1.roles not in {KEY, STRING}<br>	∧ -2.diff_offset ≤ 19<br>	∧ -3.label not in {<-tab>, <space>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {(, ,, ;, }}<br>	∧ +1.roles not in {COMMENT, EXPRESSION}<br>	∧ +1.length ≤ 1<br>	∧ +2.roles not in {ARGUMENT, COMMENT}<br>	∧ ^1.internal_type not in {ArrayExpression, MemberExpression}<br>⇒ y = ␣<br>Confidence: 0.988. Support: 12509.` |
| 218 | `  -1.reserved = )<br>	∧ -1.roles not in {STRING}<br>	∧ +1.reserved not in {), ]}<br>	∧ +1.roles not in {STRING}<br>	∧ ^1.roles in {QUALIFIED}<br>⇒ y = ∅<br>Confidence: 0.888. Support: 2290.` |
| 219 | `  -1.reserved not in {), [}<br>	∧ -1.roles not in {STRING}<br>	∧ -2.diff_col ≥ 4<br>	∧ +1.reserved not in {), ]}<br>	∧ +1.roles not in {STRING}<br>	∧ +2.reserved = )<br>	∧ ^1.roles in {QUALIFIED}<br>⇒ y = ∅<br>Confidence: 0.978. Support: 716.` |
| 220 | `  -1.reserved not in {), [}<br>	∧ -1.roles not in {STRING}<br>	∧ +1.reserved not in {), ]}<br>	∧ +1.roles not in {STRING}<br>	∧ +2.reserved not in {)}<br>	∧ ^1.roles in {QUALIFIED}<br>⇒ y = ∅<br>Confidence: 0.999. Support: 20512.` |
| 221 | `  -1.roles not in {STRING}<br>	∧ +1.roles not in {STRING}<br>	∧ +2.roles in {ARGUMENT}<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ∅<br>Confidence: 0.995. Support: 11644.` |
| 222 | `  -1.roles not in {STRING}<br>	∧ +1.reserved = ;<br>	∧ +1.roles not in {STRING}<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ∅<br>Confidence: 0.988. Support: 7978.` |
| 223 | `  -1.reserved = ;<br>	∧ -1.roles not in {STRING}<br>	∧ +1.reserved = }<br>	∧ +1.roles not in {STRING}<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ⏎⇥⁻<br>Confidence: 0.909. Support: 2258.` |
| 224 | `  -1.reserved = ;<br>	∧ -1.roles not in {STRING}<br>	∧ +1.reserved not in {;, }}<br>	∧ +1.roles not in {STRING}<br>	∧ +1.length ≤ 5<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ +2.length ≥ 3<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ⏎⏎<br>Confidence: 0.814. Support: 865.` |
| 225 | `  -1.reserved = ;<br>	∧ -1.roles not in {STRING}<br>	∧ +1.reserved not in {;, }}<br>	∧ +1.roles not in {STRING}<br>	∧ +1.length ≤ 5<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ +2.length ≤ 2<br>	∧ +5.length ≥ 13<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ⏎⏎<br>Confidence: 0.894. Support: 437.` |
| 226 | `  •••start_col ≥ 10<br>	∧ -1.reserved = ;<br>	∧ -1.roles not in {STRING}<br>	∧ +1.reserved not in {;, }}<br>	∧ +1.roles not in {STRING}<br>	∧ +1.length ≤ 5<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ +2.length ≤ 2<br>	∧ +5.length ≤ 12<br>	∧ ^1.roles not in {QUALIFIED}<br>	∧ ^2.roles not in {FUNCTION}<br>⇒ y = ⏎<br>Confidence: 0.810. Support: 155.` |
| 227 | `  -1.reserved = if<br>	∧ -1.roles not in {STRING}<br>	∧ +1.reserved = (<br>	∧ +1.roles not in {STRING}<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ␣<br>Confidence: 0.999. Support: 619.` |
| 228 | `  -1.reserved not in {;, if}<br>	∧ -1.roles in {EXPRESSION} and not in {STRING}<br>	∧ +1.reserved = (<br>	∧ +1.roles not in {STRING}<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ∅<br>Confidence: 1.000. Support: 4096.` |
| 229 | `  -1.reserved not in {;, if}<br>	∧ -1.roles not in {EXPRESSION, STRING}<br>	∧ +1.reserved = (<br>	∧ +1.roles not in {STRING}<br>	∧ +2.reserved = )<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ∅<br>Confidence: 0.999. Support: 535.` |
| 230 | `  -1.roles not in {STRING}<br>	∧ +1.reserved = ,<br>	∧ +1.roles not in {STRING}<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ∅<br>Confidence: 1.000. Support: 3345.` |
| 231 | `  -1.reserved = {<br>	∧ -1.roles not in {STRING}<br>	∧ +1.reserved not in {(, ,}<br>	∧ +1.roles not in {STRING}<br>	∧ +2.roles in {COMMENT} and not in {ARGUMENT}<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ⏎⏎⇥⁺<br>Confidence: 0.916. Support: 185.` |
| 232 | `  -1.reserved = {<br>	∧ -1.roles not in {STRING}<br>	∧ +1.reserved not in {(, ,}<br>	∧ +1.roles not in {STRING}<br>	∧ +2.roles not in {ARGUMENT, COMMENT}<br>	∧ +3.roles not in {VALUE}<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ⏎⇥⁺<br>Confidence: 0.890. Support: 2652.` |
| 233 | `  -1.internal_type = CommentLine<br>	∧ -1.reserved not in {;, {}<br>	∧ -1.roles not in {STRING}<br>	∧ +1.reserved not in {(, ,, ;}<br>	∧ +1.roles not in {STRING}<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ⏎<br>Confidence: 0.973. Support: 1451.` |
| 234 | `  -1.internal_type not in {CommentLine}<br>	∧ -1.reserved not in {;, {}<br>	∧ -1.roles not in {STRING}<br>	∧ +1.reserved not in {,, ;}<br>	∧ +1.roles in {COMMENT} and not in {STRING}<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ∅<br>Confidence: 1.000. Support: 1470.` |
| 235 | `  -1.internal_type not in {CommentLine}<br>	∧ -1.reserved not in {;, {}<br>	∧ -1.roles in {KEY} and not in {STRING}<br>	∧ +1.reserved not in {,, ;}<br>	∧ +1.roles not in {STRING}<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ∅<br>Confidence: 1.000. Support: 1374.` |
| 236 | `  •••start_col ≥ 27<br>	∧ -1.internal_type not in {CommentLine}<br>	∧ -1.reserved not in {{}<br>	∧ -1.roles not in {KEY, STRING}<br>	∧ +1.reserved = }<br>	∧ +1.roles not in {COMMENT, KEY, STRING}<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ␣<br>Confidence: 0.803. Support: 276.` |
| 237 | `  •••start_col ≤ 26<br>	∧ -1.internal_type not in {CommentLine}<br>	∧ -1.reserved not in {{}<br>	∧ -1.roles not in {KEY, STRING}<br>	∧ +1.reserved = }<br>	∧ +1.roles not in {COMMENT, KEY, STRING}<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ⏎⇥⁻<br>Confidence: 0.929. Support: 597.` |
| 238 | `  -1.internal_type not in {CommentLine}<br>	∧ -1.reserved not in {;, {}<br>	∧ -1.roles not in {KEY, STRING}<br>	∧ +1.reserved not in {(, ,, ;, }}<br>	∧ +1.roles not in {COMMENT, KEY, STRING}<br>	∧ +2.roles in {COMMENT} and not in {ARGUMENT}<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ⏎⏎<br>Confidence: 0.831. Support: 316.` |
| 239 | `  -1.internal_type not in {CommentLine}<br>	∧ -1.reserved not in {;, {}<br>	∧ -1.roles not in {KEY, STRING}<br>	∧ +1.reserved not in {(, ,, ;, }}<br>	∧ +1.roles not in {COMMENT, KEY, STRING}<br>	∧ +2.reserved not in {=}<br>	∧ +2.roles not in {ARGUMENT, COMMENT}<br>	∧ ^1.internal_type = VariableDeclaration<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ␣<br>Confidence: 0.806. Support: 528.` |
| 240 | `  -1.internal_type not in {CommentLine}<br>	∧ -1.reserved = (<br>	∧ -1.roles not in {KEY, STRING}<br>	∧ +1.reserved = )<br>	∧ +1.roles not in {KEY, STRING}<br>	∧ +2.roles not in {COMMENT}<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ∅<br>Confidence: 0.998. Support: 1641.` |
| 241 | `  -1.internal_type not in {CommentLine}<br>	∧ -1.reserved not in {{}<br>	∧ -1.roles not in {KEY, STRING}<br>	∧ +1.reserved = )<br>	∧ +1.roles not in {COMMENT, KEY, STRING}<br>	∧ +2.roles not in {ARGUMENT, COMMENT}<br>	∧ ^1.internal_type not in {VariableDeclaration}<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ␣<br>Confidence: 0.983. Support: 5705.` |
| 242 | `  -1.internal_type not in {CommentLine}<br>	∧ -1.reserved = }<br>	∧ -1.roles not in {KEY, STRING}<br>	∧ +1.reserved not in {(, ), ,, }}<br>	∧ +1.roles not in {COMMENT, KEY, STRING}<br>	∧ +2.roles not in {ARGUMENT, COMMENT}<br>	∧ ^1.internal_type not in {VariableDeclaration}<br>	∧ ^1.roles not in {BLOCK, QUALIFIED}<br>⇒ y = ␣<br>Confidence: 0.902. Support: 200.` |
| 243 | `  -1.internal_type not in {CommentLine}<br>	∧ -1.reserved not in {;, {, }}<br>	∧ -1.roles not in {KEY, STRING}<br>	∧ -3.diff_offset ≥ 4<br>	∧ +1.reserved not in {(, ), ,, ;, }}<br>	∧ +1.roles not in {COMMENT, KEY, STRING}<br>	∧ +2.roles not in {ARGUMENT, COMMENT}<br>	∧ ^1.internal_type not in {VariableDeclaration}<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ␣<br>Confidence: 0.966. Support: 24510.` |
| 244 | `  •••start_col ≥ 17<br>	∧ -1.internal_type not in {CommentLine}<br>	∧ -1.reserved not in {;, {, }}<br>	∧ -1.roles not in {KEY, STRING}<br>	∧ -3.diff_offset ≤ 3<br>	∧ +1.reserved not in {(, ), ,, ;, }}<br>	∧ +1.roles not in {COMMENT, KEY, STRING}<br>	∧ +2.roles not in {ARGUMENT, COMMENT}<br>	∧ +4.roles not in {LITERAL}<br>	∧ ^1.internal_type not in {VariableDeclaration}<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ␣<br>Confidence: 0.891. Support: 2448.` |
| 245 | `  -1.internal_type = StringLiteral<br>	∧ +1.reserved = )<br>	∧ +1.roles not in {STRING}<br>⇒ y = "␣<br>Confidence: 0.981. Support: 4920.` |
| 246 | `  -1.internal_type = StringLiteral<br>	∧ +1.reserved not in {)}<br>	∧ +1.roles not in {STRING}<br>	∧ ^1.roles in {OPERATOR}<br>⇒ y = "␣<br>Confidence: 0.999. Support: 614.` |
| 247 | `  -1.internal_type = StringLiteral<br>	∧ +1.reserved not in {), ], }}<br>	∧ +1.roles not in {STRING}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = "<br>Confidence: 0.984. Support: 4156.` |
| 248 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = [<br>	∧ +1.roles not in {STRING}<br>	∧ ^1.roles in {IDENTIFIER}<br>⇒ y = ␣<br>Confidence: 0.996. Support: 695.` |
| 249 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {[}<br>	∧ -2.roles in {IDENTIFIER}<br>	∧ +1.reserved = )<br>	∧ +1.roles not in {STRING}<br>	∧ ^1.roles in {IDENTIFIER}<br>⇒ y = ∅<br>Confidence: 0.999. Support: 406.` |
| 250 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {[}<br>	∧ -2.roles not in {IDENTIFIER}<br>	∧ +1.reserved = )<br>	∧ +1.roles not in {STRING}<br>	∧ ^1.roles in {IDENTIFIER}<br>⇒ y = ␣<br>Confidence: 0.987. Support: 750.` |
| 251 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {[}<br>	∧ +1.reserved = ]<br>	∧ +1.roles not in {STRING}<br>	∧ ^1.roles in {IDENTIFIER}<br>⇒ y = ␣<br>Confidence: 0.998. Support: 228.` |
| 252 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = )<br>	∧ +1.reserved not in {), ]}<br>	∧ +1.roles not in {STRING}<br>	∧ ^1.roles in {IDENTIFIER}<br>⇒ y = ∅<br>Confidence: 0.884. Support: 2194.` |
| 253 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {), [}<br>	∧ -2.diff_offset ≥ 4<br>	∧ +1.reserved not in {), ]}<br>	∧ +1.roles not in {STRING}<br>	∧ +2.reserved = )<br>	∧ ^1.roles in {IDENTIFIER}<br>⇒ y = ∅<br>Confidence: 0.954. Support: 658.` |
| 254 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {), [}<br>	∧ +1.reserved not in {), ]}<br>	∧ +1.roles not in {STRING}<br>	∧ +2.reserved not in {)}<br>	∧ ^1.roles in {IDENTIFIER}<br>⇒ y = ∅<br>Confidence: 0.999. Support: 20503.` |
| 255 | `  •••start_col ≥ 7<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = ;<br>	∧ -4.length ≥ 14<br>	∧ +1.roles in {EXPRESSION, LEFT} and not in {STRING}<br>	∧ ^1.roles not in {IDENTIFIER}<br>⇒ y = ⏎⏎<br>Confidence: 0.814. Support: 153.` |
| 256 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = {<br>	∧ +1.roles in {EXPRESSION} and not in {STRING}<br>	∧ +3.roles not in {VALUE}<br>	∧ ^1.roles not in {IDENTIFIER}<br>⇒ y = ⏎⇥⁺<br>Confidence: 0.877. Support: 1966.` |
| 257 | `  -1.internal_type = CommentLine<br>	∧ -1.reserved not in {;, {}<br>	∧ +1.roles in {EXPRESSION} and not in {STRING}<br>	∧ ^1.roles not in {IDENTIFIER}<br>⇒ y = ⏎<br>Confidence: 0.982. Support: 624.` |
| 258 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {;, {}<br>	∧ +1.roles in {EXPRESSION} and not in {KEY, STRING}<br>	∧ +2.reserved not in {=}<br>	∧ ^1.internal_type = VariableDeclaration<br>	∧ ^1.roles not in {IDENTIFIER}<br>⇒ y = ␣<br>Confidence: 0.960. Support: 316.` |
| 259 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {;, {}<br>	∧ -3.diff_offset ≥ 4<br>	∧ -5.diff_offset ≥ 7<br>	∧ +1.roles in {EXPRESSION} and not in {KEY, STRING}<br>	∧ ^1.internal_type not in {VariableDeclaration}<br>	∧ ^1.roles not in {IDENTIFIER}<br>⇒ y = ␣<br>Confidence: 0.966. Support: 14324.` |
| 260 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved = ,<br>	∧ -3.diff_offset ≤ 3<br>	∧ -5.diff_offset ≥ 7<br>	∧ +1.roles in {EXPRESSION} and not in {KEY, STRING}<br>	∧ ^1.internal_type not in {VariableDeclaration}<br>	∧ ^1.roles not in {IDENTIFIER}<br>⇒ y = ␣<br>Confidence: 0.955. Support: 590.` |
| 261 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {,, ;, {}<br>	∧ -3.diff_offset ≤ 3<br>	∧ -5.diff_offset ≥ 7<br>	∧ +1.roles in {EXPRESSION} and not in {KEY, LITERAL}<br>	∧ ^1.internal_type not in {VariableDeclaration}<br>	∧ ^1.roles not in {IDENTIFIER}<br>⇒ y = ␣<br>Confidence: 0.823. Support: 348.` |
| 262 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {;, {}<br>	∧ -3.reserved = (<br>	∧ -5.diff_offset ≤ 6<br>	∧ +1.roles in {EXPRESSION} and not in {KEY, STRING}<br>	∧ ^1.internal_type not in {VariableDeclaration}<br>	∧ ^1.roles not in {IDENTIFIER}<br>⇒ y = ∅<br>Confidence: 0.924. Support: 229.` |
| 263 | `  •••start_col ≥ 4<br>	∧ -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {;, {}<br>	∧ -3.reserved not in {(}<br>	∧ -5.diff_offset ≤ 6<br>	∧ +1.roles in {EXPRESSION} and not in {KEY, STRING}<br>	∧ ^1.internal_type not in {VariableDeclaration}<br>	∧ ^1.roles not in {IDENTIFIER}<br>⇒ y = ␣<br>Confidence: 0.831. Support: 783.` |
| 264 | `  -1.internal_type not in {StringLiteral}<br>	∧ +1.roles not in {EXPRESSION, STRING}<br>	∧ +2.roles in {ARGUMENT}<br>	∧ ^1.roles not in {IDENTIFIER}<br>⇒ y = ∅<br>Confidence: 0.994. Support: 11769.` |
| 265 | `  -1.internal_type not in {StringLiteral}<br>	∧ -2.diff_offset ≤ 12<br>	∧ +1.reserved = ;<br>	∧ +1.roles not in {EXPRESSION, STRING}<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.roles not in {IDENTIFIER}<br>⇒ y = ∅<br>Confidence: 0.998. Support: 7850.` |
| 266 | `  -1.internal_type = NumericLiteral<br>	∧ +1.reserved = }<br>	∧ +1.roles not in {EXPRESSION, STRING}<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.roles not in {IDENTIFIER}<br>⇒ y = ␣<br>Confidence: 0.873. Support: 153.` |
| 267 | `  -1.internal_type not in {NumericLiteral, StringLiteral}<br>	∧ +1.reserved = }<br>	∧ +1.roles not in {EXPRESSION, STRING}<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.roles not in {IDENTIFIER}<br>⇒ y = ⏎⇥⁻<br>Confidence: 0.889. Support: 2905.` |
| 268 | `  -1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = ,<br>	∧ +1.roles not in {EXPRESSION, STRING}<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.roles not in {IDENTIFIER}<br>⇒ y = ∅<br>Confidence: 1.000. Support: 3359.` |
| 269 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = if<br>	∧ +1.reserved = (<br>	∧ +1.roles not in {EXPRESSION, STRING}<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.roles not in {IDENTIFIER}<br>⇒ y = ␣<br>Confidence: 0.999. Support: 530.` |
| 270 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {if}<br>	∧ -1.roles in {EXPRESSION}<br>	∧ +1.reserved = (<br>	∧ +1.roles not in {EXPRESSION, STRING}<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.roles not in {IDENTIFIER}<br>⇒ y = ∅<br>Confidence: 1.000. Support: 4161.` |
| 271 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {if}<br>	∧ -1.roles not in {EXPRESSION}<br>	∧ +1.reserved = (<br>	∧ +1.roles not in {EXPRESSION, STRING}<br>	∧ +2.reserved = )<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.roles not in {IDENTIFIER}<br>⇒ y = ∅<br>Confidence: 0.999. Support: 532.` |
| 272 | `  -1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {,, ;, }}<br>	∧ +1.roles in {COMMENT} and not in {EXPRESSION, STRING}<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.roles not in {IDENTIFIER}<br>⇒ y = ∅<br>Confidence: 0.988. Support: 1588.` |
| 273 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.roles in {KEY}<br>	∧ +1.reserved not in {,, ;, }}<br>	∧ +1.roles not in {COMMENT, EXPRESSION, STRING}<br>	∧ ^1.roles not in {IDENTIFIER}<br>⇒ y = ∅<br>Confidence: 1.000. Support: 1319.` |
| 274 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = ;<br>	∧ -1.roles not in {KEY}<br>	∧ +1.reserved not in {(, ,, ;, return, }}<br>	∧ +1.roles not in {EXPRESSION, STRING}<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ +2.length ≥ 2<br>	∧ ^1.roles not in {IDENTIFIER}<br>⇒ y = ⏎⏎<br>Confidence: 0.815. Support: 947.` |
| 275 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = ;<br>	∧ -1.roles not in {KEY}<br>	∧ +1.reserved not in {(, ,, ;, }}<br>	∧ +1.roles not in {COMMENT, EXPRESSION, STRING}<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ +2.length ≤ 1<br>	∧ ^1.roles not in {IDENTIFIER}<br>	∧ ^2.roles not in {FUNCTION}<br>⇒ y = ⏎<br>Confidence: 0.822. Support: 177.` |
| 276 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = (<br>	∧ -1.roles not in {KEY}<br>	∧ +1.reserved = )<br>	∧ +1.roles not in {EXPRESSION, STRING}<br>	∧ ^1.roles not in {IDENTIFIER}<br>⇒ y = ∅<br>Confidence: 0.998. Support: 1623.` |
| 277 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = (<br>	∧ -1.roles not in {KEY}<br>	∧ +1.reserved not in {(, ), ,, }}<br>	∧ +1.roles not in {COMMENT, EXPRESSION, STRING}<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.roles not in {IDENTIFIER}<br>⇒ y = ␣<br>Confidence: 0.981. Support: 1013.` |
| 278 | `  -1.internal_type = CommentLine<br>	∧ -1.reserved not in {(, ;}<br>	∧ -1.roles not in {KEY}<br>	∧ +1.reserved not in {(, ,, ;, }}<br>	∧ +1.roles not in {COMMENT, EXPRESSION, STRING}<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.roles not in {IDENTIFIER}<br>⇒ y = ⏎<br>Confidence: 0.980. Support: 788.` |
| 279 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved = {<br>	∧ -1.roles not in {KEY}<br>	∧ +1.reserved not in {,, }}<br>	∧ +1.roles not in {COMMENT, EXPRESSION, STRING}<br>	∧ +2.internal_type = CommentLine<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.roles not in {IDENTIFIER}<br>⇒ y = ⏎⏎⇥⁺<br>Confidence: 0.954. Support: 184.` |
| 280 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved = {<br>	∧ -1.roles not in {KEY}<br>	∧ +1.reserved not in {,, }}<br>	∧ +1.roles not in {COMMENT, EXPRESSION, STRING}<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.roles not in {IDENTIFIER}<br>⇒ y = ⏎⇥⁺<br>Confidence: 0.906. Support: 728.` |
| 281 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {(, ;, {}<br>	∧ -1.roles not in {KEY}<br>	∧ +1.reserved not in {(, ,, ;, }}<br>	∧ +1.roles not in {COMMENT, EXPRESSION, STRING}<br>	∧ +2.roles in {COMMENT} and not in {ARGUMENT}<br>	∧ ^1.roles not in {IDENTIFIER}<br>⇒ y = ⏎⏎<br>Confidence: 0.861. Support: 348.` |
| 282 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {(, ;, {}<br>	∧ -1.roles not in {KEY}<br>	∧ -2.diff_offset ≤ 20<br>	∧ +1.reserved not in {(, ,, ;, }}<br>	∧ +1.roles not in {COMMENT, EXPRESSION, STRING}<br>	∧ +1.length ≥ 2<br>	∧ +2.roles not in {ARGUMENT, COMMENT}<br>	∧ ^1.internal_type not in {BlockStatement}<br>	∧ ^1.roles not in {IDENTIFIER, LIST}<br>⇒ y = ␣<br>Confidence: 0.936. Support: 2785.` |
| 283 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {(, ;, {}<br>	∧ -1.roles not in {KEY}<br>	∧ -2.diff_offset ≤ 20<br>	∧ +1.reserved not in {(, ,, ;, }}<br>	∧ +1.roles not in {COMMENT, EXPRESSION, STRING}<br>	∧ +1.length ≤ 1<br>	∧ +2.roles not in {ARGUMENT, COMMENT}<br>	∧ ^1.roles not in {IDENTIFIER, LIST}<br>⇒ y = ␣<br>Confidence: 0.976. Support: 13577.` |
| 284 | `  -1.roles in {STRING}<br>	∧ -4.diff_line = 0<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = )<br>⇒ y = "␣<br>Confidence: 0.999. Support: 4791.` |
| 285 | `  -1.reserved = [<br>	∧ -1.roles not in {STRING}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ ^1.roles in {IDENTIFIER}<br>⇒ y = ␣<br>Confidence: 0.999. Support: 722.` |
| 286 | `  -1.reserved not in {[}<br>	∧ -1.roles not in {STRING}<br>	∧ -2.internal_type = Identifier<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = )<br>	∧ ^1.roles in {IDENTIFIER}<br>⇒ y = ∅<br>Confidence: 0.999. Support: 405.` |
| 287 | `  -1.reserved not in {[}<br>	∧ -1.roles not in {STRING}<br>	∧ -2.internal_type not in {Identifier}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = )<br>	∧ ^1.roles in {IDENTIFIER}<br>⇒ y = ␣<br>Confidence: 0.990. Support: 764.` |
| 288 | `  -1.reserved not in {[}<br>	∧ -1.roles not in {STRING}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = ]<br>	∧ ^1.roles in {IDENTIFIER}<br>⇒ y = ␣<br>Confidence: 0.998. Support: 227.` |
| 289 | `  -1.reserved not in {[}<br>	∧ -1.roles not in {STRING}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = }<br>	∧ ^1.roles in {IDENTIFIER}<br>⇒ y = ⏎⇥⁻<br>Confidence: 0.801. Support: 143.` |
| 290 | `  -1.reserved not in {[}<br>	∧ -1.roles not in {STRING}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {), ], }}<br>	∧ ^1.roles in {IDENTIFIER}<br>⇒ y = ∅<br>Confidence: 0.988. Support: 23496.` |
| 291 | `  -1.roles not in {STRING}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +2.roles in {ARGUMENT}<br>	∧ ^1.roles not in {IDENTIFIER}<br>⇒ y = ∅<br>Confidence: 0.994. Support: 11711.` |
| 292 | `  -1.roles not in {STRING}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = ;<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.roles not in {IDENTIFIER}<br>⇒ y = ∅<br>Confidence: 0.988. Support: 8031.` |
| 293 | `  -1.reserved = ;<br>	∧ -1.roles not in {STRING}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = }<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.roles not in {IDENTIFIER}<br>⇒ y = ⏎⇥⁻<br>Confidence: 0.912. Support: 2231.` |
| 294 | `  •••start_col ≤ 6<br>	∧ -1.reserved = ;<br>	∧ -1.roles not in {STRING}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {;, }}<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ +2.length ≤ 2<br>	∧ +3.length ≥ 3<br>	∧ ^1.roles not in {IDENTIFIER}<br>⇒ y = ⏎⏎<br>Confidence: 0.824. Support: 741.` |
| 295 | `  •••start_col ≤ 6<br>	∧ -1.reserved = ;<br>	∧ -1.roles not in {STRING}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {;, }}<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ +2.length ≤ 2<br>	∧ +3.length ≤ 2<br>	∧ ^1.roles not in {IDENTIFIER}<br>⇒ y = ⏎<br>Confidence: 0.835. Support: 155.` |
| 296 | `  -1.reserved = if<br>	∧ -1.roles not in {STRING}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = (<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.roles not in {IDENTIFIER}<br>⇒ y = ␣<br>Confidence: 0.999. Support: 630.` |
| 297 | `  -1.internal_type = Identifier<br>	∧ -1.reserved not in {;, if}<br>	∧ -1.roles not in {STRING}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = (<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.roles not in {IDENTIFIER}<br>⇒ y = ∅<br>Confidence: 1.000. Support: 3972.` |
| 298 | `  -1.internal_type not in {Identifier}<br>	∧ -1.reserved not in {;, if}<br>	∧ -1.roles not in {STRING}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = (<br>	∧ +2.reserved = )<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.roles not in {IDENTIFIER}<br>⇒ y = ∅<br>Confidence: 0.999. Support: 552.` |
| 299 | `  -1.roles not in {STRING}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = ,<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.roles not in {IDENTIFIER}<br>⇒ y = ∅<br>Confidence: 1.000. Support: 3253.` |
| 300 | `  -1.reserved = {<br>	∧ -1.roles not in {STRING}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {(, ,}<br>	∧ +2.roles in {COMMENT} and not in {ARGUMENT}<br>	∧ ^1.roles not in {IDENTIFIER}<br>⇒ y = ⏎⏎⇥⁺<br>Confidence: 0.913. Support: 202.` |
| 301 | `  -1.reserved = {<br>	∧ -1.roles not in {STRING}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {(, ,}<br>	∧ +2.roles not in {ARGUMENT, COMMENT}<br>	∧ +3.roles not in {VALUE}<br>	∧ ^1.roles not in {IDENTIFIER}<br>⇒ y = ⏎⇥⁺<br>Confidence: 0.891. Support: 2706.` |
| 302 | `  -1.internal_type = CommentLine<br>	∧ -1.reserved not in {;, {}<br>	∧ -1.roles not in {STRING}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {(, ,, ;}<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.roles not in {IDENTIFIER}<br>⇒ y = ⏎<br>Confidence: 0.969. Support: 1485.` |
| 303 | `  -1.internal_type not in {CommentLine}<br>	∧ -1.reserved not in {;, {}<br>	∧ -1.roles not in {STRING}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {,, ;}<br>	∧ +1.roles in {COMMENT}<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.roles not in {IDENTIFIER}<br>⇒ y = ∅<br>Confidence: 1.000. Support: 1529.` |
| 304 | `  -1.internal_type not in {CommentLine}<br>	∧ -1.reserved not in {;, {}<br>	∧ -1.roles in {KEY} and not in {STRING}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {,, ;}<br>	∧ ^1.roles not in {IDENTIFIER}<br>⇒ y = ∅<br>Confidence: 1.000. Support: 1301.` |
| 305 | `  •••start_col ≥ 29<br>	∧ -1.internal_type not in {CommentLine}<br>	∧ -1.reserved not in {{}<br>	∧ -1.roles not in {KEY, STRING}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = }<br>	∧ +1.roles not in {COMMENT, KEY}<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.roles not in {IDENTIFIER}<br>⇒ y = ␣<br>Confidence: 0.862. Support: 258.` |
| 306 | `  •••start_col ≤ 29<br>	∧ -1.internal_type not in {CommentLine}<br>	∧ -1.reserved not in {{}<br>	∧ -1.roles not in {KEY, STRING}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = }<br>	∧ +1.roles not in {COMMENT, KEY}<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.roles not in {IDENTIFIER}<br>⇒ y = ⏎⇥⁻<br>Confidence: 0.937. Support: 624.` |
| 307 | `  -1.internal_type not in {CommentLine}<br>	∧ -1.reserved not in {;, {}<br>	∧ -1.roles not in {KEY, STRING}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {(, ,, ;, }}<br>	∧ +1.roles not in {COMMENT, KEY}<br>	∧ +2.roles in {COMMENT} and not in {ARGUMENT}<br>	∧ ^1.internal_type not in {VariableDeclaration}<br>	∧ ^1.roles not in {IDENTIFIER}<br>⇒ y = ⏎⏎<br>Confidence: 0.815. Support: 311.` |
| 308 | `  -1.internal_type not in {CommentLine}<br>	∧ -1.reserved = (<br>	∧ -1.roles not in {KEY, STRING}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = )<br>	∧ +1.roles not in {KEY}<br>	∧ +2.roles not in {COMMENT}<br>	∧ ^1.roles not in {IDENTIFIER}<br>⇒ y = ∅<br>Confidence: 1.000. Support: 1613.` |
| 309 | `  -1.internal_type not in {CommentLine}<br>	∧ -1.reserved not in {{}<br>	∧ -1.roles not in {KEY, STRING}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = )<br>	∧ +1.roles not in {COMMENT, KEY}<br>	∧ +2.roles not in {ARGUMENT, COMMENT}<br>	∧ ^1.internal_type not in {VariableDeclaration}<br>	∧ ^1.roles not in {IDENTIFIER}<br>⇒ y = ␣<br>Confidence: 0.986. Support: 5656.` |
| 310 | `  -1.internal_type not in {CommentLine}<br>	∧ -1.reserved = }<br>	∧ -1.roles not in {KEY, STRING}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {(, ), ,, }}<br>	∧ +1.roles not in {COMMENT, KEY}<br>	∧ +2.roles not in {ARGUMENT, COMMENT}<br>	∧ ^1.internal_type not in {VariableDeclaration}<br>	∧ ^1.roles not in {BLOCK, IDENTIFIER}<br>⇒ y = ␣<br>Confidence: 0.899. Support: 223.` |
| 311 | `  -1.internal_type not in {CommentLine}<br>	∧ -1.reserved not in {;, {, }}<br>	∧ -1.roles not in {KEY, STRING}<br>	∧ -3.diff_offset ≥ 4<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {(, ), ,, ;, }}<br>	∧ +1.roles not in {COMMENT, KEY}<br>	∧ +2.roles not in {ARGUMENT, COMMENT}<br>	∧ ^1.internal_type not in {VariableDeclaration}<br>	∧ ^1.roles not in {IDENTIFIER}<br>⇒ y = ␣<br>Confidence: 0.963. Support: 24572.` |
| 312 | `  -1.internal_type not in {CommentLine}<br>	∧ -1.reserved not in {;, {, }}<br>	∧ -1.roles not in {KEY, STRING}<br>	∧ -2.reserved = )<br>	∧ -3.diff_offset ≤ 3<br>	∧ -3.reserved = (<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {(, ), ,, ;, }}<br>	∧ +1.roles not in {COMMENT, KEY}<br>	∧ +2.roles not in {ARGUMENT, COMMENT}<br>	∧ ^1.internal_type not in {VariableDeclaration}<br>	∧ ^1.roles not in {IDENTIFIER}<br>⇒ y = ␣<br>Confidence: 0.955. Support: 254.` |
| 313 | `  •••start_col ≥ 6<br>	∧ -1.internal_type not in {CommentLine}<br>	∧ -1.reserved not in {;, {, }}<br>	∧ -1.roles not in {KEY, STRING}<br>	∧ -3.diff_offset ≤ 3<br>	∧ -3.reserved not in {(}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {(, ), ,, ;, }}<br>	∧ +1.roles not in {COMMENT, KEY}<br>	∧ +2.roles not in {ARGUMENT, COMMENT}<br>	∧ +4.roles not in {MAP}<br>	∧ ^1.internal_type not in {VariableDeclaration}<br>	∧ ^1.roles not in {IDENTIFIER}<br>⇒ y = ␣<br>Confidence: 0.892. Support: 2701.` |
| 314 | `  -1.reserved = [<br>	∧ -1.roles not in {STRING}<br>	∧ +1.roles not in {STRING}<br>	∧ ^1.internal_type = MemberExpression<br>⇒ y = ␣<br>Confidence: 0.998. Support: 689.` |
| 315 | `  -1.reserved = (<br>	∧ -1.roles not in {STRING}<br>	∧ +1.reserved = )<br>	∧ +1.roles not in {STRING}<br>	∧ ^1.internal_type = MemberExpression<br>⇒ y = ∅<br>Confidence: 0.999. Support: 374.` |
| 316 | `  -1.reserved not in {(, [}<br>	∧ -1.roles not in {STRING}<br>	∧ +1.reserved = )<br>	∧ +1.roles not in {STRING}<br>	∧ ^1.internal_type = MemberExpression<br>⇒ y = ␣<br>Confidence: 0.991. Support: 764.` |
| 317 | `  -1.reserved not in {[}<br>	∧ -1.roles not in {STRING}<br>	∧ +1.reserved = ]<br>	∧ +1.roles not in {STRING}<br>	∧ ^1.internal_type = MemberExpression<br>⇒ y = ␣<br>Confidence: 0.998. Support: 212.` |
| 318 | `  -1.reserved not in {[}<br>	∧ -1.roles not in {STRING}<br>	∧ +1.reserved = }<br>	∧ +1.roles not in {STRING}<br>	∧ ^1.internal_type = MemberExpression<br>⇒ y = ⏎⇥⁻<br>Confidence: 0.825. Support: 140.` |
| 319 | `  -1.reserved not in {[}<br>	∧ -1.roles not in {STRING}<br>	∧ +1.reserved not in {), ], }}<br>	∧ +1.roles not in {STRING}<br>	∧ ^1.internal_type = MemberExpression<br>⇒ y = ∅<br>Confidence: 0.987. Support: 23548.` |
| 320 | `  -1.reserved = {<br>	∧ -1.roles not in {STRING}<br>	∧ +1.roles in {EXPRESSION} and not in {STRING}<br>	∧ +3.roles not in {VALUE}<br>	∧ ^1.internal_type not in {MemberExpression}<br>⇒ y = ⏎⇥⁺<br>Confidence: 0.892. Support: 1874.` |
| 321 | `  -1.internal_type = CommentLine<br>	∧ -1.reserved not in {;, {}<br>	∧ -1.roles not in {STRING}<br>	∧ +1.roles in {EXPRESSION} and not in {KEY, STRING}<br>	∧ ^1.internal_type not in {MemberExpression}<br>⇒ y = ⏎<br>Confidence: 0.972. Support: 518.` |
| 322 | `  -1.internal_type not in {CommentLine}<br>	∧ -1.reserved not in {;, {}<br>	∧ -1.roles not in {STRING}<br>	∧ +1.roles in {EXPRESSION} and not in {KEY, STRING}<br>	∧ +2.reserved not in {=}<br>	∧ ^1.internal_type = VariableDeclaration<br>⇒ y = ␣<br>Confidence: 0.968. Support: 297.` |
| 323 | `  -1.internal_type not in {CommentLine}<br>	∧ -1.reserved not in {;, {}<br>	∧ -1.roles not in {STRING}<br>	∧ -3.diff_offset ≥ 4<br>	∧ -5.diff_offset ≥ 7<br>	∧ +1.roles in {EXPRESSION} and not in {KEY, STRING}<br>	∧ ^1.internal_type not in {MemberExpression, VariableDeclaration}<br>⇒ y = ␣<br>Confidence: 0.966. Support: 14419.` |
| 324 | `  -1.internal_type not in {CommentLine}<br>	∧ -1.reserved = ,<br>	∧ -1.roles not in {STRING}<br>	∧ -3.diff_offset ≤ 3<br>	∧ -5.diff_offset ≥ 7<br>	∧ +1.roles in {EXPRESSION} and not in {KEY, STRING}<br>	∧ ^1.internal_type not in {MemberExpression, VariableDeclaration}<br>⇒ y = ␣<br>Confidence: 0.948. Support: 567.` |
| 325 | `  -1.internal_type not in {CommentLine}<br>	∧ -1.reserved not in {,, ;, {}<br>	∧ -1.roles not in {STRING}<br>	∧ -3.diff_offset ≤ 3<br>	∧ -5.diff_offset ≥ 7<br>	∧ +1.roles in {EXPRESSION} and not in {KEY, LITERAL}<br>	∧ ^1.internal_type not in {MemberExpression, VariableDeclaration}<br>⇒ y = ␣<br>Confidence: 0.816. Support: 290.` |
| 326 | `  -1.internal_type not in {CommentLine}<br>	∧ -1.reserved not in {;, {}<br>	∧ -1.roles not in {STRING}<br>	∧ -3.reserved = (<br>	∧ -5.diff_offset ≤ 6<br>	∧ +1.roles in {EXPRESSION} and not in {KEY, STRING}<br>	∧ ^1.internal_type not in {MemberExpression, VariableDeclaration}<br>⇒ y = ∅<br>Confidence: 0.944. Support: 242.` |
| 327 | `  •••start_col ≥ 4<br>	∧ -1.internal_type not in {CommentLine}<br>	∧ -1.reserved not in {;, {}<br>	∧ -1.roles not in {STRING}<br>	∧ -3.reserved not in {(}<br>	∧ -5.diff_offset ≤ 6<br>	∧ +1.roles in {EXPRESSION} and not in {KEY, STRING}<br>	∧ ^1.internal_type not in {MemberExpression, VariableDeclaration}<br>⇒ y = ␣<br>Confidence: 0.821. Support: 802.` |
| 328 | `  -1.roles not in {STRING}<br>	∧ +1.roles not in {EXPRESSION, STRING}<br>	∧ +2.roles in {ARGUMENT}<br>	∧ ^1.internal_type not in {MemberExpression}<br>⇒ y = ∅<br>Confidence: 0.995. Support: 11730.` |
| 329 | `  -1.roles not in {STRING}<br>	∧ +1.reserved = ;<br>	∧ +1.roles not in {EXPRESSION, STRING}<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.internal_type not in {MemberExpression}<br>⇒ y = ∅<br>Confidence: 0.990. Support: 8021.` |
| 330 | `  -1.roles in {NUMBER} and not in {STRING}<br>	∧ +1.reserved = }<br>	∧ +1.roles not in {EXPRESSION, STRING}<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.internal_type not in {MemberExpression}<br>⇒ y = ␣<br>Confidence: 0.856. Support: 156.` |
| 331 | `  -1.roles not in {NUMBER, STRING}<br>	∧ +1.reserved = }<br>	∧ +1.roles not in {EXPRESSION, STRING}<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.internal_type not in {MemberExpression}<br>⇒ y = ⏎⇥⁻<br>Confidence: 0.881. Support: 2954.` |
| 332 | `  -1.roles not in {STRING}<br>	∧ +1.reserved = ,<br>	∧ +1.roles not in {EXPRESSION, STRING}<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.internal_type not in {MemberExpression}<br>⇒ y = ∅<br>Confidence: 1.000. Support: 3404.` |
| 333 | `  -1.reserved = if<br>	∧ -1.roles not in {STRING}<br>	∧ +1.reserved = (<br>	∧ +1.roles not in {EXPRESSION, STRING}<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.internal_type not in {MemberExpression}<br>⇒ y = ␣<br>Confidence: 0.999. Support: 590.` |
| 334 | `  -1.reserved not in {if}<br>	∧ -1.roles in {IDENTIFIER} and not in {STRING}<br>	∧ +1.reserved = (<br>	∧ +1.roles not in {EXPRESSION, STRING}<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.internal_type not in {MemberExpression}<br>⇒ y = ∅<br>Confidence: 1.000. Support: 4130.` |
| 335 | `  -1.reserved not in {if}<br>	∧ -1.roles not in {IDENTIFIER, STRING}<br>	∧ +1.reserved = (<br>	∧ +1.roles not in {EXPRESSION, STRING}<br>	∧ +2.reserved = )<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.internal_type not in {MemberExpression}<br>⇒ y = ∅<br>Confidence: 0.999. Support: 540.` |
| 336 | `  -1.roles not in {STRING}<br>	∧ +1.reserved not in {,, ;, }}<br>	∧ +1.roles in {COMMENT} and not in {EXPRESSION, STRING}<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.internal_type not in {MemberExpression}<br>⇒ y = ∅<br>Confidence: 0.987. Support: 1483.` |
| 337 | `  -1.roles in {KEY} and not in {STRING}<br>	∧ +1.reserved not in {,, ;, }}<br>	∧ +1.roles not in {COMMENT, EXPRESSION, STRING}<br>	∧ ^1.internal_type not in {MemberExpression}<br>⇒ y = ∅<br>Confidence: 1.000. Support: 1290.` |
| 338 | `  -1.reserved = ;<br>	∧ -1.roles not in {KEY, STRING}<br>	∧ +1.reserved not in {(, ,, ;, }}<br>	∧ +1.roles not in {COMMENT, EXPRESSION, STRING}<br>	∧ +1.length ≤ 4<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ +2.length ≥ 2<br>	∧ ^1.internal_type not in {MemberExpression}<br>⇒ y = ⏎⏎<br>Confidence: 0.820. Support: 906.` |
| 339 | `  -1.reserved = (<br>	∧ -1.roles not in {KEY, STRING}<br>	∧ +1.reserved = )<br>	∧ +1.roles not in {EXPRESSION, STRING}<br>	∧ ^1.internal_type not in {MemberExpression}<br>⇒ y = ∅<br>Confidence: 1.000. Support: 1588.` |
| 340 | `  -1.reserved = (<br>	∧ -1.roles not in {KEY, STRING}<br>	∧ +1.reserved not in {(, ), ,, }}<br>	∧ +1.roles not in {COMMENT, EXPRESSION, STRING}<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.internal_type not in {MemberExpression}<br>⇒ y = ␣<br>Confidence: 0.984. Support: 982.` |
| 341 | `  -1.internal_type = CommentLine<br>	∧ -1.reserved not in {(, ;}<br>	∧ -1.roles not in {KEY, STRING}<br>	∧ +1.reserved not in {(, ,, ;, }}<br>	∧ +1.roles not in {COMMENT, EXPRESSION, STRING}<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.internal_type not in {MemberExpression}<br>⇒ y = ⏎<br>Confidence: 0.985. Support: 766.` |
| 342 | `  -1.internal_type not in {CommentLine}<br>	∧ -1.reserved = {<br>	∧ -1.roles not in {KEY, STRING}<br>	∧ +1.reserved not in {,, }}<br>	∧ +1.roles not in {COMMENT, EXPRESSION, STRING}<br>	∧ +2.roles in {COMMENT} and not in {ARGUMENT}<br>	∧ ^1.internal_type not in {MemberExpression}<br>⇒ y = ⏎⏎⇥⁺<br>Confidence: 0.903. Support: 191.` |
| 343 | `  -1.internal_type not in {CommentLine}<br>	∧ -1.reserved = {<br>	∧ -1.roles not in {KEY, STRING}<br>	∧ +1.reserved not in {,, }}<br>	∧ +1.roles not in {COMMENT, EXPRESSION, STRING}<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.internal_type not in {MemberExpression}<br>⇒ y = ⏎⇥⁺<br>Confidence: 0.936. Support: 738.` |
| 344 | `  -1.internal_type not in {CommentLine}<br>	∧ -1.reserved not in {(, ;, {}<br>	∧ -1.roles not in {KEY, STRING}<br>	∧ +1.reserved not in {(, ,, ;, }}<br>	∧ +1.roles not in {COMMENT, EXPRESSION, STRING}<br>	∧ +2.roles in {COMMENT} and not in {ARGUMENT}<br>	∧ ^1.internal_type not in {MemberExpression}<br>⇒ y = ⏎⏎<br>Confidence: 0.891. Support: 326.` |
| 345 | `  -1.internal_type not in {CommentLine}<br>	∧ -1.reserved not in {(, ;, {}<br>	∧ -1.roles not in {KEY, STRING}<br>	∧ -3.label not in {<space>}<br>	∧ +1.reserved not in {(, ,, ;, }}<br>	∧ +1.roles not in {COMMENT, EXPRESSION, STRING}<br>	∧ +1.length ≥ 2<br>	∧ +2.roles not in {ARGUMENT, COMMENT}<br>	∧ ^1.internal_type not in {MemberExpression}<br>	∧ ^1.roles not in {BLOCK}<br>⇒ y = ␣<br>Confidence: 0.933. Support: 2545.` |
| 346 | `  -1.internal_type not in {CommentLine}<br>	∧ -1.reserved not in {(, ;, {}<br>	∧ -1.roles not in {KEY, STRING}<br>	∧ -3.label not in {<space>}<br>	∧ +1.reserved not in {(, ,, ;, }}<br>	∧ +1.roles not in {COMMENT, EXPRESSION, STRING}<br>	∧ +1.length ≤ 1<br>	∧ +2.roles not in {ARGUMENT, COMMENT}<br>	∧ ^1.internal_type not in {MemberExpression}<br>⇒ y = ␣<br>Confidence: 0.981. Support: 13358.` |

<details>
    <summary>Machine-readable report</summary>
```json
{"javascript": {"avg_rule_len": 9.326589595375722, "max_conf": 0.9998798370361328, "max_support": 24572, "min_conf": 0.8004807829856873, "min_support": 129, "num_rules": 346}}
```
</details>
