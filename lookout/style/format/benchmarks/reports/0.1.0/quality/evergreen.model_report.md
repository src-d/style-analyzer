# Model report for file:///tmp/top-repos-quality-repos-0i_v0ztk/evergreen HEAD ba22d511dad83c072842e47801ef42697d142f7c

### Dump

```json
{'created_at': datetime.datetime(2019, 1, 31, 11, 31, 46, 943705),
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
 'uuid': '727d58df-b1d2-4fee-8322-07f86de7adf4',
 'version': [1]}
style.format.analyzer.FormatAnalyzer/[1] file:///tmp/top-repos-quality-repos-0i_v0ztk/evergreen ba22d511dad83c072842e47801ef42697d142f7c

# javascript
66 rules, avg.len. 10.6
## train
PPCR: 0.902851
### report
macro
{'f1-score': 0.6182971268590753,
 'precision': 0.632135234098738,
 'recall': 0.608142589745029,
 'support': 72080}
micro
{'f1-score': 0.9524972253052164,
 'precision': 0.9524972253052164,
 'recall': 0.9524972253052164,
 'support': 72080}
weighted
{'f1-score': 0.9498333590120437,
 'precision': 0.9486264009177691,
 'recall': 0.9524972253052164,
 'support': 72080}
### report_full
macro
{'f1-score': 0.5571192292422905,
 'precision': 0.632135234098738,
 'recall': 0.5115884004120385,
 'support': 79836}
micro
{'f1-score': 0.9038679270122962,
 'precision': 0.9524972253052164,
 'recall': 0.859962923994188,
 'support': 79836}
weighted
{'f1-score': 0.892402918714448,
 'precision': 0.9390555501331619,
 'recall': 0.859962923994188,
 'support': 79836}
## test
PPCR: 0.920519
### report
macro
{'f1-score': 0.5923576159100546,
 'precision': 0.6193968008860635,
 'recall': 0.5821532946375256,
 'support': 16944}
micro
{'f1-score': 0.9326605288007555,
 'precision': 0.9326605288007555,
 'recall': 0.9326605288007555,
 'support': 16944}
weighted
{'f1-score': 0.9298580460158782,
 'precision': 0.936049036265096,
 'recall': 0.9326605288007555,
 'support': 16944}
### report_full
macro
{'f1-score': 0.5358133090456343,
 'precision': 0.6193968008860635,
 'recall': 0.49762019668377366,
 'support': 18407}
micro
{'f1-score': 0.8940624027608837,
 'precision': 0.9326605288007555,
 'recall': 0.8585320801868854,
 'support': 18407}
weighted
{'f1-score': 0.8812946274578233,
 'precision': 0.9260063176294039,
 'recall': 0.8585320801868854,
 'support': 18407}
```

## javascript
### Summary
66 rules, avg.len. 10.6

| | |
|-|-|
|Min support|90|
|Max support|26218|
|Min confidence|0.8051643371582031|
|Max confidence|0.9993489384651184|

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
| 1 | `  +1.internal_type = StringLiteral<br>	∧ ^1.roles in {DECLARATION}<br>⇒ y = ␣'<br>Confidence: 0.995. Support: 881.` |
| 2 | `  -1.reserved = {<br>	∧ -4.length ≤ 1<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ ^1.roles in {DECLARATION}<br>⇒ y = ⏎␣⁺␣⁺<br>Confidence: 0.951. Support: 478.` |
| 3 | `  -1.reserved not in {{}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = ><br>	∧ ^1.roles in {DECLARATION}<br>⇒ y = ∅<br>Confidence: 0.999. Support: 389.` |
| 4 | `  -1.reserved not in {{}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = ,<br>	∧ ^1.roles in {DECLARATION}<br>⇒ y = ∅<br>Confidence: 0.998. Support: 327.` |
| 5 | `  -1.reserved not in {{}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = )<br>	∧ ^1.roles in {DECLARATION}<br>⇒ y = ∅<br>Confidence: 0.984. Support: 277.` |
| 6 | `  -1.reserved = (<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {), ,, <, >}<br>	∧ ^1.roles in {DECLARATION}<br>⇒ y = ∅<br>Confidence: 0.989. Support: 134.` |
| 7 | `  -1.reserved not in {(, {}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {), ,, >}<br>	∧ +1.roles in {UNANNOTATED}<br>	∧ ^1.roles in {DECLARATION}<br>⇒ y = ∅<br>Confidence: 0.997. Support: 161.` |
| 8 | `  -1.reserved not in {(, {}<br>	∧ -1.roles in {KEY}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {), ,, >}<br>	∧ +1.roles not in {UNANNOTATED}<br>	∧ ^1.roles in {DECLARATION}<br>⇒ y = ∅<br>Confidence: 0.996. Support: 128.` |
| 9 | `  -1.reserved not in {(, {}<br>	∧ -1.roles not in {KEY}<br>	∧ -2.length ≤ 10<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {), ,, >}<br>	∧ +1.roles not in {UNANNOTATED}<br>	∧ ^1.roles in {DECLARATION}<br>⇒ y = ␣<br>Confidence: 0.956. Support: 4399.` |
| 10 | `  -1.reserved = :<br>	∧ +1.roles in {MAP, STRING}<br>	∧ ^1.roles not in {DECLARATION}<br>⇒ y = ␣'<br>Confidence: 0.999. Support: 429.` |
| 11 | `  -1.reserved = :<br>	∧ +1.roles in {MAP} and not in {STRING}<br>	∧ ^1.roles not in {DECLARATION}<br>⇒ y = ␣<br>Confidence: 0.999. Support: 452.` |
| 12 | `  -1.reserved = {<br>	∧ +1.roles in {MAP} and not in {VALUE}<br>	∧ +4.reserved not in {}}<br>	∧ ^1.roles not in {DECLARATION}<br>⇒ y = ⏎␣⁺␣⁺<br>Confidence: 0.844. Support: 452.` |
| 13 | `  -1.reserved not in {:, {}<br>	∧ -3.label in {<space>}<br>	∧ +1.roles in {KEY, MAP}<br>	∧ ^1.roles not in {DECLARATION}<br>⇒ y = ⏎<br>Confidence: 0.920. Support: 230.` |
| 14 | `  -1.reserved not in {:, {}<br>	∧ -3.label in {<space>}<br>	∧ +1.roles in {MAP} and not in {KEY}<br>	∧ ^1.roles not in {DECLARATION}<br>⇒ y = ␣<br>Confidence: 0.922. Support: 172.` |
| 15 | `  -1.reserved not in {:, {}<br>	∧ -3.label not in {<space>}<br>	∧ +1.roles in {MAP}<br>	∧ ^1.roles not in {DECLARATION}<br>⇒ y = ⏎<br>Confidence: 0.898. Support: 1570.` |
| 16 | `  -1.internal_type = StringLiteral<br>	∧ -3.roles in {UNANNOTATED}<br>	∧ -4.length ≥ 2<br>	∧ +1.roles not in {MAP}<br>	∧ +1.length ≥ 2<br>	∧ ^1.roles not in {DECLARATION}<br>⇒ y = "⏎<br>Confidence: 0.955. Support: 301.` |
| 17 | `  -1.internal_type = StringLiteral<br>	∧ -3.roles in {UNANNOTATED}<br>	∧ -4.length ≤ 1<br>	∧ +1.roles not in {MAP}<br>	∧ +1.length ≥ 2<br>	∧ ^1.roles not in {DECLARATION}<br>⇒ y = "␣<br>Confidence: 0.996. Support: 130.` |
| 18 | `  -1.internal_type = StringLiteral<br>	∧ -3.roles not in {UNANNOTATED}<br>	∧ +1.roles not in {MAP}<br>	∧ +1.length ≥ 6<br>	∧ ^1.roles not in {DECLARATION}<br>⇒ y = '⏎<br>Confidence: 0.921. Support: 739.` |
| 19 | `  -1.internal_type = StringLiteral<br>	∧ -2.reserved not in {=}<br>	∧ +1.reserved not in {}}<br>	∧ +1.roles not in {MAP}<br>	∧ +1.length ≤ 1<br>	∧ ^1.roles not in {DECLARATION}<br>⇒ y = '<br>Confidence: 0.933. Support: 858.` |
| 20 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = :<br>	∧ +1.roles not in {MAP}<br>	∧ ^1.roles not in {DECLARATION}<br>⇒ y = ␣<br>Confidence: 0.950. Support: 1197.` |
| 21 | `  •••start_col ≤ 21<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {:}<br>	∧ -3.diff_offset ≥ 8<br>	∧ +1.roles not in {MAP}<br>	∧ ^1.roles not in {DECLARATION}<br>	∧ ^2.internal_type = File<br>⇒ y = ␣<br>Confidence: 0.941. Support: 1470.` |
| 22 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {:}<br>	∧ -3.diff_offset ≤ 7<br>	∧ +1.roles not in {MAP}<br>	∧ +1.length ≥ 3<br>	∧ ^1.roles not in {DECLARATION}<br>	∧ ^2.internal_type = File<br>⇒ y = ⏎⏎<br>Confidence: 0.805. Support: 213.` |
| 23 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {:}<br>	∧ -3.diff_offset ≤ 7<br>	∧ +1.reserved = }<br>	∧ +1.roles not in {MAP}<br>	∧ +1.length ≤ 2<br>	∧ ^1.roles not in {DECLARATION}<br>	∧ ^2.internal_type = File<br>⇒ y = ⏎␣⁻␣⁻<br>Confidence: 0.971. Support: 157.` |
| 24 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {:}<br>	∧ +1.internal_type = StringLiteral<br>	∧ +1.roles not in {MAP}<br>	∧ ^1.roles in {OPERATOR} and not in {DECLARATION}<br>	∧ ^2.internal_type not in {File}<br>⇒ y = ␣'<br>Confidence: 0.997. Support: 159.` |
| 25 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {:}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.roles not in {MAP}<br>	∧ ^1.roles in {OPERATOR} and not in {DECLARATION}<br>	∧ ^2.internal_type not in {File}<br>⇒ y = ␣<br>Confidence: 0.902. Support: 1280.` |
| 26 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = }<br>	∧ +1.roles in {KEY} and not in {MAP}<br>	∧ +2.reserved = =<br>	∧ ^1.roles not in {DECLARATION, OPERATOR}<br>	∧ ^2.internal_type not in {File}<br>⇒ y = ⏎⏎<br>Confidence: 0.996. Support: 136.` |
| 27 | `  •••start_col ≥ 33<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = }<br>	∧ -2.roles in {NUMBER}<br>	∧ +1.roles not in {KEY, MAP}<br>	∧ +2.reserved = =<br>	∧ ^1.roles not in {DECLARATION}<br>	∧ ^2.internal_type not in {File}<br>⇒ y = ␣<br>Confidence: 0.854. Support: 134.` |
| 28 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = }<br>	∧ -2.roles not in {NUMBER}<br>	∧ +1.roles not in {KEY, MAP}<br>	∧ +2.reserved = =<br>	∧ ^1.roles not in {DECLARATION, OPERATOR}<br>	∧ ^2.internal_type not in {File}<br>⇒ y = ⏎<br>Confidence: 0.865. Support: 582.` |
| 29 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {:, }}<br>	∧ -1.length ≥ 2<br>	∧ -4.label not in {<newline>}<br>	∧ +1.roles not in {MAP}<br>	∧ +2.reserved = =<br>	∧ ^1.roles not in {DECLARATION}<br>	∧ ^2.internal_type not in {File}<br>⇒ y = ␣<br>Confidence: 0.814. Support: 904.` |
| 30 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {:, }}<br>	∧ -1.length ≤ 1<br>	∧ -3.roles in {EXPRESSION}<br>	∧ +1.roles not in {MAP}<br>	∧ +2.reserved = =<br>	∧ ^1.roles not in {DECLARATION}<br>	∧ ^2.internal_type not in {File}<br>⇒ y = ␣<br>Confidence: 0.858. Support: 102.` |
| 31 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {:, }}<br>	∧ -1.length ≤ 1<br>	∧ -3.roles not in {EXPRESSION}<br>	∧ +1.roles not in {MAP}<br>	∧ +2.reserved = =<br>	∧ ^1.roles not in {DECLARATION, OPERATOR}<br>	∧ ^2.internal_type not in {File}<br>⇒ y = ∅<br>Confidence: 0.864. Support: 601.` |
| 32 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {:}<br>	∧ +1.internal_type = StringLiteral<br>	∧ +1.roles not in {MAP}<br>	∧ +2.reserved not in {=}<br>	∧ ^1.roles not in {DECLARATION, OPERATOR}<br>	∧ ^2.internal_type = JSXOpeningElement<br>⇒ y = "<br>Confidence: 0.999. Support: 768.` |
| 33 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, :}<br>	∧ +1.internal_type = StringLiteral<br>	∧ +1.roles not in {MAP}<br>	∧ +2.reserved not in {=}<br>	∧ ^1.roles not in {DECLARATION, OPERATOR}<br>	∧ ^2.internal_type not in {File, JSXOpeningElement}<br>⇒ y = '<br>Confidence: 0.866. Support: 363.` |
| 34 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = ,<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.roles not in {MAP}<br>	∧ +2.internal_type = CommentBlock<br>	∧ +2.reserved not in {=}<br>	∧ ^1.roles not in {DECLARATION, OPERATOR}<br>	∧ ^2.internal_type not in {File}<br>⇒ y = ⏎⏎<br>Confidence: 0.966. Support: 337.` |
| 35 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = ,<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.roles not in {MAP}<br>	∧ +2.internal_type not in {CommentBlock}<br>	∧ +2.reserved not in {=}<br>	∧ ^1.roles in {MAP} and not in {DECLARATION, OPERATOR}<br>	∧ ^2.internal_type not in {File}<br>⇒ y = ⏎<br>Confidence: 0.895. Support: 147.` |
| 36 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = ,<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.roles not in {MAP}<br>	∧ +2.reserved not in {=}<br>	∧ ^1.internal_type = CallExpression<br>	∧ ^1.roles not in {DECLARATION, MAP}<br>	∧ ^2.internal_type not in {File}<br>⇒ y = ␣<br>Confidence: 0.888. Support: 263.` |
| 37 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, :}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = )<br>	∧ +1.roles not in {MAP}<br>	∧ +2.reserved not in {=}<br>	∧ ^1.roles in {STATEMENT} and not in {DECLARATION, OPERATOR}<br>	∧ ^2.internal_type not in {File}<br>⇒ y = ∅<br>Confidence: 0.929. Support: 474.` |
| 38 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, :}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = }<br>	∧ +1.roles not in {MAP}<br>	∧ +2.reserved not in {=}<br>	∧ ^1.roles in {STATEMENT} and not in {DECLARATION, OPERATOR}<br>	∧ ^2.internal_type not in {File}<br>⇒ y = ⏎␣⁻␣⁻<br>Confidence: 0.853. Support: 241.` |
| 39 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = {<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {), }}<br>	∧ +1.roles not in {MAP}<br>	∧ +2.reserved not in {=}<br>	∧ ^1.roles in {STATEMENT} and not in {DECLARATION, OPERATOR}<br>	∧ ^2.internal_type not in {File}<br>⇒ y = ⏎␣⁺␣⁺<br>Confidence: 0.997. Support: 154.` |
| 40 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, :, {}<br>	∧ -3.label in {<newline>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {), }}<br>	∧ +1.roles not in {MAP}<br>	∧ +2.reserved not in {=}<br>	∧ ^1.internal_type = BlockStatement<br>	∧ ^1.roles in {STATEMENT} and not in {DECLARATION}<br>	∧ ^2.internal_type not in {File}<br>⇒ y = ␣<br>Confidence: 0.997. Support: 152.` |
| 41 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {(, ,, :, {}<br>	∧ -3.label not in {<newline>}<br>	∧ -4.diff_line ≥ 1<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {), }}<br>	∧ +1.roles not in {MAP}<br>	∧ +1.length ≤ 1<br>	∧ +2.reserved not in {=}<br>	∧ ^1.internal_type = BlockStatement<br>	∧ ^1.roles in {STATEMENT} and not in {DECLARATION}<br>	∧ ^2.internal_type not in {File}<br>⇒ y = ␣<br>Confidence: 0.928. Support: 90.` |
| 42 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, :}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = }<br>	∧ +1.roles not in {MAP}<br>	∧ +2.reserved not in {=}<br>	∧ ^1.roles not in {DECLARATION, OPERATOR, STATEMENT}<br>	∧ ^2.internal_type not in {File}<br>	∧ ^2.roles in {TYPE}<br>⇒ y = ⏎␣⁻␣⁻<br>Confidence: 0.970. Support: 386.` |
| 43 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, :}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {}}<br>	∧ +1.roles not in {MAP}<br>	∧ +1.length ≤ 1<br>	∧ +2.reserved not in {=}<br>	∧ ^1.roles not in {DECLARATION, STATEMENT}<br>	∧ ^2.internal_type = ClassBody<br>	∧ ^2.roles in {TYPE}<br>⇒ y = ␣<br>Confidence: 0.897. Support: 521.` |
| 44 | `  •••start_col ≥ 5<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, :}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {}}<br>	∧ +1.roles not in {MAP}<br>	∧ +2.reserved not in {=}<br>	∧ ^1.roles not in {DECLARATION, OPERATOR, STATEMENT}<br>	∧ ^2.internal_type not in {ClassBody, File}<br>	∧ ^2.roles in {TYPE}<br>⇒ y = ∅<br>Confidence: 0.883. Support: 320.` |
| 45 | `  •••start_col ≤ 5<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, :}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {}}<br>	∧ +1.roles not in {MAP}<br>	∧ +2.reserved not in {=}<br>	∧ ^1.roles not in {DECLARATION, OPERATOR, STATEMENT}<br>	∧ ^2.internal_type not in {ClassBody, File}<br>	∧ ^2.roles in {TYPE}<br>⇒ y = ⏎⏎<br>Confidence: 0.997. Support: 167.` |
| 46 | `  -1.internal_type = CommentLine<br>	∧ -1.reserved not in {,, :}<br>	∧ -4.diff_offset ≥ 5<br>	∧ +1.roles not in {MAP}<br>	∧ +1.length ≥ 1<br>	∧ +2.reserved not in {=}<br>	∧ ^1.roles not in {DECLARATION, OPERATOR, STATEMENT}<br>	∧ ^2.internal_type not in {File}<br>	∧ ^2.roles not in {TYPE}<br>⇒ y = ⏎<br>Confidence: 0.903. Support: 139.` |
| 47 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {,, :}<br>	∧ -2.internal_type = CommentBlock<br>	∧ -4.diff_offset ≥ 5<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.roles not in {MAP}<br>	∧ +2.reserved not in {=}<br>	∧ ^1.roles not in {DECLARATION, OPERATOR, STATEMENT}<br>	∧ ^2.internal_type not in {File}<br>	∧ ^2.roles not in {TYPE}<br>⇒ y = ⏎<br>Confidence: 0.965. Support: 130.` |
| 48 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {,, :}<br>	∧ -2.internal_type not in {CommentBlock}<br>	∧ -4.diff_offset ≥ 5<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = }<br>	∧ +1.roles not in {MAP}<br>	∧ +2.reserved not in {=}<br>	∧ ^1.roles in {UNANNOTATED} and not in {DECLARATION, OPERATOR, STATEMENT}<br>	∧ ^2.internal_type not in {File}<br>	∧ ^2.roles not in {TYPE}<br>⇒ y = ∅<br>Confidence: 0.926. Support: 1733.` |
| 49 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {,, :}<br>	∧ -2.internal_type not in {CommentBlock}<br>	∧ -4.diff_offset ≥ 5<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = }<br>	∧ +1.roles not in {MAP}<br>	∧ +2.reserved not in {=}<br>	∧ ^1.internal_type = JSXElement<br>	∧ ^1.roles not in {DECLARATION, OPERATOR, STATEMENT, UNANNOTATED}<br>	∧ ^2.internal_type not in {File}<br>	∧ ^2.roles not in {TYPE}<br>⇒ y = ∅<br>Confidence: 0.869. Support: 172.` |
| 50 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {,, :}<br>	∧ -2.internal_type not in {CommentBlock}<br>	∧ -4.diff_offset ≥ 5<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = }<br>	∧ +1.roles not in {MAP}<br>	∧ +2.reserved not in {=}<br>	∧ +2.roles in {STRING}<br>	∧ ^1.internal_type not in {JSXElement}<br>	∧ ^1.roles not in {DECLARATION, OPERATOR, STATEMENT, UNANNOTATED}<br>	∧ ^2.internal_type not in {File}<br>	∧ ^2.roles not in {TYPE}<br>⇒ y = ∅<br>Confidence: 0.854. Support: 106.` |
| 51 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved = default<br>	∧ -4.diff_offset ≥ 5<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {}}<br>	∧ +1.roles not in {MAP}<br>	∧ +2.reserved not in {=}<br>	∧ ^1.roles not in {DECLARATION, STATEMENT}<br>	∧ ^2.internal_type not in {File}<br>	∧ ^2.roles not in {TYPE}<br>⇒ y = ␣<br>Confidence: 0.937. Support: 103.` |
| 52 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved = }<br>	∧ -2.internal_type not in {CommentBlock}<br>	∧ -3.diff_offset ≤ 8<br>	∧ -4.diff_offset ≥ 5<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {}}<br>	∧ +1.roles not in {MAP}<br>	∧ +2.reserved not in {=, >}<br>	∧ +2.roles not in {COMMENT}<br>	∧ +2.length ≥ 9<br>	∧ ^1.internal_type not in {JSXOpeningElement}<br>	∧ ^1.roles not in {DECLARATION, OPERATOR, STATEMENT}<br>	∧ ^2.internal_type not in {File}<br>	∧ ^2.roles not in {TYPE}<br>⇒ y = ∅<br>Confidence: 0.832. Support: 182.` |
| 53 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved = }<br>	∧ -2.internal_type not in {CommentBlock}<br>	∧ -4.diff_offset ≥ 5<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {}}<br>	∧ +1.roles not in {MAP}<br>	∧ +2.reserved not in {=, >}<br>	∧ +2.roles not in {COMMENT}<br>	∧ +2.length ≤ 8<br>	∧ ^1.internal_type not in {JSXOpeningElement}<br>	∧ ^1.roles not in {DECLARATION, OPERATOR, STATEMENT}<br>	∧ ^2.internal_type not in {File}<br>	∧ ^2.roles not in {TYPE}<br>⇒ y = ∅<br>Confidence: 0.978. Support: 834.` |
| 54 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {,, }}<br>	∧ -3.length ≥ 5<br>	∧ -4.diff_offset ≥ 5<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {}}<br>	∧ +1.roles in {INCOMPLETE} and not in {MAP}<br>	∧ +2.reserved not in {=}<br>	∧ ^1.roles not in {DECLARATION, STATEMENT}<br>	∧ ^2.internal_type not in {File}<br>	∧ ^2.roles not in {TYPE}<br>⇒ y = ␣<br>Confidence: 0.951. Support: 91.` |
| 55 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {,, :, }}<br>	∧ -2.internal_type not in {CommentBlock}<br>	∧ -3.length ≤ 5<br>	∧ -4.diff_offset ≥ 5<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {}}<br>	∧ +1.roles in {INCOMPLETE} and not in {MAP}<br>	∧ +2.reserved not in {=}<br>	∧ +2.roles not in {COMMENT}<br>	∧ ^1.roles not in {DECLARATION, OPERATOR, STATEMENT}<br>	∧ ^2.internal_type not in {File}<br>	∧ ^2.roles not in {TYPE}<br>⇒ y = ∅<br>Confidence: 0.995. Support: 106.` |
| 56 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {), ,, :, >}<br>	∧ -2.internal_type not in {CommentBlock}<br>	∧ -4.diff_offset ≥ 5<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = )<br>	∧ +1.roles not in {MAP}<br>	∧ +2.reserved not in {=}<br>	∧ +2.roles not in {COMMENT}<br>	∧ ^1.roles not in {DECLARATION, OPERATOR, STATEMENT}<br>	∧ ^2.internal_type not in {File}<br>	∧ ^2.roles not in {TYPE}<br>⇒ y = ∅<br>Confidence: 0.985. Support: 497.` |
| 57 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {,}<br>	∧ -4.diff_offset ≥ 5<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = {<br>	∧ +1.roles not in {MAP}<br>	∧ +2.reserved not in {=}<br>	∧ ^1.internal_type = File<br>	∧ ^1.roles not in {DECLARATION, STATEMENT}<br>	∧ ^2.internal_type not in {File}<br>	∧ ^2.roles not in {TYPE}<br>⇒ y = ␣<br>Confidence: 0.973. Support: 92.` |
| 58 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {), ,, :, }}<br>	∧ -2.internal_type not in {CommentBlock}<br>	∧ -4.diff_offset ≥ 5<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {), {, }}<br>	∧ +1.roles not in {MAP}<br>	∧ +2.reserved not in {=}<br>	∧ +2.roles not in {COMMENT}<br>	∧ +2.length ≥ 2<br>	∧ ^1.internal_type = File<br>	∧ ^1.roles not in {DECLARATION, OPERATOR, STATEMENT}<br>	∧ ^2.internal_type not in {File}<br>	∧ ^2.roles not in {TYPE}<br>⇒ y = ∅<br>Confidence: 0.966. Support: 1561.` |
| 59 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {), ,, :, }}<br>	∧ -1.length ≤ 4<br>	∧ -2.internal_type not in {CommentBlock}<br>	∧ -4.diff_offset ≥ 5<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {), {, }}<br>	∧ +1.roles not in {MAP}<br>	∧ +2.reserved not in {=}<br>	∧ +2.roles not in {COMMENT}<br>	∧ +2.length ≤ 1<br>	∧ ^1.internal_type = File<br>	∧ ^1.roles not in {DECLARATION, OPERATOR, STATEMENT}<br>	∧ ^2.internal_type not in {File}<br>	∧ ^2.roles not in {TYPE}<br>⇒ y = ∅<br>Confidence: 0.859. Support: 160.` |
| 60 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {), ,, :, }}<br>	∧ -2.internal_type not in {CommentBlock}<br>	∧ -4.diff_offset ≥ 5<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {), }}<br>	∧ +1.roles not in {MAP}<br>	∧ +2.reserved not in {=}<br>	∧ +2.roles not in {COMMENT}<br>	∧ ^1.internal_type not in {File}<br>	∧ ^1.roles not in {DECLARATION, OPERATOR, STATEMENT}<br>	∧ ^2.internal_type not in {File}<br>	∧ ^2.roles not in {TYPE}<br>⇒ y = ∅<br>Confidence: 0.987. Support: 26218.` |
| 61 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, :}<br>	∧ -4.diff_offset ≥ 5<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.roles not in {MAP}<br>	∧ +1.length = 0<br>	∧ +2.reserved not in {=}<br>	∧ ^1.roles not in {DECLARATION, OPERATOR, STATEMENT}<br>	∧ ^2.internal_type not in {File}<br>	∧ ^2.roles not in {TYPE}<br>⇒ y = ⏎<br>Confidence: 0.938. Support: 168.` |
| 62 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, :}<br>	∧ -2.reserved = =<br>	∧ -3.label in {<space>}<br>	∧ -4.diff_offset ≤ 4<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.roles not in {MAP}<br>	∧ +2.reserved not in {=}<br>	∧ ^1.roles not in {DECLARATION, STATEMENT}<br>	∧ ^2.internal_type not in {File}<br>	∧ ^2.roles not in {TYPE}<br>⇒ y = ␣<br>Confidence: 0.972. Support: 198.` |
| 63 | `  •••start_col ≤ 36<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, :}<br>	∧ -2.reserved not in {=}<br>	∧ -3.label in {<space>}<br>	∧ -4.diff_offset ≤ 4<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.roles not in {MAP}<br>	∧ +2.reserved not in {=}<br>	∧ ^1.roles not in {DECLARATION, STATEMENT}<br>	∧ ^2.internal_type not in {File}<br>	∧ ^2.roles not in {TYPE}<br>⇒ y = ␣<br>Confidence: 0.856. Support: 94.` |
| 64 | `  -1.diff_offset ≥ 4<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, :}<br>	∧ -3.label not in {<space>}<br>	∧ -4.diff_offset ≤ 4<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.roles not in {MAP}<br>	∧ +2.reserved not in {=}<br>	∧ ^1.roles not in {DECLARATION, STATEMENT}<br>	∧ ^2.internal_type not in {File}<br>	∧ ^2.roles not in {TYPE}<br>⇒ y = ␣<br>Confidence: 0.982. Support: 194.` |
| 65 | `  -1.diff_offset ≤ 3<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = {<br>	∧ -3.label not in {<space>}<br>	∧ -4.diff_offset ≤ 4<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.roles not in {IDENTIFIER, MAP}<br>	∧ +2.reserved not in {=}<br>	∧ ^1.roles not in {DECLARATION, OPERATOR, STATEMENT}<br>	∧ ^2.internal_type not in {File}<br>	∧ ^2.roles not in {TYPE}<br>⇒ y = ⏎␣⁺␣⁺<br>Confidence: 0.919. Support: 118.` |
| 66 | `  -1.diff_offset ≤ 3<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, :, {}<br>	∧ -3.label not in {<space>}<br>	∧ -4.diff_offset ≤ 4<br>	∧ -4.reserved not in {=}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.roles not in {MAP}<br>	∧ +2.reserved not in {=, >}<br>	∧ ^1.roles not in {DECLARATION, OPERATOR, STATEMENT}<br>	∧ ^2.internal_type not in {File}<br>	∧ ^2.roles not in {TYPE}<br>⇒ y = ∅<br>Confidence: 0.899. Support: 790.` |

<details>
    <summary>Machine-readable report</summary>
```json
{"javascript": {"avg_rule_len": 10.621212121212121, "max_conf": 0.9993489384651184, "max_support": 26218, "min_conf": 0.8051643371582031, "min_support": 90, "num_rules": 66}}
```
</details>
