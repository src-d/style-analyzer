# Model report for file:///tmp/top-repos-quality-repos-rvax3_kv/create-react-app HEAD 32106d216e4c31fda30ec475f9f03186d116c893

### Dump

```json
{'created_at': datetime.datetime(2019, 1, 31, 11, 20, 35, 500464),
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
 'uuid': '67085601-34ab-45a9-8fac-2517349e3477',
 'version': [1]}
style.format.analyzer.FormatAnalyzer/[1] file:///tmp/top-repos-quality-repos-rvax3_kv/create-react-app 32106d216e4c31fda30ec475f9f03186d116c893

# javascript
22 rules, avg.len. 7.0
## train
PPCR: 0.873219
### report
macro
{'f1-score': 0.7849934602301616,
 'precision': 0.8136669518433663,
 'recall': 0.7660969876303383,
 'support': 14774}
micro
{'f1-score': 0.936442398808718,
 'precision': 0.936442398808718,
 'recall': 0.936442398808718,
 'support': 14774}
weighted
{'f1-score': 0.9333205976406108,
 'precision': 0.9346873227948315,
 'recall': 0.936442398808718,
 'support': 14774}
### report_full
macro
{'f1-score': 0.7284933891969796,
 'precision': 0.8136669518433663,
 'recall': 0.675465707667928,
 'support': 16919}
micro
{'f1-score': 0.8730634524973968,
 'precision': 0.936442398808718,
 'recall': 0.8177197233879071,
 'support': 16919}
weighted
{'f1-score': 0.8581988117023985,
 'precision': 0.9189263362421047,
 'recall': 0.8177197233879071,
 'support': 16919}
## test
PPCR: 0.908629
### report
macro
{'f1-score': 0.7991018689718453,
 'precision': 0.8130111639909869,
 'recall': 0.7944695323900224,
 'support': 2148}
micro
{'f1-score': 0.9450651769087524,
 'precision': 0.9450651769087524,
 'recall': 0.9450651769087524,
 'support': 2148}
weighted
{'f1-score': 0.9416651759368563,
 'precision': 0.9426846292594484,
 'recall': 0.9450651769087524,
 'support': 2148}
### report_full
macro
{'f1-score': 0.7532415019775845,
 'precision': 0.8130111639909869,
 'recall': 0.7162046438325713,
 'support': 2364}
micro
{'f1-score': 0.899822695035461,
 'precision': 0.9450651769087524,
 'recall': 0.8587140439932318,
 'support': 2364}
weighted
{'f1-score': 0.8877925988438191,
 'precision': 0.9304900077515337,
 'recall': 0.8587140439932318,
 'support': 2364}
```

## javascript
### Summary
22 rules, avg.len. 7.0

| | |
|-|-|
|Min support|97|
|Max support|5902|
|Min confidence|0.813524603843689|
|Max confidence|0.9992763996124268|

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
| 1 | `  -1.internal_type = StringLiteral<br>⇒ y = '<br>Confidence: 0.999. Support: 691.` |
| 2 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = (<br>	∧ +1.internal_type = StringLiteral<br>⇒ y = '<br>Confidence: 0.933. Support: 367.` |
| 3 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {(}<br>	∧ -2.label in {<newline>}<br>	∧ +1.internal_type = StringLiteral<br>⇒ y = '<br>Confidence: 0.996. Support: 131.` |
| 4 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {(}<br>	∧ -2.label in {'} and not in {<newline>}<br>	∧ +1.internal_type = StringLiteral<br>⇒ y = ⏎<br>Confidence: 0.974. Support: 97.` |
| 5 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {(}<br>	∧ -2.label not in {', <newline>}<br>	∧ +1.internal_type = StringLiteral<br>⇒ y = ␣'<br>Confidence: 0.877. Support: 288.` |
| 6 | `  -1.internal_type = CommentLine<br>⇒ y = ⏎<br>Confidence: 0.972. Support: 444.` |
| 7 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved = ;<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = }<br>⇒ y = ⏎␣⁻␣⁻<br>Confidence: 0.998. Support: 213.` |
| 8 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved = ,<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = }<br>⇒ y = ⏎␣⁻␣⁻<br>Confidence: 0.996. Support: 141.` |
| 9 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved = ,<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {}}<br>	∧ ^1.internal_type = CallExpression<br>⇒ y = ␣<br>Confidence: 0.831. Support: 192.` |
| 10 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved = ,<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {}}<br>	∧ +1.roles in {KEY}<br>	∧ ^1.internal_type not in {CallExpression}<br>⇒ y = ⏎<br>Confidence: 0.955. Support: 209.` |
| 11 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved = ,<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {}}<br>	∧ +2.length ≥ 3<br>	∧ ^1.internal_type not in {CallExpression}<br>⇒ y = ⏎<br>Confidence: 0.835. Support: 136.` |
| 12 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {,, ;}<br>	∧ -2.label in {<space>}<br>	∧ -3.diff_col ≥ 5<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ ^1.roles in {DECLARATION, FUNCTION}<br>⇒ y = ∅<br>Confidence: 0.819. Support: 113.` |
| 13 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {,, ;}<br>	∧ -2.label not in {<space>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ ^1.internal_type = ArrowFunctionExpression<br>	∧ ^1.roles in {DECLARATION, FUNCTION}<br>⇒ y = ␣<br>Confidence: 0.837. Support: 132.` |
| 14 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {,, ;}<br>	∧ -4.diff_offset ≥ 6<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ ^1.roles in {DECLARATION} and not in {FUNCTION}<br>⇒ y = ␣<br>Confidence: 0.985. Support: 565.` |
| 15 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved = :<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ ^1.roles not in {DECLARATION}<br>⇒ y = ␣<br>Confidence: 0.999. Support: 334.` |
| 16 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved = {<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ ^1.roles not in {DECLARATION}<br>⇒ y = ⏎␣⁺␣⁺<br>Confidence: 0.838. Support: 386.` |
| 17 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved = const<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ ^1.roles not in {DECLARATION}<br>⇒ y = ␣<br>Confidence: 0.997. Support: 198.` |
| 18 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {,, ;, const, {}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = {<br>	∧ ^1.roles not in {DECLARATION}<br>	∧ ^2.roles in {STATEMENT}<br>⇒ y = ␣<br>Confidence: 0.957. Support: 152.` |
| 19 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {,, :, ;, const, {}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {{}<br>	∧ +1.length ≤ 2<br>	∧ +2.reserved not in {>}<br>	∧ ^1.roles not in {DECLARATION, OPERATOR}<br>	∧ ^2.roles in {FILE}<br>⇒ y = ∅<br>Confidence: 0.814. Support: 244.` |
| 20 | `  -1.diff_offset ≥ 2<br>	∧ -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {,, ;, const, {}<br>	∧ -4.diff_line ≥ 1<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ ^1.roles in {STATEMENT} and not in {DECLARATION}<br>	∧ ^2.roles not in {FILE}<br>⇒ y = ␣<br>Confidence: 0.903. Support: 108.` |
| 21 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {,, :, ;, const, {}<br>	∧ -4.diff_line = 0<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {{}<br>	∧ +2.reserved not in {>}<br>	∧ ^1.roles in {STATEMENT} and not in {DECLARATION, OPERATOR}<br>	∧ ^2.roles not in {FILE}<br>⇒ y = ∅<br>Confidence: 0.908. Support: 759.` |
| 22 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {,, :, ;, const, {}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {{}<br>	∧ +2.reserved not in {>}<br>	∧ ^1.roles not in {DECLARATION, OPERATOR, STATEMENT}<br>	∧ ^2.roles not in {FILE}<br>⇒ y = ∅<br>Confidence: 0.941. Support: 5902.` |

<details>
    <summary>Machine-readable report</summary>
```json
{"javascript": {"avg_rule_len": 7.045454545454546, "max_conf": 0.9992763996124268, "max_support": 5902, "min_conf": 0.813524603843689, "min_support": 97, "num_rules": 22}}
```
</details>
