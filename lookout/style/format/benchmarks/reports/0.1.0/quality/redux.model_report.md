# Model report for file:///tmp/top-repos-quality-repos-7y4u1kvh/redux HEAD 902484ed735d38aec06683c847810a7218d8dba2

### Dump

```json
{'created_at': datetime.datetime(2019, 1, 31, 11, 25, 58, 645444),
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
 'uuid': '4787e5f3-9a0f-4292-8064-eac7f5149dff',
 'version': [1]}
style.format.analyzer.FormatAnalyzer/[1] file:///tmp/top-repos-quality-repos-7y4u1kvh/redux 902484ed735d38aec06683c847810a7218d8dba2

# javascript
31 rules, avg.len. 7.5
## train
PPCR: 0.904551
### report
macro
{'f1-score': 0.7320558123700469,
 'precision': 0.7391812714958012,
 'recall': 0.7271727255569198,
 'support': 27587}
micro
{'f1-score': 0.9319244571718563,
 'precision': 0.9319244571718563,
 'recall': 0.9319244571718563,
 'support': 27587}
weighted
{'f1-score': 0.9296489939402477,
 'precision': 0.9283567599118401,
 'recall': 0.9319244571718563,
 'support': 27587}
### report_full
macro
{'f1-score': 0.6737483746908509,
 'precision': 0.7391812714958012,
 'recall': 0.6317257351186272,
 'support': 30498}
micro
{'f1-score': 0.8852199363002496,
 'precision': 0.9319244571718563,
 'recall': 0.8429733097252279,
 'support': 30498}
weighted
{'f1-score': 0.8761186213237194,
 'precision': 0.9224450334593938,
 'recall': 0.8429733097252279,
 'support': 30498}
## test
PPCR: 0.904837
### report
macro
{'f1-score': 0.7253663912575381,
 'precision': 0.7387592060643485,
 'recall': 0.7136748023270829,
 'support': 7987}
micro
{'f1-score': 0.9397771378490046,
 'precision': 0.9397771378490046,
 'recall': 0.9397771378490046,
 'support': 7987}
weighted
{'f1-score': 0.9370864440465176,
 'precision': 0.9352878727190845,
 'recall': 0.9397771378490046,
 'support': 7987}
### report_full
macro
{'f1-score': 0.6482811335235519,
 'precision': 0.7387592060643485,
 'recall': 0.5961454572618483,
 'support': 8827}
micro
{'f1-score': 0.8928274057333174,
 'precision': 0.9397771378490046,
 'recall': 0.8503455307579019,
 'support': 8827}
weighted
{'f1-score': 0.8808114588544647,
 'precision': 0.9291307872754554,
 'recall': 0.8503455307579019,
 'support': 8827}
```

## javascript
### Summary
31 rules, avg.len. 7.5

| | |
|-|-|
|Min support|90|
|Max support|3540|
|Min confidence|0.8149651885032654|
|Max confidence|0.9995429515838623|

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
                     'min_samples_split': 190,
                     'n_estimators': 10,
                     'prune_attributes': True,
                     'prune_branches_algorithms': ['reduced-error'],
                     'prune_dataset_ratio': 0.2,
                     'top_down_greedy_budget': [False, 0.5]}}
```

### Rules

| rule number | description |
|----:|:-----|
| 1 | `  ^1.roles in {QUALIFIED}<br>⇒ y = ∅<br>Confidence: 0.976. Support: 3234.` |
| 2 | `  -1.reserved = (<br>	∧ +1.internal_type = StringLiteral<br>	∧ +1.length ≥ 2<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = '<br>Confidence: 0.976. Support: 315.` |
| 3 | `  -1.reserved not in {(}<br>	∧ +1.internal_type = StringLiteral<br>	∧ +1.length ≥ 2<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ␣'<br>Confidence: 0.877. Support: 696.` |
| 4 | `  -1.reserved = (<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.length ≥ 2<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ∅<br>Confidence: 0.936. Support: 815.` |
| 5 | `  -1.reserved = {<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.length ≥ 2<br>	∧ ^1.roles in {UNANNOTATED} and not in {QUALIFIED}<br>⇒ y = ∅<br>Confidence: 0.939. Support: 140.` |
| 6 | `  -1.reserved = {<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.length ≥ 2<br>	∧ +2.reserved = }<br>	∧ ^1.roles not in {QUALIFIED, UNANNOTATED}<br>⇒ y = ␣<br>Confidence: 0.922. Support: 174.` |
| 7 | `  -1.reserved = {<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.length ≥ 2<br>	∧ +2.reserved not in {,, }}<br>	∧ ^1.roles not in {QUALIFIED, UNANNOTATED}<br>⇒ y = ⏎␣⁺␣⁺<br>Confidence: 0.870. Support: 847.` |
| 8 | `  -1.reserved not in {(, {}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.length ≥ 2<br>	∧ ^1.internal_type = JSXElement<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ∅<br>Confidence: 0.996. Support: 364.` |
| 9 | `  -1.reserved not in {(, {}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.length ≥ 2<br>	∧ ^1.internal_type = ObjectExpression<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ⏎<br>Confidence: 0.872. Support: 490.` |
| 10 | `  •••start_col ≥ 6<br>	∧ -1.internal_type = StringLiteral<br>	∧ -1.reserved not in {(, {}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = import<br>	∧ +1.length ≥ 2<br>	∧ ^1.internal_type not in {JSXElement, ObjectExpression}<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = '⏎<br>Confidence: 0.988. Support: 200.` |
| 11 | `  •••start_col ≤ 13<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = )<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.length ≥ 2<br>	∧ ^1.internal_type not in {JSXElement, ObjectExpression}<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ⏎⏎<br>Confidence: 0.839. Support: 90.` |
| 12 | `  •••start_col ≥ 6<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {(, ), ,, ;, {}<br>	∧ -2.diff_offset ≤ 18<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {return}<br>	∧ +1.length ≥ 2<br>	∧ ^1.internal_type not in {BlockStatement, JSXElement, ObjectExpression}<br>	∧ ^1.roles in {EXPRESSION, OPERATOR} and not in {QUALIFIED}<br>⇒ y = ␣<br>Confidence: 0.885. Support: 222.` |
| 13 | `  •••start_col ≥ 6<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {(, ), ,, ;, {}<br>	∧ -2.diff_offset ≤ 18<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {return}<br>	∧ +1.roles not in {IDENTIFIER}<br>	∧ +1.length ≥ 2<br>	∧ ^1.internal_type not in {BlockStatement, JSXElement, ObjectExpression}<br>	∧ ^1.roles in {EXPRESSION} and not in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.898. Support: 93.` |
| 14 | `  •••start_col ≥ 6<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {(, ), ;, {}<br>	∧ -2.diff_offset ≤ 18<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {return}<br>	∧ +1.length ≥ 2<br>	∧ ^1.internal_type not in {BlockStatement, JSXElement, ObjectExpression}<br>	∧ ^1.roles not in {EXPRESSION, QUALIFIED}<br>⇒ y = ␣<br>Confidence: 0.899. Support: 2542.` |
| 15 | `  •••start_col ≤ 5<br>	∧ -1.reserved not in {(, {}<br>	∧ -3.diff_col ≥ 1<br>	∧ -3.diff_offset ≤ 4<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.length ≥ 2<br>	∧ ^1.internal_type not in {JSXElement, ObjectExpression}<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ⏎⏎<br>Confidence: 0.901. Support: 287.` |
| 16 | `  •••start_col ≤ 5<br>	∧ -1.reserved not in {(, {}<br>	∧ -3.diff_col = 0<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.length ≥ 2<br>	∧ ^1.internal_type not in {JSXElement, ObjectExpression}<br>⇒ y = ∅<br>Confidence: 0.986. Support: 178.` |
| 17 | `  -1.roles in {UNANNOTATED}<br>	∧ +1.reserved = =<br>	∧ +1.length ≤ 1<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ∅<br>Confidence: 0.997. Support: 161.` |
| 18 | `  -1.roles not in {UNANNOTATED}<br>	∧ +1.reserved = =<br>	∧ +1.length ≤ 1<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ␣<br>Confidence: 1.000. Support: 1094.` |
| 19 | `  -1.internal_type = StringLiteral<br>	∧ +1.reserved not in {=}<br>	∧ +1.length ≤ 1<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = '<br>Confidence: 0.915. Support: 604.` |
| 20 | `  -1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = }<br>	∧ +1.length ≤ 1<br>	∧ ^1.roles in {DECLARATION} and not in {QUALIFIED}<br>⇒ y = ␣<br>Confidence: 0.946. Support: 267.` |
| 21 | `  -1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = }<br>	∧ +1.length ≤ 1<br>	∧ ^1.roles in {UNANNOTATED} and not in {DECLARATION, QUALIFIED}<br>⇒ y = ∅<br>Confidence: 0.902. Support: 127.` |
| 22 | `  -1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = }<br>	∧ +1.length ≤ 1<br>	∧ ^1.roles not in {DECLARATION, QUALIFIED, UNANNOTATED}<br>⇒ y = ⏎␣⁻␣⁻<br>Confidence: 0.815. Support: 862.` |
| 23 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = (<br>	∧ +1.reserved = {<br>	∧ +1.length ≤ 1<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ∅<br>Confidence: 0.990. Support: 154.` |
| 24 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {(}<br>	∧ +1.reserved = {<br>	∧ +1.length ≤ 1<br>	∧ ^1.roles in {UNANNOTATED} and not in {QUALIFIED}<br>⇒ y = ∅<br>Confidence: 0.926. Support: 168.` |
| 25 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {(}<br>	∧ +1.reserved = {<br>	∧ +1.length ≤ 1<br>	∧ ^1.roles not in {QUALIFIED, UNANNOTATED}<br>⇒ y = ␣<br>Confidence: 0.923. Support: 990.` |
| 26 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = ,<br>	∧ +1.reserved not in {=, {, }}<br>	∧ +1.length ≤ 1<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ␣<br>Confidence: 0.892. Support: 246.` |
| 27 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = :<br>	∧ +1.reserved not in {=, {, }}<br>	∧ +1.length ≤ 1<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ␣<br>Confidence: 0.983. Support: 203.` |
| 28 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, :, >}<br>	∧ -1.roles in {EXPRESSION}<br>	∧ +1.reserved not in {=, {, }}<br>	∧ +1.length ≤ 1<br>	∧ ^1.roles not in {OPERATOR, VARIABLE}<br>⇒ y = ∅<br>Confidence: 0.987. Support: 3540.` |
| 29 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, :, >}<br>	∧ -1.roles not in {EXPRESSION}<br>	∧ +1.reserved = (<br>	∧ +1.length ≤ 1<br>	∧ ^1.roles in {FUNCTION} and not in {OPERATOR, QUALIFIED, VARIABLE}<br>⇒ y = ␣<br>Confidence: 0.895. Support: 128.` |
| 30 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, :, >}<br>	∧ -1.roles not in {EXPRESSION}<br>	∧ +1.reserved not in {(, =, {, }}<br>	∧ +1.length ≤ 1<br>	∧ ^1.roles not in {OPERATOR, VARIABLE}<br>⇒ y = ∅<br>Confidence: 0.899. Support: 2590.` |
| 31 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, :}<br>	∧ +1.reserved not in {=, {, }}<br>	∧ +1.length = 0<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ⏎<br>Confidence: 0.879. Support: 120.` |

<details>
    <summary>Machine-readable report</summary>
```json
{"javascript": {"avg_rule_len": 7.548387096774194, "max_conf": 0.9995429515838623, "max_support": 3540, "min_conf": 0.8149651885032654, "min_support": 90, "num_rules": 31}}
```
</details>
