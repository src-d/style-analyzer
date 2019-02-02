# Model report for file:///tmp/top-repos-quality-repos-8x5e9our/express HEAD b4eb1f59d39d801d7365c86b04500f16faeb0b1c

### Dump

```json
{'created_at': datetime.datetime(2019, 1, 31, 11, 42, 51, 748266),
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
 'uuid': '7bece49c-f3e1-4efd-ac55-8d88e1ca7719',
 'version': [1]}
style.format.analyzer.FormatAnalyzer/[1] file:///tmp/top-repos-quality-repos-8x5e9our/express b4eb1f59d39d801d7365c86b04500f16faeb0b1c

# javascript
33 rules, avg.len. 7.6
## train
PPCR: 0.933391
### report
macro
{'f1-score': 0.9428306313926123,
 'precision': 0.9530549200796903,
 'recall': 0.9344866180335555,
 'support': 65651}
micro
{'f1-score': 0.9701146974151194,
 'precision': 0.9701146974151194,
 'recall': 0.9701146974151194,
 'support': 65651}
weighted
{'f1-score': 0.9695070981738789,
 'precision': 0.9696186115040143,
 'recall': 0.9701146974151194,
 'support': 65651}
### report_full
macro
{'f1-score': 0.9053584595959511,
 'precision': 0.9530549200796903,
 'recall': 0.8680460748104473,
 'support': 70336}
micro
{'f1-score': 0.9366924779574518,
 'precision': 0.9701146974151194,
 'recall': 0.9054964740673339,
 'support': 70336}
weighted
{'f1-score': 0.9346680201392691,
 'precision': 0.968965610656942,
 'recall': 0.9054964740673339,
 'support': 70336}
## test
PPCR: 0.932930
### report
macro
{'f1-score': 0.9268776130602956,
 'precision': 0.9348790194226287,
 'recall': 0.9227862637655133,
 'support': 14202}
micro
{'f1-score': 0.9597239825376708,
 'precision': 0.9597239825376708,
 'recall': 0.9597239825376708,
 'support': 14202}
weighted
{'f1-score': 0.9586458740706317,
 'precision': 0.9592028378959658,
 'recall': 0.9597239825376708,
 'support': 14202}
### report_full
macro
{'f1-score': 0.8879519353809906,
 'precision': 0.9348790194226287,
 'recall': 0.8558710487290134,
 'support': 15223}
micro
{'f1-score': 0.9264231096006798,
 'precision': 0.9597239825376708,
 'recall': 0.8953557117519543,
 'support': 15223}
weighted
{'f1-score': 0.9232634601355898,
 'precision': 0.9584374680117346,
 'recall': 0.8953557117519543,
 'support': 15223}
```

## javascript
### Summary
33 rules, avg.len. 7.6

| | |
|-|-|
|Min support|107|
|Max support|24571|
|Min confidence|0.8062499761581421|
|Max confidence|0.9998205304145813|

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
                     'min_samples_split': 240,
                     'n_estimators': 10,
                     'prune_attributes': True,
                     'prune_branches_algorithms': ['reduced-error'],
                     'prune_dataset_ratio': 0.2,
                     'top_down_greedy_budget': [False, 0.5]}}
```

### Rules

| rule number | description |
|----:|:-----|
| 1 | `  -1.internal_type = StringLiteral<br>	∧ +1.reserved = }<br>⇒ y = '␣<br>Confidence: 0.996. Support: 123.` |
| 2 | `  -1.internal_type = StringLiteral<br>	∧ +1.reserved not in {}}<br>	∧ ^1.roles in {OPERATOR}<br>⇒ y = '␣<br>Confidence: 0.882. Support: 114.` |
| 3 | `  -1.internal_type = StringLiteral<br>	∧ +1.reserved not in {}}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = '<br>Confidence: 0.995. Support: 3615.` |
| 4 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = (<br>	∧ +1.roles in {STRING}<br>⇒ y = '<br>Confidence: 1.000. Support: 2786.` |
| 5 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {(}<br>	∧ +1.roles in {STRING}<br>⇒ y = ␣'<br>Confidence: 0.924. Support: 1101.` |
| 6 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = ,<br>	∧ +1.roles not in {STRING}<br>⇒ y = ␣<br>Confidence: 0.980. Support: 3057.` |
| 7 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,}<br>	∧ -2.diff_col ≥ 3<br>	∧ +1.reserved = =<br>	∧ +1.roles not in {STRING}<br>⇒ y = ␣<br>Confidence: 1.000. Support: 1274.` |
| 8 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = var<br>	∧ -2.diff_col ≥ 3<br>	∧ +1.roles not in {STRING}<br>⇒ y = ␣<br>Confidence: 0.999. Support: 865.` |
| 9 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {var}<br>	∧ -2.diff_col ≥ 3<br>	∧ +1.roles not in {STRING}<br>	∧ ^1.roles in {OPERATOR}<br>⇒ y = ␣<br>Confidence: 0.947. Support: 409.` |
| 10 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, var}<br>	∧ -2.diff_col ≥ 3<br>	∧ -4.roles in {CALLEE}<br>	∧ +1.reserved not in {=}<br>	∧ +1.roles not in {STRING}<br>	∧ +4.roles in {STRING}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ⏎<br>Confidence: 0.955. Support: 438.` |
| 11 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, var}<br>	∧ -2.diff_col ≥ 3<br>	∧ -4.roles in {CALLEE}<br>	∧ +1.reserved not in {=}<br>	∧ +1.roles not in {STRING}<br>	∧ +4.roles not in {STRING}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.924. Support: 164.` |
| 12 | `  -1.internal_type = CommentLine<br>	∧ -1.reserved not in {,, var}<br>	∧ -2.diff_col ≥ 3<br>	∧ -4.roles not in {CALLEE}<br>	∧ +1.reserved not in {=, }}<br>	∧ +1.roles not in {STRING}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ⏎<br>Confidence: 0.853. Support: 329.` |
| 13 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved = if<br>	∧ -2.diff_col ≥ 3<br>	∧ -4.roles not in {CALLEE}<br>	∧ +1.reserved not in {}}<br>	∧ +1.roles not in {STRING}<br>⇒ y = ␣<br>Confidence: 0.992. Support: 196.` |
| 14 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved = return<br>	∧ -2.diff_col ≥ 3<br>	∧ -4.roles not in {CALLEE}<br>	∧ +1.reserved not in {}}<br>	∧ +1.roles not in {STRING}<br>⇒ y = ␣<br>Confidence: 0.997. Support: 181.` |
| 15 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved = :<br>	∧ -2.diff_col ≥ 3<br>	∧ -4.roles not in {CALLEE}<br>	∧ +1.reserved not in {}}<br>	∧ +1.roles not in {STRING}<br>⇒ y = ␣<br>Confidence: 0.959. Support: 181.` |
| 16 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -2.diff_col ≥ 3<br>	∧ -4.label in {<space>}<br>	∧ -4.roles not in {CALLEE}<br>	∧ +1.reserved = {<br>	∧ +1.roles not in {STRING}<br>⇒ y = ␣<br>Confidence: 0.997. Support: 187.` |
| 17 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -2.diff_col ≥ 3<br>	∧ -4.label not in {<space>}<br>	∧ -4.roles not in {CALLEE}<br>	∧ +1.reserved = {<br>	∧ +1.roles not in {STRING}<br>	∧ ^1.roles in {STATEMENT}<br>⇒ y = ␣<br>Confidence: 0.928. Support: 118.` |
| 18 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {,, :, var}<br>	∧ -2.diff_col ≥ 3<br>	∧ -4.label not in {<space>}<br>	∧ -4.roles not in {CALLEE}<br>	∧ -5.diff_col ≥ 15<br>	∧ +1.reserved = {<br>	∧ +1.roles not in {STRING}<br>	∧ ^1.roles not in {OPERATOR, STATEMENT}<br>⇒ y = ∅<br>Confidence: 0.931. Support: 440.` |
| 19 | `  •••start_col ≤ 33<br>	∧ -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {,, :, var}<br>	∧ -2.diff_col ≥ 3<br>	∧ -4.label not in {<space>}<br>	∧ -4.roles not in {CALLEE}<br>	∧ -5.diff_col ≤ 14<br>	∧ +1.reserved = {<br>	∧ +1.roles not in {STRING}<br>	∧ ^1.roles not in {OPERATOR, STATEMENT}<br>⇒ y = ∅<br>Confidence: 0.877. Support: 159.` |
| 20 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved = new<br>	∧ -2.diff_col ≥ 3<br>	∧ -4.roles not in {CALLEE}<br>	∧ +1.reserved not in {{, }}<br>	∧ +1.roles not in {STRING}<br>⇒ y = ␣<br>Confidence: 0.995. Support: 107.` |
| 21 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {,, :, ;, function, var}<br>	∧ -2.diff_col ≥ 3<br>	∧ -4.roles not in {CALLEE}<br>	∧ +1.reserved not in {=, {, }}<br>	∧ +1.roles not in {STRING}<br>	∧ +1.length ≤ 1<br>	∧ ^1.roles not in {OPERATOR}<br>	∧ ^2.internal_type = BlockStatement<br>⇒ y = ∅<br>Confidence: 0.953. Support: 286.` |
| 22 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {,, :, ;, function, var}<br>	∧ -2.diff_col ≥ 3<br>	∧ -4.roles not in {CALLEE}<br>	∧ +1.reserved not in {=, {, }}<br>	∧ +1.roles not in {STRING}<br>	∧ ^1.roles not in {OPERATOR}<br>	∧ ^2.internal_type not in {BlockStatement}<br>⇒ y = ∅<br>Confidence: 0.984. Support: 24571.` |
| 23 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = {<br>	∧ -2.diff_col ≤ 2<br>	∧ +1.roles not in {STRING}<br>	∧ +3.roles in {VALUE}<br>⇒ y = ␣<br>Confidence: 0.922. Support: 174.` |
| 24 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = {<br>	∧ -2.diff_col ≤ 2<br>	∧ +1.roles not in {STRING}<br>	∧ +3.roles not in {VALUE}<br>⇒ y = ⏎␣⁺␣⁺<br>Confidence: 0.975. Support: 1774.` |
| 25 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, {}<br>	∧ -2.diff_col ≤ 2<br>	∧ +1.reserved = }<br>	∧ +1.roles not in {STRING}<br>⇒ y = ⏎␣⁻␣⁻<br>Confidence: 0.970. Support: 1460.` |
| 26 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = =<br>	∧ -2.diff_col ≤ 2<br>	∧ +1.reserved not in {}}<br>	∧ +1.roles not in {STRING}<br>⇒ y = ␣<br>Confidence: 0.999. Support: 1296.` |
| 27 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = ;<br>	∧ -2.diff_col ≤ 2<br>	∧ -4.diff_col ≥ 9<br>	∧ +1.reserved not in {}}<br>	∧ +1.roles not in {STRING}<br>	∧ +3.length ≥ 2<br>⇒ y = ⏎⏎<br>Confidence: 0.905. Support: 833.` |
| 28 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = ;<br>	∧ -2.diff_col ≤ 2<br>	∧ -4.diff_col ≤ 8<br>	∧ +1.reserved not in {}}<br>	∧ +1.roles not in {STRING}<br>	∧ +3.length ≥ 2<br>	∧ +5.roles not in {EXPRESSION}<br>⇒ y = ⏎⏎<br>Confidence: 0.806. Support: 240.` |
| 29 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = ;<br>	∧ -2.diff_col ≤ 2<br>	∧ +1.reserved not in {}}<br>	∧ +1.roles not in {STRING}<br>	∧ +3.length ≤ 1<br>⇒ y = ⏎<br>Confidence: 0.841. Support: 393.` |
| 30 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, ;, =, {}<br>	∧ -2.diff_col ≤ 2<br>	∧ -3.roles in {ARGUMENT}<br>	∧ +1.reserved = .<br>	∧ +1.roles not in {STRING}<br>⇒ y = ⏎<br>Confidence: 0.929. Support: 743.` |
| 31 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, ;, =, {}<br>	∧ -2.diff_col ≤ 2<br>	∧ +1.reserved not in {., }}<br>	∧ +1.roles not in {STRING}<br>	∧ +5.reserved = function<br>⇒ y = ⏎⏎<br>Confidence: 0.965. Support: 499.` |
| 32 | `  -1.diff_col ≤ 2<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, ;, =, {}<br>	∧ -2.diff_col ≤ 2<br>	∧ -4.roles in {EXPRESSION}<br>	∧ -5.label in {<space>}<br>	∧ +1.reserved not in {., }}<br>	∧ +1.roles not in {STRING}<br>	∧ +3.reserved not in {=}<br>	∧ +5.reserved not in {function}<br>	∧ ^1.internal_type not in {BinaryExpression}<br>⇒ y = ∅<br>Confidence: 0.856. Support: 191.` |
| 33 | `  -1.diff_col ≤ 2<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, ;, =, {}<br>	∧ -2.diff_col ≤ 2<br>	∧ -5.label not in {<space>}<br>	∧ +1.reserved not in {., }}<br>	∧ +1.roles not in {STRING}<br>	∧ +3.reserved not in {=}<br>	∧ +5.reserved not in {function}<br>	∧ ^1.internal_type not in {BinaryExpression}<br>⇒ y = ∅<br>Confidence: 0.917. Support: 4060.` |

<details>
    <summary>Machine-readable report</summary>
```json
{"javascript": {"avg_rule_len": 7.575757575757576, "max_conf": 0.9998205304145813, "max_support": 24571, "min_conf": 0.8062499761581421, "min_support": 107, "num_rules": 33}}
```
</details>
