# Model report for file:///tmp/top-repos-quality-repos-1qjo6yjz/30-seconds-of-code HEAD 3a122c9cfcbdc091227879a06a32bc67ccd0d35d

### Dump

```json
{'created_at': datetime.datetime(2019, 1, 31, 11, 37, 54, 290355),
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
 'uuid': '0fdd63b8-bf53-4bbe-b157-94cbcb903cb4',
 'version': [1]}
style.format.analyzer.FormatAnalyzer/[1] file:///tmp/top-repos-quality-repos-1qjo6yjz/30-seconds-of-code 3a122c9cfcbdc091227879a06a32bc67ccd0d35d

# javascript
58 rules, avg.len. 8.1
## train
PPCR: 0.978421
### report
macro
{'f1-score': 0.8090825005561911,
 'precision': 0.8435301832742599,
 'recall': 0.779568824724395,
 'support': 42847}
micro
{'f1-score': 0.9625411347352206,
 'precision': 0.9625411347352206,
 'recall': 0.9625411347352206,
 'support': 42847}
weighted
{'f1-score': 0.9617870413342331,
 'precision': 0.9625579890717566,
 'recall': 0.9625411347352206,
 'support': 42847}
### report_full
macro
{'f1-score': 0.8040075904979809,
 'precision': 0.8435301832742599,
 'recall': 0.7699704914264529,
 'support': 43792}
micro
{'f1-score': 0.9520423827606506,
 'precision': 0.9625411347352206,
 'recall': 0.9417701863354038,
 'support': 43792}
weighted
{'f1-score': 0.9460982123437488,
 'precision': 0.9522314179299237,
 'recall': 0.9417701863354038,
 'support': 43792}
## test
PPCR: 0.977135
### report
macro
{'f1-score': 0.7975819296173337,
 'precision': 0.8335396621452323,
 'recall': 0.7677065454119358,
 'support': 10769}
micro
{'f1-score': 0.9583991085523261,
 'precision': 0.9583991085523261,
 'recall': 0.9583991085523261,
 'support': 10769}
weighted
{'f1-score': 0.9575394625976755,
 'precision': 0.9582765331291881,
 'recall': 0.9583991085523261,
 'support': 10769}
### report_full
macro
{'f1-score': 0.7923657400800854,
 'precision': 0.8335396621452323,
 'recall': 0.7577180709443826,
 'support': 11021}
micro
{'f1-score': 0.9473152822395595,
 'precision': 0.9583991085523261,
 'recall': 0.9364848924779966,
 'support': 11021}
weighted
{'f1-score': 0.9417767619475002,
 'precision': 0.9489577623954942,
 'recall': 0.9364848924779966,
 'support': 11021}
```

## javascript
### Summary
58 rules, avg.len. 8.1

| | |
|-|-|
|Min support|136|
|Max support|19229|
|Min confidence|0.8225806355476379|
|Max confidence|0.9997386336326599|

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
| 1 | `  -1.internal_type = StringLiteral<br>⇒ y = '<br>Confidence: 1.000. Support: 1913.` |
| 2 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = (<br>	∧ +1.internal_type = StringLiteral<br>⇒ y = '<br>Confidence: 0.996. Support: 1568.` |
| 3 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = ;<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = }<br>⇒ y = ⏎␣⁻␣⁻<br>Confidence: 0.998. Support: 917.` |
| 4 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = ;<br>	∧ -4.length ≥ 9<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {}}<br>⇒ y = ⏎⏎<br>Confidence: 0.941. Support: 229.` |
| 5 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = ;<br>	∧ -4.length ≤ 8<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {}}<br>⇒ y = ⏎<br>Confidence: 0.949. Support: 1325.` |
| 6 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = ,<br>	∧ +1.internal_type not in {StringLiteral}<br>⇒ y = ␣<br>Confidence: 0.922. Support: 2109.` |
| 7 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, ;}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = =<br>⇒ y = ␣<br>Confidence: 1.000. Support: 1727.` |
| 8 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, ;}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = {<br>⇒ y = ␣<br>Confidence: 0.929. Support: 1224.` |
| 9 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, ;}<br>	∧ -3.reserved = ><br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {=, {}<br>	∧ +2.reserved = (<br>⇒ y = ⏎␣⁺␣⁺<br>Confidence: 0.997. Support: 842.` |
| 10 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = {<br>	∧ -3.reserved not in {>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.roles in {EXPRESSION}<br>	∧ ^1.roles in {EXPRESSION}<br>⇒ y = ␣<br>Confidence: 0.892. Support: 180.` |
| 11 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, ;, {}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.roles in {EXPRESSION}<br>	∧ ^1.roles in {EXPRESSION, OPERATOR}<br>⇒ y = ␣<br>Confidence: 0.864. Support: 136.` |
| 12 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, ;, {}<br>	∧ -3.reserved not in {>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {=, {}<br>	∧ +1.roles in {EXPRESSION}<br>	∧ ^1.roles in {EXPRESSION} and not in {FUNCTION, OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.968. Support: 3342.` |
| 13 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, ;}<br>	∧ -3.reserved = const<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {=, {}<br>	∧ +1.roles in {EXPRESSION}<br>	∧ ^1.roles not in {EXPRESSION}<br>⇒ y = ∅<br>Confidence: 0.972. Support: 235.` |
| 14 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, ;}<br>	∧ -3.reserved not in {>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.roles in {EXPRESSION}<br>	∧ ^1.roles not in {EXPRESSION}<br>⇒ y = ␣<br>Confidence: 0.914. Support: 1571.` |
| 15 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, ;}<br>	∧ -3.reserved not in {>}<br>	∧ -4.roles not in {KEY}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {=, {}<br>	∧ +1.roles not in {EXPRESSION}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.970. Support: 16513.` |
| 16 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, ;}<br>	∧ -3.reserved = ><br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {=, {}<br>	∧ +1.roles in {CALL}<br>⇒ y = ⏎␣⁺␣⁺<br>Confidence: 0.999. Support: 790.` |
| 17 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = {<br>	∧ -3.reserved not in {>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.roles in {EXPRESSION}<br>	∧ ^1.roles in {CALL}<br>⇒ y = ␣<br>Confidence: 0.936. Support: 148.` |
| 18 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, ;, {}<br>	∧ -3.reserved not in {>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {=, {}<br>	∧ +1.roles in {EXPRESSION}<br>	∧ ^1.roles in {CALL}<br>⇒ y = ∅<br>Confidence: 0.939. Support: 3241.` |
| 19 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, ;}<br>	∧ -3.reserved not in {>}<br>	∧ -5.reserved = ;<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {=, {}<br>	∧ +1.roles in {EXPRESSION}<br>	∧ ^1.roles not in {CALL}<br>⇒ y = ∅<br>Confidence: 0.946. Support: 286.` |
| 20 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, ;}<br>	∧ -3.reserved not in {>}<br>	∧ -5.reserved not in {;}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {=, {}<br>	∧ +1.roles in {EXPRESSION}<br>	∧ ^1.roles in {IDENTIFIER} and not in {CALL}<br>⇒ y = ∅<br>Confidence: 0.997. Support: 161.` |
| 21 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, ;}<br>	∧ -3.reserved not in {>}<br>	∧ -5.reserved not in {;}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.roles in {EXPRESSION}<br>	∧ ^1.roles not in {CALL, IDENTIFIER}<br>⇒ y = ␣<br>Confidence: 0.898. Support: 1800.` |
| 22 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = ;<br>	∧ -4.length ≥ 8<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {}}<br>⇒ y = ⏎⏎<br>Confidence: 0.937. Support: 246.` |
| 23 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = ;<br>	∧ -4.length ≤ 7<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {}}<br>⇒ y = ⏎<br>Confidence: 0.953. Support: 1309.` |
| 24 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, ;}<br>	∧ -2.label in {<space>}<br>	∧ -3.reserved not in {>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.roles in {EXPRESSION}<br>	∧ +4.length ≤ 1<br>	∧ ^1.roles in {EXPRESSION} and not in {CALL}<br>⇒ y = ␣<br>Confidence: 0.877. Support: 158.` |
| 25 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, ;}<br>	∧ -3.reserved not in {>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.roles in {EXPRESSION}<br>	∧ +4.length ≤ 1<br>	∧ ^1.roles not in {CALL, EXPRESSION}<br>⇒ y = ␣<br>Confidence: 0.915. Support: 1475.` |
| 26 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, ;}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = {<br>	∧ +4.roles not in {MAP}<br>⇒ y = ␣<br>Confidence: 0.992. Support: 1122.` |
| 27 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, ;}<br>	∧ -3.reserved = ><br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {=, {}<br>	∧ ^2.roles in {STATEMENT}<br>⇒ y = ⏎␣⁺␣⁺<br>Confidence: 0.997. Support: 798.` |
| 28 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, ;}<br>	∧ -2.label not in {<space>}<br>	∧ -3.reserved not in {>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {=, {}<br>	∧ +1.roles in {EXPRESSION}<br>	∧ ^1.roles in {EXPRESSION} and not in {FUNCTION}<br>⇒ y = ∅<br>Confidence: 0.935. Support: 3463.` |
| 29 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, ;}<br>	∧ -3.reserved not in {>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {=, {}<br>	∧ +1.roles in {EXPRESSION}<br>	∧ +4.roles in {CALLEE}<br>	∧ ^1.roles not in {EXPRESSION}<br>⇒ y = ∅<br>Confidence: 0.972. Support: 232.` |
| 30 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, ;}<br>	∧ -3.reserved not in {>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.roles in {EXPRESSION}<br>	∧ +4.roles not in {CALLEE}<br>	∧ ^1.roles not in {EXPRESSION}<br>⇒ y = ␣<br>Confidence: 0.920. Support: 1515.` |
| 31 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, ;}<br>	∧ -3.reserved not in {>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.roles in {EXPRESSION}<br>	∧ +2.reserved = :<br>	∧ ^1.roles in {EXPRESSION}<br>⇒ y = ␣<br>Confidence: 0.881. Support: 155.` |
| 32 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, ;}<br>	∧ -3.reserved not in {>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.roles in {EXPRESSION}<br>	∧ ^1.roles in {EXPRESSION, OPERATOR}<br>⇒ y = ␣<br>Confidence: 0.908. Support: 146.` |
| 33 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, ;}<br>	∧ -3.reserved not in {>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {=, {}<br>	∧ +1.roles in {EXPRESSION}<br>	∧ +2.reserved not in {:}<br>	∧ ^1.roles in {EXPRESSION} and not in {FUNCTION, OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.968. Support: 3311.` |
| 34 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, ;}<br>	∧ -3.reserved not in {>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {=, {}<br>	∧ +1.roles in {EXPRESSION}<br>	∧ +4.roles in {CALL}<br>	∧ ^1.roles not in {EXPRESSION}<br>⇒ y = ∅<br>Confidence: 0.990. Support: 255.` |
| 35 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, ;}<br>	∧ -3.reserved not in {>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.roles in {EXPRESSION}<br>	∧ +4.roles not in {CALL}<br>	∧ ^1.roles not in {EXPRESSION}<br>⇒ y = ␣<br>Confidence: 0.901. Support: 1563.` |
| 36 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = {<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {=, {}<br>	∧ +1.roles in {MAP}<br>	∧ +4.roles in {CALL}<br>⇒ y = ∅<br>Confidence: 0.980. Support: 225.` |
| 37 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = {<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.roles in {MAP}<br>	∧ +4.roles not in {CALL}<br>⇒ y = ␣<br>Confidence: 0.823. Support: 217.` |
| 38 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = {<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {=, {}<br>	∧ +1.roles not in {MAP}<br>⇒ y = ⏎␣⁺␣⁺<br>Confidence: 0.990. Support: 891.` |
| 39 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, ;, {}<br>	∧ -4.reserved = const<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {=, {}<br>	∧ ^2.internal_type = VariableDeclaration<br>⇒ y = ∅<br>Confidence: 0.998. Support: 245.` |
| 40 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, ;, {}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ ^2.internal_type = VariableDeclaration<br>⇒ y = ␣<br>Confidence: 0.960. Support: 719.` |
| 41 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = const<br>	∧ +1.internal_type not in {StringLiteral}<br>⇒ y = ␣<br>Confidence: 0.999. Support: 368.` |
| 42 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = :<br>	∧ +1.internal_type not in {StringLiteral}<br>⇒ y = ␣<br>Confidence: 0.992. Support: 314.` |
| 43 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {;, const, {}<br>	∧ -3.reserved = :<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = }<br>⇒ y = ␣<br>Confidence: 0.921. Support: 146.` |
| 44 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, :, ;, const, {}<br>	∧ -2.label in {<space>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {=, {, }}<br>	∧ ^1.internal_type = ArrowFunctionExpression<br>	∧ ^1.roles not in {OPERATOR}<br>	∧ ^2.internal_type not in {VariableDeclaration}<br>⇒ y = ∅<br>Confidence: 0.978. Support: 202.` |
| 45 | `  -1.reserved not in {,, :, ;, const, {}<br>	∧ -1.length ≤ 1<br>	∧ +1.internal_type = Identifier<br>	∧ +1.reserved not in {=, {, }}<br>	∧ ^1.internal_type not in {ArrowFunctionExpression, File}<br>	∧ ^1.roles not in {OPERATOR, STATEMENT}<br>	∧ ^2.internal_type not in {VariableDeclaration}<br>⇒ y = ∅<br>Confidence: 0.987. Support: 2630.` |
| 46 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, :, ;, const, {}<br>	∧ +1.internal_type not in {Identifier, StringLiteral}<br>	∧ +1.reserved not in {=, {, }}<br>	∧ ^1.internal_type not in {ArrowFunctionExpression, File}<br>	∧ ^1.roles not in {OPERATOR, STATEMENT}<br>	∧ ^2.internal_type not in {VariableDeclaration}<br>⇒ y = ∅<br>Confidence: 0.991. Support: 15756.` |
| 47 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = {<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {=, {}<br>	∧ +1.roles in {MAP}<br>	∧ +4.length ≥ 7<br>⇒ y = ∅<br>Confidence: 0.971. Support: 226.` |
| 48 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = {<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.roles in {MAP}<br>	∧ +4.length ≤ 6<br>⇒ y = ␣<br>Confidence: 0.877. Support: 231.` |
| 49 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = =<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ ^2.internal_type = VariableDeclaration<br>⇒ y = ␣<br>Confidence: 0.992. Support: 703.` |
| 50 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, ;, =, {}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {=, {}<br>	∧ ^2.internal_type = VariableDeclaration<br>⇒ y = ∅<br>Confidence: 0.909. Support: 303.` |
| 51 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {;, const, {}<br>	∧ -5.diff_offset ≤ 6<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = }<br>⇒ y = ␣<br>Confidence: 0.858. Support: 137.` |
| 52 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, :, ;, const, {}<br>	∧ -2.label in {<space>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {=, {, }}<br>	∧ ^1.roles in {DECLARATION} and not in {OPERATOR}<br>	∧ ^2.internal_type not in {VariableDeclaration}<br>⇒ y = ∅<br>Confidence: 0.946. Support: 196.` |
| 53 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, :, ;, const, {}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {=, {, }}<br>	∧ ^1.roles not in {DECLARATION, OPERATOR}<br>	∧ ^2.internal_type not in {VariableDeclaration}<br>⇒ y = ∅<br>Confidence: 0.975. Support: 19229.` |
| 54 | `  -1.diff_offset ≤ 1<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, ;, {}<br>	∧ -3.reserved not in {>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {=, {}<br>	∧ +1.roles in {EXPRESSION}<br>	∧ ^1.roles in {EXPRESSION} and not in {FUNCTION, OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.987. Support: 3176.` |
| 55 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, ;}<br>	∧ -3.reserved = ><br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {=, {}<br>	∧ +1.roles in {CALLEE}<br>⇒ y = ⏎␣⁺␣⁺<br>Confidence: 0.999. Support: 814.` |
| 56 | `  -1.diff_offset ≤ 1<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, ;, {}<br>	∧ -2.diff_offset ≥ 5<br>	∧ -3.label in {<space>}<br>	∧ -3.reserved not in {>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {=, {}<br>	∧ +1.roles in {EXPRESSION}<br>	∧ ^1.roles in {EXPRESSION} and not in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.997. Support: 190.` |
| 57 | `  -1.diff_offset ≤ 1<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, ;, {}<br>	∧ -3.label not in {<space>}<br>	∧ -3.reserved not in {>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {=, {}<br>	∧ +1.roles in {EXPRESSION}<br>	∧ ^1.roles in {EXPRESSION} and not in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.986. Support: 3040.` |
| 58 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, ;}<br>	∧ -3.reserved not in {>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {=, {}<br>	∧ +1.roles not in {EXPRESSION}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.965. Support: 16469.` |

<details>
    <summary>Machine-readable report</summary>
```json
{"javascript": {"avg_rule_len": 8.10344827586207, "max_conf": 0.9997386336326599, "max_support": 19229, "min_conf": 0.8225806355476379, "min_support": 136, "num_rules": 58}}
```
</details>
