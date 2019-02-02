# Model report for file:///tmp/top-repos-quality-repos-hda3ai6m/freeCodeCamp HEAD cf65516cce60645a417e44c4fcea7418ca920572

### Dump

```json
{'created_at': datetime.datetime(2019, 1, 31, 11, 50, 37, 700231),
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
 'uuid': 'ee1a53c7-3637-4674-b804-c9d61157e5f3',
 'version': [1]}
style.format.analyzer.FormatAnalyzer/[1] file:///tmp/top-repos-quality-repos-hda3ai6m/freeCodeCamp cf65516cce60645a417e44c4fcea7418ca920572

# javascript
74 rules, avg.len. 9.9
## train
PPCR: 0.910841
### report
macro
{'f1-score': 0.6953068829163643,
 'precision': 0.7168545913145257,
 'recall': 0.6775420257169813,
 'support': 90104}
micro
{'f1-score': 0.9515781763295748,
 'precision': 0.9515781763295748,
 'recall': 0.9515781763295748,
 'support': 90104}
weighted
{'f1-score': 0.9481489999110467,
 'precision': 0.9464716343566157,
 'recall': 0.9515781763295748,
 'support': 90104}
### report_full
macro
{'f1-score': 0.6271572202643334,
 'precision': 0.7168545913145257,
 'recall': 0.5673540979773806,
 'support': 98924}
micro
{'f1-score': 0.9071777726051168,
 'precision': 0.9515781763295748,
 'recall': 0.8667360802232017,
 'support': 98924}
weighted
{'f1-score': 0.8986206757648334,
 'precision': 0.9429907372720785,
 'recall': 0.8667360802232017,
 'support': 98924}
## test
PPCR: 0.904415
### report
macro
{'f1-score': 0.6901949206470775,
 'precision': 0.7162723978994432,
 'recall': 0.6691859054298318,
 'support': 20712}
micro
{'f1-score': 0.9510428736964078,
 'precision': 0.9510428736964078,
 'recall': 0.9510428736964078,
 'support': 20712}
weighted
{'f1-score': 0.9484725385506224,
 'precision': 0.9473981863152069,
 'recall': 0.9510428736964078,
 'support': 20712}
### report_full
macro
{'f1-score': 0.6077473746740902,
 'precision': 0.7162723978994432,
 'recall': 0.5441627570926612,
 'support': 22901}
micro
{'f1-score': 0.9033086465044825,
 'precision': 0.9510428736964078,
 'recall': 0.8601371119165102,
 'support': 22901}
weighted
{'f1-score': 0.8943022966355582,
 'precision': 0.9435845654096415,
 'recall': 0.8601371119165102,
 'support': 22901}
```

## javascript
### Summary
74 rules, avg.len. 9.9

| | |
|-|-|
|Min support|90|
|Max support|19974|
|Min confidence|0.8023256063461304|
|Max confidence|0.9997157454490662|

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
| 1 | `  -1.internal_type = StringLiteral<br>	∧ -4.length ≥ 8<br>	∧ +1.internal_type = JSXIdentifier<br>⇒ y = '⏎<br>Confidence: 0.998. Support: 214.` |
| 2 | `  -1.internal_type = StringLiteral<br>	∧ +1.internal_type not in {JSXIdentifier}<br>	∧ ^1.roles in {OPERATOR}<br>⇒ y = '␣<br>Confidence: 0.906. Support: 176.` |
| 3 | `  -1.internal_type = StringLiteral<br>	∧ -3.roles in {UNANNOTATED}<br>	∧ -5.roles in {UNANNOTATED}<br>	∧ +1.internal_type not in {JSXIdentifier}<br>	∧ +1.reserved not in {}}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = '<br>Confidence: 0.907. Support: 124.` |
| 4 | `  -1.internal_type = StringLiteral<br>	∧ -3.roles not in {UNANNOTATED}<br>	∧ +1.internal_type not in {JSXIdentifier}<br>	∧ +1.reserved not in {}}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = '<br>Confidence: 0.935. Support: 2778.` |
| 5 | `  -1.internal_type not in {StringLiteral}<br>	∧ +1.length ≥ 2<br>	∧ ^1.roles in {IDENTIFIER}<br>⇒ y = ∅<br>Confidence: 0.996. Support: 4486.` |
| 6 | `  -1.internal_type not in {StringLiteral}<br>	∧ +1.internal_type = StringLiteral<br>	∧ +1.length ≥ 2<br>	∧ +2.reserved = ;<br>	∧ ^1.roles not in {IDENTIFIER}<br>⇒ y = ␣'<br>Confidence: 0.990. Support: 969.` |
| 7 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = :<br>	∧ +1.internal_type = StringLiteral<br>	∧ +1.length ≥ 2<br>	∧ +2.reserved not in {;}<br>	∧ ^1.roles not in {IDENTIFIER}<br>⇒ y = ␣'<br>Confidence: 0.983. Support: 545.` |
| 8 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = ,<br>	∧ -3.length ≥ 2<br>	∧ +1.internal_type = StringLiteral<br>	∧ +1.length ≥ 2<br>	∧ +2.reserved not in {;}<br>	∧ ^1.roles not in {IDENTIFIER}<br>⇒ y = ⏎<br>Confidence: 0.935. Support: 283.` |
| 9 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = ,<br>	∧ -3.length ≤ 1<br>	∧ +1.internal_type = StringLiteral<br>	∧ +1.length ≥ 2<br>	∧ +2.reserved not in {;}<br>	∧ ^1.roles not in {IDENTIFIER}<br>⇒ y = ␣'<br>Confidence: 0.886. Support: 189.` |
| 10 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, :}<br>	∧ -2.label in {<space>}<br>	∧ -3.internal_type = Identifier<br>	∧ +1.internal_type = StringLiteral<br>	∧ +1.length ≥ 2<br>	∧ +2.reserved not in {;}<br>	∧ ^1.roles not in {IDENTIFIER}<br>⇒ y = ␣'<br>Confidence: 0.949. Support: 128.` |
| 11 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, :}<br>	∧ -2.label not in {<space>}<br>	∧ +1.internal_type = StringLiteral<br>	∧ +1.length ≥ 2<br>	∧ +2.reserved not in {;}<br>	∧ ^1.roles not in {IDENTIFIER}<br>⇒ y = '<br>Confidence: 0.917. Support: 1587.` |
| 12 | `  -1.internal_type not in {StringLiteral}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.length ≥ 2<br>	∧ ^1.internal_type = JSXElement<br>	∧ ^1.roles not in {IDENTIFIER}<br>⇒ y = ∅<br>Confidence: 0.999. Support: 2414.` |
| 13 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = (<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.length ≥ 2<br>	∧ ^1.internal_type not in {JSXElement}<br>	∧ ^1.roles not in {IDENTIFIER}<br>⇒ y = ∅<br>Confidence: 0.885. Support: 2382.` |
| 14 | `  •••start_col ≥ 7<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = ;<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = export<br>	∧ +1.length ≥ 2<br>	∧ +2.length ≥ 6<br>	∧ ^1.internal_type not in {JSXElement}<br>	∧ ^1.roles not in {IDENTIFIER}<br>⇒ y = ⏎⏎<br>Confidence: 0.963. Support: 94.` |
| 15 | `  •••start_col ≥ 7<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = ;<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {export}<br>	∧ +1.length ≥ 6<br>	∧ +3.length ≥ 2<br>	∧ ^1.internal_type not in {BlockStatement, JSXElement}<br>	∧ ^1.roles not in {IDENTIFIER}<br>⇒ y = ⏎<br>Confidence: 0.815. Support: 694.` |
| 16 | `  •••start_col ≥ 7<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = ;<br>	∧ -5.length ≥ 4<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {export}<br>	∧ +1.length ≥ 2<br>	∧ +3.length ≤ 1<br>	∧ ^1.internal_type not in {BlockStatement, JSXElement}<br>	∧ ^1.roles not in {IDENTIFIER}<br>⇒ y = ⏎⏎<br>Confidence: 0.934. Support: 159.` |
| 17 | `  •••start_col ≤ 6<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = ;<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.length ≥ 2<br>	∧ ^1.internal_type not in {JSXElement}<br>	∧ ^1.roles not in {IDENTIFIER}<br>⇒ y = ⏎⏎<br>Confidence: 0.808. Support: 321.` |
| 18 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = {<br>	∧ -3.roles in {UNANNOTATED}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.length ≥ 2<br>	∧ +2.reserved = }<br>	∧ ^1.internal_type not in {JSXElement}<br>	∧ ^1.roles not in {IDENTIFIER}<br>⇒ y = ∅<br>Confidence: 0.980. Support: 280.` |
| 19 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = {<br>	∧ -3.roles not in {UNANNOTATED}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.length ≥ 2<br>	∧ +2.reserved = }<br>	∧ ^1.internal_type not in {JSXElement}<br>	∧ ^1.roles not in {IDENTIFIER}<br>⇒ y = ␣<br>Confidence: 0.952. Support: 506.` |
| 20 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = {<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.length ≥ 2<br>	∧ +2.reserved not in {}}<br>	∧ ^1.internal_type not in {JSXElement}<br>	∧ ^1.roles in {UNANNOTATED} and not in {IDENTIFIER}<br>⇒ y = ∅<br>Confidence: 0.850. Support: 143.` |
| 21 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = {<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.length ≥ 2<br>	∧ +2.reserved = ,<br>	∧ +5.roles not in {EXPRESSION}<br>	∧ ^1.internal_type not in {JSXElement}<br>	∧ ^1.roles not in {IDENTIFIER, UNANNOTATED}<br>⇒ y = ␣<br>Confidence: 0.836. Support: 174.` |
| 22 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = {<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.length ≥ 2<br>	∧ +2.reserved not in {,, }}<br>	∧ ^1.internal_type not in {JSXElement}<br>	∧ ^1.roles not in {IDENTIFIER, UNANNOTATED}<br>⇒ y = ⏎␣⁺␣⁺<br>Confidence: 0.901. Support: 2093.` |
| 23 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = ,<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.length ≥ 2<br>	∧ ^1.internal_type = ObjectExpression<br>	∧ ^1.roles not in {IDENTIFIER}<br>⇒ y = ⏎<br>Confidence: 0.924. Support: 1414.` |
| 24 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {(, ,, ;, {}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.length ≥ 2<br>	∧ ^1.internal_type = ObjectExpression<br>	∧ ^1.roles not in {IDENTIFIER}<br>⇒ y = ∅<br>Confidence: 0.995. Support: 106.` |
| 25 | `  •••start_col ≥ 6<br>	∧ -1.internal_type = CommentLine<br>	∧ -1.reserved not in {(, ;, {}<br>	∧ +1.length ≥ 2<br>	∧ ^1.internal_type not in {JSXElement, ObjectExpression}<br>	∧ ^1.roles not in {IDENTIFIER}<br>⇒ y = ⏎<br>Confidence: 0.946. Support: 640.` |
| 26 | `  •••start_col ≥ 6<br>	∧ -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved = ,<br>	∧ -2.roles not in {EXPRESSION}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.length ≥ 2<br>	∧ ^1.internal_type not in {JSXElement, ObjectExpression}<br>	∧ ^1.roles in {LITERAL} and not in {IDENTIFIER}<br>⇒ y = ⏎<br>Confidence: 0.952. Support: 94.` |
| 27 | `  •••start_col ≥ 6<br>	∧ -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {(, ,, ;, {}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.length ≥ 2<br>	∧ ^1.internal_type not in {JSXElement, ObjectExpression}<br>	∧ ^1.roles in {LITERAL} and not in {IDENTIFIER}<br>⇒ y = ∅<br>Confidence: 0.984. Support: 474.` |
| 28 | `  •••start_col ≥ 6<br>	∧ -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {(, ;, {}<br>	∧ -1.length ≥ 4<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.length ≥ 2<br>	∧ ^1.internal_type not in {JSXElement, ObjectExpression}<br>	∧ ^1.roles not in {IDENTIFIER, LITERAL}<br>⇒ y = ␣<br>Confidence: 0.957. Support: 3920.` |
| 29 | `  •••start_col ≥ 6<br>	∧ -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {(, ;, {, }}<br>	∧ -1.length ≤ 3<br>	∧ -2.internal_type not in {Identifier}<br>	∧ -3.label in {<newline>}<br>	∧ -4.reserved = {<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.roles in {EXPRESSION}<br>	∧ +1.length ≥ 2<br>	∧ ^1.internal_type not in {JSXElement, ObjectExpression}<br>	∧ ^1.roles not in {IDENTIFIER, LITERAL}<br>⇒ y = ∅<br>Confidence: 0.855. Support: 141.` |
| 30 | `  •••start_col ≥ 6<br>	∧ -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {(, ;, {, }}<br>	∧ -1.length ≤ 3<br>	∧ -3.label in {<newline>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.roles not in {EXPRESSION}<br>	∧ +1.length ≥ 2<br>	∧ ^1.internal_type not in {JSXElement, ObjectExpression}<br>	∧ ^1.roles not in {IDENTIFIER, LITERAL}<br>⇒ y = ∅<br>Confidence: 0.985. Support: 296.` |
| 31 | `  •••start_col ≥ 6<br>	∧ -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {(, ;, {}<br>	∧ -1.length ≤ 3<br>	∧ -2.reserved not in {(}<br>	∧ -3.label not in {<newline>}<br>	∧ -4.label in {<newline>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.length ≥ 2<br>	∧ ^1.internal_type = ObjectPattern<br>	∧ ^1.roles not in {IDENTIFIER, LITERAL}<br>⇒ y = ⏎<br>Confidence: 0.978. Support: 206.` |
| 32 | `  •••start_col ≥ 6<br>	∧ -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {(, ;, {}<br>	∧ -1.length ≤ 3<br>	∧ -2.reserved not in {(}<br>	∧ -3.label not in {<newline>}<br>	∧ -4.label not in {<newline>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.roles in {VALUE}<br>	∧ +1.length ≥ 2<br>	∧ ^1.internal_type = ObjectPattern<br>	∧ ^1.roles not in {IDENTIFIER, LITERAL}<br>⇒ y = ␣<br>Confidence: 0.895. Support: 186.` |
| 33 | `  •••start_col ≤ 20<br>	∧ -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {(, ;, {}<br>	∧ -1.length ≤ 3<br>	∧ -2.reserved not in {(}<br>	∧ -3.label not in {<newline>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.length ≥ 2<br>	∧ +2.reserved = =<br>	∧ ^1.internal_type not in {JSXElement, ObjectExpression, ObjectPattern}<br>	∧ ^1.roles in {UNANNOTATED} and not in {IDENTIFIER, LITERAL}<br>⇒ y = ␣<br>Confidence: 0.854. Support: 161.` |
| 34 | `  •••start_col ≥ 6<br>	∧ -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {(, ;, {}<br>	∧ -1.length ≤ 3<br>	∧ -2.reserved not in {(}<br>	∧ -3.label not in {<newline>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.length ≥ 2<br>	∧ +2.reserved not in {=}<br>	∧ ^1.internal_type not in {JSXElement, ObjectExpression, ObjectPattern}<br>	∧ ^1.roles in {UNANNOTATED} and not in {IDENTIFIER, LITERAL}<br>⇒ y = ∅<br>Confidence: 0.972. Support: 123.` |
| 35 | `  •••start_col ≥ 6<br>	∧ -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved = ,<br>	∧ -1.length ≤ 3<br>	∧ -3.diff_offset ≥ 5<br>	∧ -3.label not in {<newline>}<br>	∧ -4.label in {<newline>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.length ≥ 2<br>	∧ ^1.internal_type not in {JSXElement, ObjectExpression}<br>	∧ ^1.roles not in {IDENTIFIER, LITERAL, UNANNOTATED}<br>⇒ y = ⏎<br>Confidence: 0.886. Support: 180.` |
| 36 | `  •••start_col ≥ 6<br>	∧ -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved = ,<br>	∧ -1.length ≤ 3<br>	∧ -3.diff_offset ≥ 5<br>	∧ -3.label not in {<newline>}<br>	∧ -4.label not in {<newline>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.length ≥ 2<br>	∧ ^1.internal_type not in {JSXElement, ObjectExpression, ObjectPattern}<br>	∧ ^1.roles not in {IDENTIFIER, LITERAL, UNANNOTATED}<br>⇒ y = ␣<br>Confidence: 0.891. Support: 758.` |
| 37 | `  •••start_col ≥ 6<br>	∧ -1.diff_offset ≥ 2<br>	∧ -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {(, ,, ;, {}<br>	∧ -1.length ≤ 3<br>	∧ -2.reserved not in {(}<br>	∧ -3.diff_offset ≥ 5<br>	∧ -3.label not in {<newline>}<br>	∧ -4.diff_col ≥ 7<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.length ≥ 2<br>	∧ ^1.internal_type not in {JSXElement, ObjectExpression, ObjectPattern}<br>	∧ ^1.roles not in {IDENTIFIER, LITERAL, UNANNOTATED}<br>⇒ y = ␣<br>Confidence: 0.857. Support: 381.` |
| 38 | `  •••start_col ≥ 6<br>	∧ -1.diff_offset ≤ 1<br>	∧ -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {(, ,, ;, {}<br>	∧ -1.length ≤ 3<br>	∧ -2.reserved not in {(}<br>	∧ -3.diff_offset ≥ 5<br>	∧ -3.label not in {<newline>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.length ≥ 2<br>	∧ ^1.internal_type not in {JSXElement, ObjectExpression, ObjectPattern}<br>	∧ ^1.roles not in {IDENTIFIER, LITERAL, UNANNOTATED}<br>⇒ y = ␣<br>Confidence: 0.972. Support: 2767.` |
| 39 | `  •••start_col ≥ 6<br>	∧ -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {(, ;, {}<br>	∧ -1.length ≤ 3<br>	∧ -2.reserved not in {(}<br>	∧ -3.diff_offset ≤ 4<br>	∧ -3.label not in {<newline>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.length ≥ 2<br>	∧ ^1.internal_type not in {JSXElement, ObjectExpression, ObjectPattern}<br>	∧ ^1.roles not in {IDENTIFIER, LITERAL, UNANNOTATED}<br>	∧ ^2.roles in {LITERAL}<br>⇒ y = ∅<br>Confidence: 0.893. Support: 117.` |
| 40 | `  •••start_col ≥ 6<br>	∧ -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved = =<br>	∧ -1.length ≤ 3<br>	∧ -3.diff_offset ≤ 4<br>	∧ -3.label not in {<newline>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.length ≥ 2<br>	∧ ^1.internal_type not in {JSXElement, ObjectExpression, ObjectPattern}<br>	∧ ^1.roles not in {IDENTIFIER, LITERAL, UNANNOTATED}<br>	∧ ^2.roles not in {LITERAL}<br>⇒ y = ␣<br>Confidence: 0.994. Support: 266.` |
| 41 | `  •••start_col ≥ 6<br>	∧ -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {(, ,, ;, =, {}<br>	∧ -1.length ≤ 3<br>	∧ -2.reserved not in {(, =}<br>	∧ -3.diff_offset ≤ 4<br>	∧ -3.label not in {<newline>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.length ≥ 7<br>	∧ ^1.internal_type not in {JSXElement, ObjectExpression, ObjectPattern}<br>	∧ ^1.roles not in {IDENTIFIER, LITERAL, UNANNOTATED}<br>	∧ ^2.roles not in {LITERAL}<br>⇒ y = ∅<br>Confidence: 0.802. Support: 129.` |
| 42 | `  •••start_col ≤ 5<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = }<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.length ≥ 2<br>	∧ ^1.internal_type not in {JSXElement, ObjectExpression}<br>	∧ ^1.roles not in {IDENTIFIER, STATEMENT}<br>⇒ y = ⏎⏎<br>Confidence: 0.939. Support: 320.` |
| 43 | `  •••start_col ≤ 5<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {(, ;, {, }}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.length ≥ 2<br>	∧ ^1.internal_type not in {JSXElement, ObjectExpression}<br>	∧ ^1.roles not in {IDENTIFIER}<br>⇒ y = ∅<br>Confidence: 0.934. Support: 931.` |
| 44 | `  -1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = =<br>	∧ +1.length ≤ 1<br>	∧ ^1.roles in {DECLARATION}<br>⇒ y = ␣<br>Confidence: 1.000. Support: 1759.` |
| 45 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = (<br>	∧ +1.reserved = {<br>	∧ +1.length ≤ 1<br>	∧ ^1.roles in {DECLARATION}<br>⇒ y = ∅<br>Confidence: 0.977. Support: 154.` |
| 46 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {(}<br>	∧ +1.reserved = {<br>	∧ +1.length ≤ 1<br>	∧ ^1.roles in {DECLARATION}<br>⇒ y = ␣<br>Confidence: 0.995. Support: 1059.` |
| 47 | `  -1.internal_type not in {StringLiteral}<br>	∧ -5.roles not in {IDENTIFIER}<br>	∧ +1.reserved = }<br>	∧ +1.length ≤ 1<br>	∧ ^1.roles in {DECLARATION}<br>⇒ y = ␣<br>Confidence: 0.904. Support: 690.` |
| 48 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.roles in {EXPRESSION}<br>	∧ +1.reserved not in {=, {, }}<br>	∧ +1.length ≤ 1<br>	∧ ^1.roles in {DECLARATION}<br>⇒ y = ∅<br>Confidence: 0.998. Support: 1489.` |
| 49 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.roles not in {EXPRESSION}<br>	∧ +1.reserved = (<br>	∧ +1.length ≤ 1<br>	∧ ^1.roles in {DECLARATION}<br>⇒ y = ␣<br>Confidence: 0.960. Support: 337.` |
| 50 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.roles not in {EXPRESSION}<br>	∧ -5.reserved = const<br>	∧ +1.reserved not in {(, =, {, }}<br>	∧ +1.length ≤ 1<br>	∧ ^1.roles in {DECLARATION}<br>⇒ y = ␣<br>Confidence: 0.965. Support: 101.` |
| 51 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.roles not in {EXPRESSION}<br>	∧ -5.reserved not in {const}<br>	∧ +1.reserved not in {(, =, {, }}<br>	∧ +1.length ≤ 1<br>	∧ ^1.roles in {DECLARATION}<br>⇒ y = ∅<br>Confidence: 0.815. Support: 1078.` |
| 52 | `  -1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = }<br>	∧ +1.length ≤ 1<br>	∧ ^1.roles in {UNANNOTATED} and not in {DECLARATION}<br>⇒ y = ∅<br>Confidence: 0.950. Support: 635.` |
| 53 | `  -1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = }<br>	∧ +1.length ≤ 1<br>	∧ +2.roles in {STRING}<br>	∧ ^1.roles not in {DECLARATION, UNANNOTATED}<br>⇒ y = ∅<br>Confidence: 0.977. Support: 154.` |
| 54 | `  •••start_col ≤ 32<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.roles in {EXPRESSION}<br>	∧ +1.reserved = }<br>	∧ +1.length ≤ 1<br>	∧ +2.roles not in {STRING}<br>	∧ ^1.roles not in {DECLARATION, UNANNOTATED}<br>⇒ y = ⏎␣⁻␣⁻<br>Confidence: 0.821. Support: 266.` |
| 55 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.roles not in {EXPRESSION}<br>	∧ +1.reserved = }<br>	∧ +1.length ≤ 1<br>	∧ +2.roles not in {STRING}<br>	∧ ^1.roles not in {DECLARATION, UNANNOTATED}<br>⇒ y = ⏎␣⁻␣⁻<br>Confidence: 0.912. Support: 1520.` |
| 56 | `  -1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = =<br>	∧ +1.length ≤ 1<br>	∧ ^1.roles in {OPERATOR} and not in {DECLARATION}<br>⇒ y = ␣<br>Confidence: 0.999. Support: 561.` |
| 57 | `  -1.internal_type not in {StringLiteral}<br>	∧ -2.diff_col ≤ 8<br>	∧ +1.reserved not in {=, }}<br>	∧ +1.length ≤ 1<br>	∧ +2.roles in {EXPRESSION}<br>	∧ ^1.roles in {OPERATOR} and not in {DECLARATION}<br>⇒ y = ␣<br>Confidence: 0.810. Support: 271.` |
| 58 | `  -1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = {<br>	∧ +1.length ≤ 1<br>	∧ ^1.roles in {UNANNOTATED} and not in {DECLARATION, OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.973. Support: 674.` |
| 59 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = (<br>	∧ +1.reserved = {<br>	∧ +1.length ≤ 1<br>	∧ ^1.roles not in {DECLARATION, OPERATOR, UNANNOTATED}<br>⇒ y = ∅<br>Confidence: 0.928. Support: 272.` |
| 60 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {(, ,}<br>	∧ +1.reserved = {<br>	∧ +1.length ≤ 1<br>	∧ ^1.roles not in {DECLARATION, OPERATOR, UNANNOTATED}<br>⇒ y = ␣<br>Confidence: 0.975. Support: 1347.` |
| 61 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = if<br>	∧ +1.reserved not in {{, }}<br>	∧ +1.length ≤ 1<br>	∧ ^1.roles not in {DECLARATION, OPERATOR}<br>⇒ y = ␣<br>Confidence: 0.982. Support: 255.` |
| 62 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = ,<br>	∧ +1.reserved not in {{, }}<br>	∧ +1.length ≤ 1<br>	∧ +2.reserved not in {>}<br>	∧ ^1.roles in {INITIALIZATION} and not in {DECLARATION, OPERATOR}<br>⇒ y = ⏎<br>Confidence: 0.882. Support: 131.` |
| 63 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = :<br>	∧ +1.reserved not in {{, }}<br>	∧ +1.length ≤ 1<br>	∧ ^1.roles not in {DECLARATION}<br>⇒ y = ␣<br>Confidence: 0.930. Support: 193.` |
| 64 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, :, if}<br>	∧ +1.reserved not in {{, }}<br>	∧ +1.length ≤ 1<br>	∧ ^1.internal_type = AssignmentPattern<br>	∧ ^1.roles not in {DECLARATION}<br>⇒ y = ␣<br>Confidence: 0.989. Support: 135.` |
| 65 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = return<br>	∧ +1.reserved not in {{, }}<br>	∧ +1.length ≤ 1<br>	∧ ^1.roles not in {DECLARATION}<br>⇒ y = ␣<br>Confidence: 0.994. Support: 90.` |
| 66 | `  •••start_col ≤ 9<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, :, if, return}<br>	∧ -2.diff_col ≥ 3<br>	∧ -3.label in {<newline>}<br>	∧ +1.reserved not in {{, }}<br>	∧ +1.roles not in {STRING}<br>	∧ +1.length ≤ 1<br>	∧ +2.reserved = ;<br>	∧ ^1.internal_type not in {ConditionalExpression}<br>	∧ ^1.roles not in {DECLARATION, OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.884. Support: 168.` |
| 67 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, :, if, return}<br>	∧ -2.diff_col ≥ 3<br>	∧ -3.label not in {<newline>}<br>	∧ +1.reserved not in {{, }}<br>	∧ +1.roles not in {STRING}<br>	∧ +1.length ≤ 1<br>	∧ +2.reserved = ;<br>	∧ ^1.internal_type not in {ConditionalExpression}<br>	∧ ^1.roles not in {DECLARATION, OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.932. Support: 1032.` |
| 68 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, :, >, if, return}<br>	∧ -2.diff_col ≥ 3<br>	∧ +1.reserved not in {{, }}<br>	∧ +1.roles not in {STRING}<br>	∧ +1.length ≤ 1<br>	∧ +2.reserved not in {;, >}<br>	∧ ^1.internal_type not in {ConditionalExpression}<br>	∧ ^1.roles not in {DECLARATION, OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.978. Support: 19974.` |
| 69 | `  •••start_col ≥ 50<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = )<br>	∧ -2.diff_col ≤ 2<br>	∧ +1.reserved = .<br>	∧ +1.roles not in {STRING}<br>	∧ +1.length ≤ 1<br>	∧ ^1.internal_type not in {ConditionalExpression}<br>	∧ ^1.roles not in {DECLARATION, OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.844. Support: 125.` |
| 70 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {), ,, :, >, if, return}<br>	∧ -2.diff_col ≤ 2<br>	∧ +1.reserved = .<br>	∧ +1.roles not in {STRING}<br>	∧ +1.length ≤ 1<br>	∧ +2.reserved not in {>}<br>	∧ ^1.internal_type not in {ConditionalExpression}<br>	∧ ^1.roles not in {DECLARATION, OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.985. Support: 102.` |
| 71 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {), ,, :, >, if, return}<br>	∧ -2.diff_col ≤ 2<br>	∧ +1.internal_type not in {Identifier}<br>	∧ +1.reserved = )<br>	∧ +1.roles not in {STRING}<br>	∧ +1.length ≤ 1<br>	∧ +2.internal_type not in {JSXIdentifier}<br>	∧ +2.reserved not in {), >}<br>	∧ ^1.internal_type not in {ConditionalExpression}<br>	∧ ^1.roles not in {DECLARATION, OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.980. Support: 330.` |
| 72 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, :, >, if, return}<br>	∧ -2.diff_col ≤ 2<br>	∧ +1.internal_type not in {Identifier}<br>	∧ +1.reserved not in {), {, }}<br>	∧ +1.roles not in {STRING}<br>	∧ +1.length ≤ 1<br>	∧ +2.internal_type not in {JSXIdentifier}<br>	∧ +2.reserved not in {), >}<br>	∧ ^1.internal_type not in {ConditionalExpression}<br>	∧ ^1.roles not in {DECLARATION, OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.968. Support: 2123.` |
| 73 | `  -1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {{, }}<br>	∧ +1.length = 0<br>	∧ +2.length ≥ 1<br>	∧ ^1.roles not in {DECLARATION, OPERATOR}<br>⇒ y = ␣'<br>Confidence: 0.805. Support: 105.` |
| 74 | `  -1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {{, }}<br>	∧ +1.length = 0<br>	∧ +2.length = 0<br>	∧ ^1.roles not in {DECLARATION, OPERATOR}<br>⇒ y = ⏎<br>Confidence: 0.980. Support: 226.` |

<details>
    <summary>Machine-readable report</summary>
```json
{"javascript": {"avg_rule_len": 9.932432432432432, "max_conf": 0.9997157454490662, "max_support": 19974, "min_conf": 0.8023256063461304, "min_support": 90, "num_rules": 74}}
```
</details>
