# Model report for file:///tmp/top-repos-quality-repos-r4bui74f/webpack HEAD babe736cfa1ef7e8014ed32ba4a4ec38049dce14

### Dump

```json
{'created_at': '2019-04-09 05:09:10',
 'datasets': [],
 'dependencies': [],
 'description': 'Model bound to style.format.analyzer.FormatAnalyzer Lookout analyzer.',
 'environment': {'packages': 'ConfigArgParse==0.13.0 Jinja2==2.10 MarkupSafe==1.1.1 PyStemmer==1.3.0 PyYAML==5.1 Pympler==0.5 SQLAlchemy==1.2.10 SQLAlchemy-Utils==0.33.3 asdf==2.3.2 bblfsh==2.12.7 boto==2.49.0 boto3==1.9.130 botocore==1.12.130 cachetools==2.0.1 certifi==2019.3.9 chardet==3.0.4 clint==0.5.1 docker==3.7.1 docker-pycreds==0.4.0 dulwich==0.19.11 grpcio==1.19.0 grpcio-tools==1.19.0 humanfriendly==4.16.1 humanize==0.5.1 idna==2.8 jmespath==0.9.4 jsonschema==2.6.0 lookout-sdk==0.4.1 lookout-sdk-ml==0.18.1 lookout-style==0.1.1 lz4==2.1.6 modelforge==0.12.1 numpy==1.16.2 packaging==19.0 pandas==0.22.0 pip==19.0.3 protobuf==3.7.0 psycopg2-binary==2.7.5 pygtrie==2.3 pyparsing==2.3.1 python-dateutil==2.8.0 python-igraph==0.7.1.post6 pytz==2018.9 requests==2.21.0 requirements-parser==0.2.0 scikit-learn==0.20.1 scikit-optimize==0.5.2 scipy==1.2.1 semantic-version==2.6.0 setuptools==40.8.0 six==1.12.0 smart-open==1.8.0 sourced-ml==0.8.2 spdx==2.5.0 stringcase==1.2.0 tabulate==0.8.2 tqdm==4.31.1 '
                             'urllib3==1.24.1 websocket-client==0.56.0 xxhash==1.3.0',
                 'platform': 'Linux-4.15.15-coreos-x86_64-with-Ubuntu-18.04-bionic',
                 'python': '3.6.7 (default, Oct 22 2018, 11:32:17) [GCC 8.2.0]'},
 'license': 'ODbL-1.0',
 'metrics': {},
 'model': 'style.format.analyzer.FormatAnalyzer',
 'references': [],
 'series': 'Lookout',
 'size': '23.4 kB',
 'tags': [],
 'uuid': '83c51ac6-0218-4b61-9da6-f7a3d10b12f8',
 'vendor': 'source{d}',
 'version': [1]}
style.format.analyzer.FormatAnalyzer/[1] file:///tmp/top-repos-quality-repos-r4bui74f/webpack babe736cfa1ef7e8014ed32ba4a4ec38049dce14

# javascript
109 rules, avg.len. 10.3
## train
PPCR: 0.949290
### report
macro
{'f1-score': 0.5750142644959401,
 'precision': 0.6087973232740608,
 'recall': 0.5500374882143371,
 'support': 294259}
micro
{'f1-score': 0.9451707509370997,
 'precision': 0.9451707509370997,
 'recall': 0.9451707509370997,
 'support': 294259}
weighted
{'f1-score': 0.941214742423377,
 'precision': 0.939348955147577,
 'recall': 0.9451707509370997,
 'support': 294259}
### report_full
macro
{'f1-score': 0.5328019283661469,
 'precision': 0.6087973232740608,
 'recall': 0.48678463724561677,
 'support': 309978}
micro
{'f1-score': 0.9205824866732756,
 'precision': 0.9451707509370997,
 'recall': 0.8972410945292891,
 'support': 309978}
weighted
{'f1-score': 0.911773205855583,
 'precision': 0.9342884942953551,
 'recall': 0.8972410945292891,
 'support': 309978}
## test
PPCR: 0.946194
### report
macro
{'f1-score': 0.5238570235462552,
 'precision': 0.544431965144844,
 'recall': 0.5065344372836948,
 'support': 73084}
micro
{'f1-score': 0.9474440369985222,
 'precision': 0.9474440369985222,
 'recall': 0.9474440369985222,
 'support': 73084}
weighted
{'f1-score': 0.9453106092168684,
 'precision': 0.9451588744542627,
 'recall': 0.9474440369985222,
 'support': 73084}
### report_full
macro
{'f1-score': 0.485911075500153,
 'precision': 0.544431965144844,
 'recall': 0.4460242465327066,
 'support': 77240}
micro
{'f1-score': 0.9212500997844655,
 'precision': 0.9474440369985222,
 'recall': 0.8964655618850337,
 'support': 77240}
weighted
{'f1-score': 0.9159195724024126,
 'precision': 0.944174376001921,
 'recall': 0.8964655618850337,
 'support': 77240}
```

## javascript
### Summary
68 rules, avg.len. 10.6

| | |
|-|-|
|Min support|91|
|Max support|35271|
|Min confidence|0.9264164566993713|
|Max confidence|0.9993908405303955|

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
| 1 | `  -1.internal_type = StringLiteral<br>	∧ ^1.roles in {QUALIFIED}<br>⇒ y = "<br>Confidence: 0.942. Support: 718.` |
| 2 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {)}<br>	∧ -3.internal_type not in {StringLiteral}<br>	∧ -4.reserved = .<br>	∧ +1.reserved not in {}}<br>	∧ +2.length ≥ 4<br>	∧ +4.roles in {ARGUMENT}<br>	∧ +5.reserved = )<br>	∧ ^1.roles in {QUALIFIED}<br>⇒ y = ∅<br>Confidence: 0.996. Support: 113.` |
| 3 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {)}<br>	∧ -3.internal_type not in {StringLiteral}<br>	∧ -4.reserved = .<br>	∧ +1.reserved not in {}}<br>	∧ +4.roles in {ARGUMENT}<br>	∧ +5.reserved not in {)}<br>	∧ ^1.roles in {QUALIFIED}<br>⇒ y = ∅<br>Confidence: 0.990. Support: 346.` |
| 4 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {)}<br>	∧ -3.internal_type not in {StringLiteral}<br>	∧ -4.reserved not in {.}<br>	∧ +1.reserved not in {}}<br>	∧ +4.roles in {ARGUMENT}<br>	∧ ^1.roles in {QUALIFIED}<br>⇒ y = ∅<br>Confidence: 0.983. Support: 3281.` |
| 5 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {)}<br>	∧ -3.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {}}<br>	∧ +4.roles not in {ARGUMENT}<br>	∧ ^1.roles in {QUALIFIED}<br>⇒ y = ∅<br>Confidence: 0.997. Support: 35271.` |
| 6 | `  •••start_line = 255<br>	∧ -1.internal_type = StringLiteral<br>	∧ -2.reserved not in {(}<br>	∧ -5.diff_offset ≥ 22<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = "<br>Confidence: 0.943. Support: 694.` |
| 7 | `  •••start_line = 255<br>	∧ -1.internal_type = StringLiteral<br>	∧ -2.reserved not in {(}<br>	∧ -5.diff_offset ≤ 21<br>	∧ ^1.roles in {LITERAL} and not in {QUALIFIED}<br>⇒ y = "<br>Confidence: 0.934. Support: 98.` |
| 8 | `  •••start_line ≤ 254<br>	∧ -1.internal_type = StringLiteral<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = "<br>Confidence: 0.975. Support: 9244.` |
| 9 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = {<br>	∧ -2.reserved not in {)}<br>	∧ +1.length ≥ 2<br>	∧ +2.reserved not in {,, }}<br>	∧ ^1.internal_type not in {FunctionDeclaration}<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ⏎⇥⁺<br>Confidence: 0.956. Support: 7572.` |
| 10 | `  •••start_line ≤ 254<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = (<br>	∧ +1.roles in {STRING}<br>	∧ +1.length ≥ 2<br>	∧ +3.length ≤ 8<br>	∧ +4.roles not in {ARGUMENT}<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = "<br>Confidence: 0.960. Support: 4032.` |
| 11 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = (<br>	∧ -4.diff_offset ≥ 24<br>	∧ +1.roles not in {STRING}<br>	∧ +1.length ≥ 2<br>	∧ +2.reserved = )<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ∅<br>Confidence: 0.978. Support: 346.` |
| 12 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = (<br>	∧ -4.diff_offset ≤ 23<br>	∧ +1.roles not in {STRING}<br>	∧ +1.length ≥ 2<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ∅<br>Confidence: 0.951. Support: 9819.` |
| 13 | `  •••start_col ≥ 15<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = ;<br>	∧ +1.length ≥ 3<br>	∧ +2.roles in {NAME}<br>	∧ ^1.internal_type not in {BlockStatement}<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ⏎⏎<br>Confidence: 0.984. Support: 91.` |
| 14 | `  •••start_line ≤ 254<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.label in {<space>}<br>	∧ -1.reserved not in {(, ;, {}<br>	∧ +1.length ≥ 2<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = "<br>Confidence: 0.963. Support: 3433.` |
| 15 | `  •••start_col ≥ 6<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved not in {(, ;, {}<br>	∧ +1.length ≥ 2<br>	∧ ^1.roles in {LITERAL, STRING} and not in {QUALIFIED}<br>⇒ y = ∅<br>Confidence: 0.979. Support: 1611.` |
| 16 | `  •••start_col ≥ 6<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved = ,<br>	∧ +1.roles in {MAP}<br>	∧ +1.length ≥ 2<br>	∧ ^1.roles in {LITERAL} and not in {QUALIFIED, STRING}<br>⇒ y = ⏎<br>Confidence: 0.946. Support: 2058.` |
| 17 | `  •••start_col ≥ 6<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved = ,<br>	∧ -5.label in {<newline>}<br>	∧ +1.roles not in {MAP}<br>	∧ +1.length ≥ 2<br>	∧ ^1.roles in {LITERAL} and not in {QUALIFIED, STRING}<br>⇒ y = ⏎<br>Confidence: 0.973. Support: 457.` |
| 18 | `  •••start_col ≥ 6<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved = ,<br>	∧ -3.roles in {LITERAL}<br>	∧ -5.label not in {<newline>}<br>	∧ +1.roles not in {MAP}<br>	∧ +1.length ≥ 2<br>	∧ ^1.roles in {LITERAL} and not in {QUALIFIED, STRING}<br>⇒ y = ␣<br>Confidence: 0.950. Support: 270.` |
| 19 | `  •••start_col ≥ 6<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {(, ,, ;, {}<br>	∧ -3.roles in {LITERAL}<br>	∧ +1.length ≥ 2<br>	∧ ^1.roles in {LITERAL} and not in {QUALIFIED}<br>⇒ y = "<br>Confidence: 0.929. Support: 161.` |
| 20 | `  •••start_col ≥ 6<br>	∧ -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved not in {(, ;, {}<br>	∧ -2.length ≤ 14<br>	∧ +1.roles not in {EXPRESSION}<br>	∧ +1.length ≥ 2<br>	∧ ^1.internal_type = File<br>	∧ ^1.roles not in {LITERAL, QUALIFIED}<br>⇒ y = ∅<br>Confidence: 0.982. Support: 1843.` |
| 21 | `  •••start_col ≥ 6<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.label in {<newline>}<br>	∧ -1.reserved not in {(, ;, {}<br>	∧ +1.length ≥ 2<br>	∧ ^1.internal_type not in {File}<br>	∧ ^1.roles not in {LITERAL, QUALIFIED}<br>⇒ y = "<br>Confidence: 0.979. Support: 362.` |
| 22 | `  •••start_col ≥ 6<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<newline>, <space>}<br>	∧ -1.reserved not in {(, ;, {}<br>	∧ -1.roles in {STATEMENT}<br>	∧ -2.length ≤ 2<br>	∧ -3.diff_offset ≥ 5<br>	∧ +1.length ≥ 2<br>	∧ ^1.internal_type not in {File}<br>	∧ ^1.roles not in {LITERAL, QUALIFIED}<br>⇒ y = ⏎<br>Confidence: 0.931. Support: 124.` |
| 23 | `  •••start_col ≥ 6<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<newline>, <space>}<br>	∧ -1.reserved = }<br>	∧ -3.diff_offset ≥ 5<br>	∧ +1.length ≥ 2<br>	∧ ^1.internal_type not in {File}<br>	∧ ^1.roles not in {BLOCK, LITERAL, QUALIFIED}<br>⇒ y = ␣<br>Confidence: 0.941. Support: 297.` |
| 24 | `  •••start_col ≥ 6<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<newline>, <space>}<br>	∧ -1.reserved = ,<br>	∧ -3.diff_offset ≥ 5<br>	∧ -3.label in {<newline>}<br>	∧ +1.length ≥ 2<br>	∧ ^1.internal_type not in {File}<br>	∧ ^1.roles not in {LITERAL, QUALIFIED}<br>⇒ y = ⏎<br>Confidence: 0.954. Support: 571.` |
| 25 | `  •••start_col ≥ 6<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<newline>, <space>}<br>	∧ -1.reserved = ,<br>	∧ -2.roles in {ARGUMENT}<br>	∧ -3.diff_offset ≥ 5<br>	∧ -3.label not in {<newline>}<br>	∧ -5.label in {<newline>}<br>	∧ +1.length ≥ 2<br>	∧ ^1.internal_type not in {File}<br>	∧ ^1.roles not in {LITERAL, QUALIFIED}<br>⇒ y = ␣<br>Confidence: 0.998. Support: 271.` |
| 26 | `  •••start_col ≥ 6<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<newline>, <space>}<br>	∧ -1.reserved = ,<br>	∧ -2.roles not in {ARGUMENT}<br>	∧ -3.diff_offset ≥ 5<br>	∧ -3.label not in {<newline>}<br>	∧ -5.label in {<newline>}<br>	∧ +1.length ≥ 2<br>	∧ ^1.internal_type not in {File}<br>	∧ ^1.roles not in {LITERAL, QUALIFIED}<br>⇒ y = ⏎<br>Confidence: 0.946. Support: 267.` |
| 27 | `  •••start_col ≥ 6<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<newline>, <space>}<br>	∧ -1.reserved = ,<br>	∧ -3.diff_offset ≥ 5<br>	∧ -3.label not in {<newline>}<br>	∧ -5.label not in {<newline>}<br>	∧ +1.length ≥ 2<br>	∧ ^1.internal_type not in {File}<br>	∧ ^1.roles not in {LITERAL, QUALIFIED}<br>⇒ y = ␣<br>Confidence: 0.963. Support: 3319.` |
| 28 | `  •••start_col ≥ 6<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<newline>, <space>}<br>	∧ -1.reserved not in {(, ,, ;, {, }}<br>	∧ -3.diff_offset ≥ 5<br>	∧ +1.roles in {VALUE}<br>	∧ +1.length ≥ 2<br>	∧ ^1.internal_type not in {File}<br>	∧ ^1.roles in {CALL} and not in {LITERAL}<br>⇒ y = ∅<br>Confidence: 0.996. Support: 124.` |
| 29 | `  •••start_col ≥ 6<br>	∧ -1.diff_col ≥ 3<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<newline>, <space>}<br>	∧ -1.reserved not in {(, ,, ;, {, }}<br>	∧ -3.diff_offset ≥ 5<br>	∧ +1.length ≥ 2<br>	∧ ^1.internal_type not in {File}<br>	∧ ^1.roles in {CALL} and not in {LITERAL, QUALIFIED}<br>⇒ y = ␣<br>Confidence: 0.971. Support: 848.` |
| 30 | `  •••start_col ≤ 62<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<newline>, <space>}<br>	∧ -1.reserved not in {(, ,, ;, {, }}<br>	∧ -3.diff_offset ≥ 9<br>	∧ -4.label in {<newline>}<br>	∧ +1.length ≥ 2<br>	∧ ^1.internal_type not in {File}<br>	∧ ^1.roles not in {CALL, LITERAL, QUALIFIED}<br>⇒ y = ␣<br>Confidence: 0.978. Support: 1363.` |
| 31 | `  •••start_col ≤ 62<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<newline>, <space>}<br>	∧ -1.reserved not in {(, ,, ;, {, }}<br>	∧ -3.diff_offset ≤ 8<br>	∧ -3.length ≤ 5<br>	∧ -4.label in {<newline>}<br>	∧ +1.length ≥ 2<br>	∧ ^1.internal_type not in {File}<br>	∧ ^1.roles not in {CALL, LITERAL, QUALIFIED}<br>⇒ y = ␣<br>Confidence: 0.974. Support: 447.` |
| 32 | `  •••start_col ≤ 62<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<newline>, <space>}<br>	∧ -1.reserved not in {(, ,, ;, {, }}<br>	∧ -3.diff_offset ≥ 5<br>	∧ -4.label not in {<newline>}<br>	∧ +1.length ≥ 2<br>	∧ ^1.internal_type not in {File}<br>	∧ ^1.roles not in {CALL, LITERAL, QUALIFIED}<br>⇒ y = ␣<br>Confidence: 0.978. Support: 17818.` |
| 33 | `  •••start_col ≥ 6<br>	∧ -1.diff_offset ≥ 11<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<newline>, <space>}<br>	∧ -1.reserved not in {(, ;, [, {}<br>	∧ -3.diff_offset ≤ 4<br>	∧ -5.label not in {<newline>}<br>	∧ +1.length ≥ 2<br>	∧ ^1.internal_type not in {File}<br>	∧ ^1.roles not in {LITERAL}<br>⇒ y = ∅<br>Confidence: 0.955. Support: 213.` |
| 34 | `  •••start_col ≤ 5<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.label in {<newline>}<br>	∧ -1.reserved not in {(, ;, {}<br>	∧ +1.length ≥ 2<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = "<br>Confidence: 0.992. Support: 910.` |
| 35 | `  •••start_col ≤ 5<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<newline>, <space>}<br>	∧ -1.reserved = }<br>	∧ +1.length ≥ 2<br>	∧ ^1.roles in {STATEMENT} and not in {QUALIFIED, SCOPE}<br>⇒ y = ␣<br>Confidence: 0.989. Support: 306.` |
| 36 | `  •••start_col ≤ 5<br>	∧ -1.diff_col ≤ 3<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<newline>, <space>}<br>	∧ -1.reserved not in {(, ;, {, }}<br>	∧ +1.length ≥ 2<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ␣<br>Confidence: 0.996. Support: 558.` |
| 37 | `  •••start_col ≤ 5<br>	∧ -1.diff_col ≤ 2<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<newline>, <space>}<br>	∧ -1.reserved = ,<br>	∧ +1.length ≥ 2<br>	∧ ^1.roles in {MAP} and not in {QUALIFIED}<br>⇒ y = ⏎<br>Confidence: 0.942. Support: 318.` |
| 38 | `  •••start_col ≤ 5<br>	∧ -1.diff_col ≤ 2<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<newline>, <space>}<br>	∧ -1.reserved not in {(, ,, ;, {, }}<br>	∧ +1.roles not in {LITERAL}<br>	∧ +1.length ≥ 2<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ∅<br>Confidence: 0.981. Support: 3396.` |
| 39 | `  -1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = =<br>	∧ +1.length ≤ 1<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ␣<br>Confidence: 0.999. Support: 9029.` |
| 40 | `  -1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = }<br>	∧ +1.length ≤ 1<br>	∧ ^1.internal_type = TemplateLiteral<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ∅<br>Confidence: 0.955. Support: 586.` |
| 41 | `  •••start_line ≤ 240<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.roles not in {IDENTIFIER}<br>	∧ +1.reserved = }<br>	∧ +1.length ≤ 1<br>	∧ +4.internal_type not in {CommentBlock}<br>	∧ ^1.internal_type = File<br>	∧ ^1.roles not in {DECLARATION, QUALIFIED}<br>⇒ y = ⏎⇥⁻<br>Confidence: 0.954. Support: 274.` |
| 42 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.roles not in {IDENTIFIER}<br>	∧ +1.reserved = }<br>	∧ +1.length ≤ 1<br>	∧ +4.internal_type not in {CommentBlock}<br>	∧ ^1.internal_type not in {File}<br>	∧ ^1.roles not in {DECLARATION, QUALIFIED}<br>⇒ y = ⏎⇥⁻<br>Confidence: 0.926. Support: 7060.` |
| 43 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = [<br>	∧ +1.reserved = {<br>	∧ +1.length ≤ 1<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ⏎⇥⁺<br>Confidence: 0.939. Support: 106.` |
| 44 | `  •••start_line ≤ 235<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {(, [}<br>	∧ +1.reserved = {<br>	∧ +1.length ≤ 1<br>	∧ ^1.internal_type = FunctionExpression<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ␣<br>Confidence: 0.990. Support: 442.` |
| 45 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {(, [}<br>	∧ +1.reserved = {<br>	∧ +1.length ≤ 1<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ␣<br>Confidence: 0.984. Support: 7213.` |
| 46 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = (<br>	∧ +1.reserved not in {=, {, }}<br>	∧ +1.roles in {EXPRESSION, STRING}<br>	∧ +1.length ≤ 1<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = "<br>Confidence: 0.927. Support: 296.` |
| 47 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = (<br>	∧ +1.reserved not in {=, {, }}<br>	∧ +1.roles in {EXPRESSION} and not in {STRING}<br>	∧ +1.length ≤ 1<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ∅<br>Confidence: 0.976. Support: 1086.` |
| 48 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved not in {(}<br>	∧ +1.reserved not in {=, {, }}<br>	∧ +1.roles in {EXPRESSION}<br>	∧ +1.length ≤ 1<br>	∧ ^1.roles in {STRING} and not in {QUALIFIED}<br>⇒ y = ∅<br>Confidence: 0.998. Support: 220.` |
| 49 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved not in {(}<br>	∧ -2.label in {<newline>}<br>	∧ -5.reserved = )<br>	∧ +1.reserved not in {{, }}<br>	∧ +1.roles in {EXPRESSION} and not in {KEY}<br>	∧ +1.length ≤ 1<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ␣<br>Confidence: 0.995. Support: 92.` |
| 50 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = if<br>	∧ +1.reserved not in {{, }}<br>	∧ +1.roles not in {EXPRESSION}<br>	∧ +1.length ≤ 1<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ␣<br>Confidence: 0.955. Support: 2215.` |
| 51 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {if}<br>	∧ +1.reserved not in {}}<br>	∧ +1.roles not in {EXPRESSION}<br>	∧ +1.length ≤ 1<br>	∧ +2.roles in {EXPRESSION}<br>	∧ ^1.internal_type = BinaryExpression<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ␣<br>Confidence: 0.968. Support: 1462.` |
| 52 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = ,<br>	∧ -5.reserved = (<br>	∧ +1.reserved not in {}}<br>	∧ +1.roles not in {EXPRESSION}<br>	∧ +1.length ≤ 1<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ␣<br>Confidence: 0.994. Support: 444.` |
| 53 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = :<br>	∧ +1.reserved not in {}}<br>	∧ +1.roles not in {EXPRESSION}<br>	∧ +1.length ≤ 1<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ␣<br>Confidence: 0.986. Support: 466.` |
| 54 | `  -1.internal_type = DirectiveLiteral<br>	∧ -1.reserved not in {,, :}<br>	∧ +1.reserved not in {=, {, }}<br>	∧ +1.roles not in {EXPRESSION}<br>	∧ +1.length ≤ 1<br>	∧ ^1.internal_type not in {BinaryExpression}<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = "<br>Confidence: 0.998. Support: 252.` |
| 55 | `  -1.internal_type not in {DirectiveLiteral, StringLiteral}<br>	∧ -1.reserved = return<br>	∧ +1.reserved not in {}}<br>	∧ +1.roles not in {EXPRESSION}<br>	∧ +1.length ≤ 1<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ␣<br>Confidence: 0.973. Support: 243.` |
| 56 | `  -1.internal_type not in {DirectiveLiteral, StringLiteral}<br>	∧ -1.reserved not in {,, :}<br>	∧ +1.reserved not in {=, {, }}<br>	∧ +1.roles not in {EXPRESSION}<br>	∧ +1.length ≤ 1<br>	∧ ^1.internal_type = CallExpression<br>	∧ ^1.roles in {CONDITION}<br>⇒ y = ∅<br>Confidence: 0.998. Support: 202.` |
| 57 | `  •••start_line ≤ 212<br>	∧ -1.internal_type not in {DirectiveLiteral, StringLiteral}<br>	∧ -1.reserved not in {,, :, =}<br>	∧ +1.reserved not in {=, {, }}<br>	∧ +1.roles not in {EXPRESSION}<br>	∧ +1.length ≤ 1<br>	∧ ^1.internal_type not in {BinaryExpression}<br>	∧ ^1.roles in {VARIABLE} and not in {CONDITION}<br>⇒ y = ∅<br>Confidence: 0.950. Support: 111.` |
| 58 | `  -1.internal_type not in {DirectiveLiteral, StringLiteral}<br>	∧ -1.reserved not in {,, :}<br>	∧ -2.diff_offset ≥ 4<br>	∧ -2.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {=, {, }}<br>	∧ +1.roles not in {EXPRESSION}<br>	∧ +1.length ≤ 1<br>	∧ ^1.internal_type not in {BinaryExpression}<br>	∧ ^1.roles in {OPERATOR} and not in {CONDITION, VARIABLE}<br>⇒ y = ∅<br>Confidence: 0.970. Support: 380.` |
| 59 | `  •••start_col ≤ 12<br>	∧ -1.internal_type not in {DirectiveLiteral, StringLiteral}<br>	∧ -1.reserved not in {,, :}<br>	∧ +1.reserved = ]<br>	∧ +1.roles not in {EXPRESSION}<br>	∧ +1.length ≤ 1<br>	∧ ^1.internal_type not in {BinaryExpression}<br>	∧ ^1.roles not in {CONDITION, OPERATOR, QUALIFIED, VARIABLE}<br>⇒ y = ⏎⇥⁻<br>Confidence: 0.962. Support: 197.` |
| 60 | `  -1.internal_type not in {DirectiveLiteral, StringLiteral}<br>	∧ -1.reserved not in {,, :}<br>	∧ -1.roles in {EXPRESSION}<br>	∧ -2.length ≥ 4<br>	∧ +1.reserved not in {=, {}<br>	∧ +1.roles not in {EXPRESSION}<br>	∧ +1.length ≤ 1<br>	∧ +2.reserved = ;<br>	∧ ^1.internal_type not in {BinaryExpression}<br>	∧ ^1.roles not in {CONDITION, OPERATOR, QUALIFIED, VARIABLE}<br>⇒ y = ⏎⇥⁻<br>Confidence: 0.991. Support: 176.` |
| 61 | `  -1.internal_type not in {DirectiveLiteral, StringLiteral}<br>	∧ -1.reserved = (<br>	∧ -1.roles not in {EXPRESSION}<br>	∧ -2.internal_type not in {StringLiteral}<br>	∧ -2.length ≥ 4<br>	∧ +1.reserved not in {=, {, }}<br>	∧ +1.roles not in {EXPRESSION}<br>	∧ +1.length ≤ 1<br>	∧ +2.reserved = ;<br>	∧ ^1.internal_type not in {BinaryExpression}<br>	∧ ^1.roles not in {CONDITION, OPERATOR, VARIABLE}<br>⇒ y = ∅<br>Confidence: 0.999. Support: 687.` |
| 62 | `  -1.internal_type not in {DirectiveLiteral, StringLiteral}<br>	∧ -1.reserved not in {,, :}<br>	∧ -2.internal_type not in {StringLiteral}<br>	∧ -2.length ≤ 3<br>	∧ +1.reserved not in {=, {, }}<br>	∧ +1.roles not in {EXPRESSION}<br>	∧ +1.length ≤ 1<br>	∧ +2.reserved = ;<br>	∧ ^1.internal_type not in {BinaryExpression}<br>	∧ ^1.roles not in {CONDITION, OPERATOR, VARIABLE}<br>⇒ y = ∅<br>Confidence: 0.961. Support: 5796.` |
| 63 | `  •••start_col ≥ 44<br>	∧ -1.internal_type not in {DirectiveLiteral, StringLiteral}<br>	∧ -1.reserved not in {,, :}<br>	∧ -2.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {=, {, }}<br>	∧ +1.roles not in {EXPRESSION}<br>	∧ +1.length ≤ 1<br>	∧ +2.reserved not in {;}<br>	∧ ^1.internal_type = TemplateLiteral<br>	∧ ^1.roles not in {CONDITION, OPERATOR, VARIABLE}<br>⇒ y = ∅<br>Confidence: 0.931. Support: 94.` |
| 64 | `  -1.internal_type not in {DirectiveLiteral, StringLiteral}<br>	∧ -1.reserved not in {,, :}<br>	∧ -1.roles in {EXPRESSION}<br>	∧ -2.internal_type not in {StringLiteral}<br>	∧ -3.length ≥ 5<br>	∧ -5.diff_line ≥ 1<br>	∧ +1.reserved = )<br>	∧ +1.roles not in {EXPRESSION}<br>	∧ +1.length ≤ 1<br>	∧ +2.reserved not in {;}<br>	∧ ^1.internal_type not in {BinaryExpression}<br>	∧ ^1.roles not in {CONDITION, OPERATOR, VARIABLE}<br>⇒ y = ∅<br>Confidence: 0.974. Support: 361.` |
| 65 | `  -1.internal_type not in {DirectiveLiteral, StringLiteral}<br>	∧ -1.reserved not in {,, :}<br>	∧ -1.roles in {EXPRESSION}<br>	∧ -2.internal_type not in {StringLiteral}<br>	∧ -5.diff_line = 0<br>	∧ +1.reserved = )<br>	∧ +1.roles not in {EXPRESSION}<br>	∧ +1.length ≤ 1<br>	∧ +2.reserved not in {;}<br>	∧ ^1.internal_type not in {BinaryExpression}<br>	∧ ^1.roles not in {CONDITION, OPERATOR, VARIABLE}<br>⇒ y = ∅<br>Confidence: 0.993. Support: 3926.` |
| 66 | `  -1.internal_type not in {DirectiveLiteral, StringLiteral}<br>	∧ -1.reserved not in {,, :}<br>	∧ -1.roles in {EXPRESSION}<br>	∧ -2.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {), =, {, }}<br>	∧ +1.roles not in {EXPRESSION}<br>	∧ +1.length ≤ 1<br>	∧ +2.reserved not in {;}<br>	∧ ^1.internal_type not in {BinaryExpression}<br>	∧ ^1.roles not in {CONDITION, OPERATOR, VARIABLE}<br>⇒ y = ∅<br>Confidence: 0.999. Support: 28768.` |
| 67 | `  -1.diff_col ≥ 7<br>	∧ -1.internal_type not in {DirectiveLiteral, StringLiteral}<br>	∧ -1.reserved not in {,, :}<br>	∧ -1.roles not in {EXPRESSION}<br>	∧ -1.length ≥ 3<br>	∧ -2.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {=, {, }}<br>	∧ +1.roles not in {EXPRESSION}<br>	∧ +1.length ≤ 1<br>	∧ +2.reserved not in {;}<br>	∧ ^1.internal_type not in {BinaryExpression}<br>	∧ ^1.roles not in {CONDITION, OPERATOR, VARIABLE}<br>⇒ y = ∅<br>Confidence: 0.947. Support: 1190.` |
| 68 | `  -1.internal_type not in {DirectiveLiteral, StringLiteral}<br>	∧ -1.reserved not in {,, :}<br>	∧ -1.roles not in {EXPRESSION}<br>	∧ -1.length ≤ 2<br>	∧ -2.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {=, {, }}<br>	∧ +1.roles not in {EXPRESSION}<br>	∧ +1.length ≤ 1<br>	∧ +2.reserved not in {;}<br>	∧ ^1.internal_type not in {BinaryExpression}<br>	∧ ^1.roles not in {CONDITION, OPERATOR, VARIABLE}<br>⇒ y = ∅<br>Confidence: 0.981. Support: 20444.` |

### Note
All statistics are calculated with respect to the "analyze" config. This means that the rules were filtered by
`confidence_threshold` and `support_threshold` values.

<details>
    <summary>Machine-readable report</summary>
```json
{"javascript": {"avg_rule_len": 10.602941176470589, "max_conf": 0.9993908405303955, "max_support": 35271, "min_conf": 0.9264164566993713, "min_support": 91, "num_rules": 68}}
```
</details>
