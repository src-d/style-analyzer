# Model report for file:///tmp/top-repos-quality-repos-d2uvm83i/jquery HEAD dae5f3ce3d2df27873d01f0d9682f6a91ad66b87

### Dump

```json
{'created_at': '2019-06-11 11:26:48',
 'datasets': [],
 'dependencies': [],
 'description': 'Model bound to style.format.analyzer.FormatAnalyzer Lookout analyzer.',
 'environment': {'packages': 'ConfigArgParse==0.13.0 Jinja2==2.10.1 MarkupSafe==1.1.1 PyStemmer==1.3.0 PyYAML==5.1 Pympler==0.5 SQLAlchemy==1.2.10 SQLAlchemy-Utils==0.33.3 asdf==2.3.3 bblfsh==2.12.7 boto==2.49.0 boto3==1.9.165 botocore==1.12.165 cachetools==2.0.1 certifi==2019.3.9 chardet==3.0.4 clint==0.5.1 docker==4.0.1 dulwich==0.19.11 google-auth==1.6.3 google-auth-httplib2==0.0.3 google-cloud-core==0.25.0 grpcio==1.20.1 grpcio-tools==1.20.1 httplib2==0.12.3 humanfriendly==4.16.1 humanize==0.5.1 idna==2.8 jmespath==0.9.4 jsonschema==2.6.0 lookout-sdk==0.4.1 lookout-sdk-ml==0.19.1 lookout-style==0.2.0 lz4==2.1.6 modelforge==0.13.4 numpy==1.16.3 packaging==19.0 pandas==0.22.0 pip==19.1.1 prometheus-client==0.6.0 protobuf==3.7.1 psycopg2-binary==2.7.5 pygtrie==2.3 pyparsing==2.4.0 python-dateutil==2.8.0 python-igraph==0.7.1.post6 pytz==2019.1 requests==2.22.0 requirements-parser==0.2.0 scikit-learn==0.20.1 scikit-optimize==0.5.2 scipy==1.3.0 semantic-version==2.6.0 setuptools==41.0.1 six==1.12.0 '
                             'smart-open==1.8.1 sortedcontainers==2.1.0 sourced-ml==0.8.2 spdx==2.5.0 stringcase==1.2.0 tabulate==0.8.2 tqdm==4.32.1 urllib3==1.24.3 websocket-client==0.56.0 xxhash==1.3.0',
                 'platform': 'Linux-4.15.0-51-generic-x86_64-with-Ubuntu-18.04-bionic',
                 'python': '3.6.8 (default, Jan 14 2019, 11:02:34) [GCC 8.0.1 20180414 (experimental) [trunk revision 259383]]'},
 'license': 'ODbL-1.0',
 'metrics': {},
 'model': 'style.format.analyzer.FormatAnalyzer',
 'references': [],
 'series': 'Lookout',
 'size': '21.7 kB',
 'tags': [],
 'uuid': 'c60c9f2d-1f1a-4ee9-b39a-ea4587753c8b',
 'vendor': 'source{d}',
 'version': [1]}
style.format.analyzer.FormatAnalyzer/[1] file:///tmp/top-repos-quality-repos-d2uvm83i/jquery dae5f3ce3d2df27873d01f0d9682f6a91ad66b87

# javascript
281 rules, avg.len. 12.1
## train
PPCR: 0.962246
### report
macro
{'f1-score': 0.6036810485092455,
 'precision': 0.6693464643139714,
 'recall': 0.5626295965479309,
 'support': 175456}
micro
{'f1-score': 0.9593630311873063,
 'precision': 0.9593630311873063,
 'recall': 0.9593630311873063,
 'support': 175456}
weighted
{'f1-score': 0.9557390824849668,
 'precision': 0.9559297655158443,
 'recall': 0.9593630311873063,
 'support': 175456}
### report_full
macro
{'f1-score': 0.5542389955206375,
 'precision': 0.6693464643139714,
 'recall': 0.4956242766478976,
 'support': 182340}
micro
{'f1-score': 0.9409048731679505,
 'precision': 0.9593630311873063,
 'recall': 0.9231435779313371,
 'support': 182340}
weighted
{'f1-score': 0.9292716576561756,
 'precision': 0.9516772324632552,
 'recall': 0.9231435779313371,
 'support': 182340}
## test
PPCR: 0.962988
### report
macro
{'f1-score': 0.591639489629853,
 'precision': 0.6655096145985069,
 'recall': 0.5472430150439005,
 'support': 46390}
micro
{'f1-score': 0.968743263634404,
 'precision': 0.968743263634404,
 'recall': 0.968743263634404,
 'support': 46390}
weighted
{'f1-score': 0.965747492214533,
 'precision': 0.9668413938164324,
 'recall': 0.968743263634404,
 'support': 46390}
### report_full
macro
{'f1-score': 0.5349858267827781,
 'precision': 0.6655096145985069,
 'recall': 0.4760739940705941,
 'support': 48173}
micro
{'f1-score': 0.9504774594714634,
 'precision': 0.968743263634404,
 'recall': 0.9328877171859755,
 'support': 48173}
weighted
{'f1-score': 0.9378036222858684,
 'precision': 0.9631814311289002,
 'recall': 0.9328877171859755,
 'support': 48173}
```

## javascript
### Summary
175 rules, avg.len. 12.2

| | |
|-|-|
|Min support|133|
|Max support|27635|
|Min confidence|0.9210526347160339|
|Max confidence|0.9999549984931946|

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
| 1 | `  -1.internal_type = StringLiteral<br>⇒ y = "<br>Confidence: 1.000. Support: 10190.` |
| 2 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.label in {<space>}<br>⇒ y = "<br>Confidence: 1.000. Support: 9072.` |
| 3 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<space>}<br>	∧ +2.roles in {EXPRESSION}<br>	∧ ^1.roles in {OPERATOR}<br>⇒ y = ␣<br>Confidence: 0.964. Support: 3652.` |
| 4 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<space>}<br>	∧ +1.reserved = {<br>	∧ +2.roles in {EXPRESSION}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ␣<br>Confidence: 0.956. Support: 2463.` |
| 5 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<space>}<br>	∧ +1.reserved not in {{}<br>	∧ +2.roles in {EXPRESSION}<br>	∧ ^1.internal_type = VariableDeclarator<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ␣<br>Confidence: 0.957. Support: 1027.` |
| 6 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<space>}<br>	∧ +1.reserved not in {{}<br>	∧ +1.roles in {COMMENT}<br>	∧ +1.length ≥ 2<br>	∧ +2.roles in {EXPRESSION}<br>	∧ ^1.internal_type not in {VariableDeclarator}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.998. Support: 663.` |
| 7 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved = if<br>	∧ +1.reserved not in {{}<br>	∧ +1.length ≤ 1<br>	∧ +2.roles in {EXPRESSION}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ␣<br>Confidence: 0.999. Support: 443.` |
| 8 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved = (<br>	∧ +1.length ≤ 1<br>	∧ +2.roles in {EXPRESSION}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ␣<br>Confidence: 0.990. Support: 363.` |
| 9 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved not in {(, if}<br>	∧ +1.length ≤ 1<br>	∧ +2.roles in {EXPRESSION}<br>	∧ ^1.internal_type = ConditionalExpression<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ␣<br>Confidence: 0.991. Support: 283.` |
| 10 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved not in {(, if}<br>	∧ +1.reserved = }<br>	∧ +1.length ≤ 1<br>	∧ +2.roles in {EXPRESSION}<br>	∧ ^1.internal_type not in {VariableDeclarator}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ⏎⇥⁻<br>Confidence: 0.970. Support: 151.` |
| 11 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved not in {(, ,, :, if}<br>	∧ -1.roles in {EXPRESSION}<br>	∧ +1.reserved not in {{, }}<br>	∧ +1.length ≤ 1<br>	∧ +2.roles in {EXPRESSION}<br>	∧ ^1.internal_type not in {VariableDeclarator}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.999. Support: 24413.` |
| 12 | `  •••start_col ≥ 11<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved not in {(, ,, :, if}<br>	∧ -1.roles not in {EXPRESSION}<br>	∧ -1.length ≥ 3<br>	∧ +1.reserved not in {{, }}<br>	∧ +1.length ≤ 1<br>	∧ +2.roles in {EXPRESSION}<br>	∧ ^1.internal_type not in {VariableDeclarator}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.989. Support: 1143.` |
| 13 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved not in {(, ,, :, if}<br>	∧ -1.roles not in {EXPRESSION}<br>	∧ -1.length ≤ 2<br>	∧ +1.reserved not in {{, }}<br>	∧ +1.length ≤ 1<br>	∧ +2.roles in {EXPRESSION}<br>	∧ ^1.internal_type not in {VariableDeclarator}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.998. Support: 6176.` |
| 14 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved = .<br>	∧ +2.roles not in {EXPRESSION}<br>⇒ y = ∅<br>Confidence: 1.000. Support: 11067.` |
| 15 | `  -1.diff_col ≤ 9<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved not in {., ;}<br>	∧ +1.reserved = ;<br>	∧ +2.roles not in {EXPRESSION}<br>⇒ y = ∅<br>Confidence: 1.000. Support: 3699.` |
| 16 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved = if<br>	∧ +1.reserved = (<br>	∧ +2.roles not in {EXPRESSION}<br>⇒ y = ␣<br>Confidence: 0.997. Support: 159.` |
| 17 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved not in {;, if}<br>	∧ +1.reserved = (<br>	∧ +2.roles not in {EXPRESSION}<br>⇒ y = ∅<br>Confidence: 0.982. Support: 2736.` |
| 18 | `  -1.diff_col ≥ 13<br>	∧ -1.internal_type = CommentLine<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved not in {., ;, {}<br>	∧ +1.reserved not in {(, ;}<br>	∧ +2.roles not in {EXPRESSION}<br>⇒ y = ⏎<br>Confidence: 0.970. Support: 1267.` |
| 19 | `  -1.diff_col ≥ 13<br>	∧ -1.internal_type not in {CommentLine}<br>	∧ -1.reserved not in {., ;, {}<br>	∧ -2.diff_line ≥ 1<br>	∧ +1.reserved not in {(, ;}<br>	∧ +2.roles not in {EXPRESSION}<br>⇒ y = "<br>Confidence: 1.000. Support: 1004.` |
| 20 | `  -1.diff_col ≤ 12<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved not in {., ;, {}<br>	∧ +1.roles in {COMMENT}<br>	∧ +2.roles not in {EXPRESSION}<br>⇒ y = ∅<br>Confidence: 0.997. Support: 904.` |
| 21 | `  -1.diff_col ≤ 12<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved not in {., ;, {}<br>	∧ +1.reserved = ,<br>	∧ +1.roles not in {KEY}<br>	∧ +2.roles not in {EXPRESSION}<br>⇒ y = ∅<br>Confidence: 0.985. Support: 709.` |
| 22 | `  -1.diff_col ≤ 12<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved not in {., ;, {}<br>	∧ -1.roles in {KEY}<br>	∧ +1.reserved not in {,}<br>	∧ +1.roles not in {KEY}<br>	∧ +2.roles not in {EXPRESSION}<br>⇒ y = ∅<br>Confidence: 0.999. Support: 479.` |
| 23 | `  -1.diff_col ≤ 12<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.label in {<newline>}<br>	∧ -1.reserved not in {., ;, {}<br>	∧ +1.reserved not in {(, ,, ;, }}<br>	∧ +2.roles not in {EXPRESSION}<br>⇒ y = "<br>Confidence: 0.998. Support: 210.` |
| 24 | `  -1.diff_col ≤ 12<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<newline>, <space>}<br>	∧ -1.reserved = (<br>	∧ -1.roles not in {KEY}<br>	∧ +1.reserved = )<br>	∧ +1.roles not in {KEY}<br>	∧ +2.roles not in {EXPRESSION}<br>⇒ y = ∅<br>Confidence: 0.997. Support: 2009.` |
| 25 | `  -1.diff_col ≤ 12<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<newline>, <space>}<br>	∧ -1.reserved = (<br>	∧ -1.roles not in {KEY}<br>	∧ +1.reserved not in {(, ), ,, }}<br>	∧ +1.roles not in {COMMENT, KEY}<br>	∧ +2.roles not in {COMMENT, EXPRESSION}<br>	∧ ^1.internal_type not in {VariableDeclaration}<br>⇒ y = ␣<br>Confidence: 0.986. Support: 10680.` |
| 26 | `  -1.diff_col ≤ 12<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<newline>, <space>}<br>	∧ -1.reserved not in {(, ., ;, {}<br>	∧ -1.roles not in {KEY}<br>	∧ -5.label not in {<newline>}<br>	∧ +1.reserved not in {(, ,, ;, }}<br>	∧ +1.roles not in {KEY}<br>	∧ +2.roles not in {EXPRESSION}<br>	∧ ^1.internal_type = ArrayExpression<br>⇒ y = ␣<br>Confidence: 0.925. Support: 459.` |
| 27 | `  -1.diff_col ≤ 12<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<newline>, <space>}<br>	∧ -1.reserved not in {(, ;, {, }}<br>	∧ -1.roles not in {IDENTIFIER, KEY}<br>	∧ -3.reserved = (<br>	∧ -4.diff_offset ≤ 5<br>	∧ +1.reserved not in {,, }}<br>	∧ +1.roles not in {KEY}<br>	∧ +1.length ≥ 2<br>	∧ +2.roles not in {EXPRESSION}<br>	∧ ^1.internal_type not in {ArrayExpression}<br>⇒ y = ∅<br>Confidence: 0.991. Support: 173.` |
| 28 | `  -1.diff_col ≤ 12<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<newline>, <space>}<br>	∧ -1.reserved not in {(, ., ;, {, }}<br>	∧ -1.roles not in {IDENTIFIER, KEY}<br>	∧ -3.reserved not in {(}<br>	∧ +1.reserved not in {(, ,, ;, }}<br>	∧ +1.roles not in {KEY}<br>	∧ +1.length ≥ 2<br>	∧ +2.roles not in {EXPRESSION}<br>	∧ ^1.internal_type not in {ArrayExpression, VariableDeclaration}<br>⇒ y = ␣<br>Confidence: 0.937. Support: 13595.` |
| 29 | `  -1.diff_col ≤ 12<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<newline>, <space>}<br>	∧ -1.reserved not in {(, ., ;, {}<br>	∧ -1.roles not in {KEY}<br>	∧ -2.length ≤ 3<br>	∧ -4.label in {<newline>}<br>	∧ +1.reserved not in {(, ,, ;, }}<br>	∧ +1.roles not in {KEY}<br>	∧ +1.length ≤ 1<br>	∧ +2.roles not in {EXPRESSION}<br>	∧ ^1.internal_type not in {ArrayExpression, VariableDeclaration}<br>⇒ y = ␣<br>Confidence: 0.965. Support: 411.` |
| 30 | `  -1.diff_col ≤ 12<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<newline>, <space>}<br>	∧ -1.reserved not in {(, ., ;, {}<br>	∧ -1.roles not in {KEY}<br>	∧ -4.label not in {<newline>}<br>	∧ +1.reserved not in {(, ,, ;, }}<br>	∧ +1.roles not in {KEY}<br>	∧ +1.length ≤ 1<br>	∧ +2.roles not in {EXPRESSION}<br>	∧ ^1.internal_type not in {ArrayExpression, VariableDeclaration}<br>⇒ y = ␣<br>Confidence: 0.984. Support: 15283.` |
| 31 | `  -1.roles in {STRING}<br>⇒ y = "<br>Confidence: 1.000. Support: 10319.` |
| 32 | `  -1.label in {<space>}<br>	∧ -1.roles not in {STRING}<br>⇒ y = "<br>Confidence: 1.000. Support: 8953.` |
| 33 | `  -1.label not in {<space>}<br>	∧ -1.roles not in {STRING}<br>	∧ +1.reserved not in {(}<br>	∧ +2.roles in {EXPRESSION}<br>	∧ ^1.roles in {OPERATOR}<br>⇒ y = ␣<br>Confidence: 0.988. Support: 3418.` |
| 34 | `  -1.label not in {<space>}<br>	∧ -1.roles not in {STRING}<br>	∧ +1.reserved = {<br>	∧ +2.roles in {EXPRESSION}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ␣<br>Confidence: 0.954. Support: 2569.` |
| 35 | `  -1.label not in {<space>}<br>	∧ -1.roles not in {STRING}<br>	∧ +1.reserved = =<br>	∧ +2.roles in {EXPRESSION}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ␣<br>Confidence: 0.999. Support: 978.` |
| 36 | `  -1.label not in {<space>}<br>	∧ -1.roles not in {STRING}<br>	∧ +1.internal_type = CommentLine<br>	∧ +1.reserved not in {=, {}<br>	∧ +1.length ≥ 2<br>	∧ +2.roles in {EXPRESSION}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.999. Support: 637.` |
| 37 | `  -1.label not in {<space>}<br>	∧ -1.reserved = if<br>	∧ -1.roles not in {STRING}<br>	∧ +1.reserved not in {{}<br>	∧ +1.length ≤ 1<br>	∧ +2.roles in {EXPRESSION}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ␣<br>Confidence: 0.999. Support: 416.` |
| 38 | `  -1.label not in {<space>}<br>	∧ -1.reserved = (<br>	∧ -1.roles not in {STRING}<br>	∧ +1.length ≤ 1<br>	∧ +2.roles in {EXPRESSION}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ␣<br>Confidence: 0.993. Support: 368.` |
| 39 | `  -1.label not in {<space>}<br>	∧ -1.reserved not in {(, if}<br>	∧ -1.roles not in {STRING}<br>	∧ +1.length ≤ 1<br>	∧ +2.roles in {EXPRESSION}<br>	∧ ^1.internal_type = ConditionalExpression<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ␣<br>Confidence: 0.985. Support: 296.` |
| 40 | `  -1.label not in {<space>}<br>	∧ -1.reserved not in {(, ,, if}<br>	∧ -1.roles not in {STRING}<br>	∧ +1.reserved = }<br>	∧ +1.length ≤ 1<br>	∧ +2.roles in {EXPRESSION}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ⏎⇥⁻<br>Confidence: 0.939. Support: 156.` |
| 41 | `  -1.label not in {<space>}<br>	∧ -1.reserved not in {(, ,, :, if}<br>	∧ -1.roles in {EXPRESSION} and not in {STRING}<br>	∧ +1.reserved not in {=, {, }}<br>	∧ +1.length ≤ 1<br>	∧ +2.roles in {EXPRESSION}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.999. Support: 23980.` |
| 42 | `  -1.diff_col ≤ 2<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved not in {(, ,, :, if}<br>	∧ -1.roles not in {EXPRESSION, STRING}<br>	∧ -5.diff_line ≥ 1<br>	∧ +1.reserved not in {=, {, }}<br>	∧ +1.length ≤ 1<br>	∧ +2.roles in {EXPRESSION}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.989. Support: 1273.` |
| 43 | `  -1.label not in {<space>}<br>	∧ -1.reserved not in {(, ,, :, if}<br>	∧ -1.roles not in {EXPRESSION, STRING}<br>	∧ -5.diff_line = 0<br>	∧ +1.reserved not in {=, {, }}<br>	∧ +1.length ≤ 1<br>	∧ +2.roles in {EXPRESSION}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.997. Support: 6032.` |
| 44 | `  -1.label not in {<space>}<br>	∧ -1.reserved = .<br>	∧ -1.roles not in {STRING}<br>	∧ +2.roles not in {EXPRESSION}<br>⇒ y = ∅<br>Confidence: 1.000. Support: 11104.` |
| 45 | `  -1.label not in {<space>}<br>	∧ -1.reserved not in {., ;}<br>	∧ -1.roles not in {STRING}<br>	∧ +1.reserved = ;<br>	∧ +2.roles not in {EXPRESSION}<br>⇒ y = ∅<br>Confidence: 0.983. Support: 3862.` |
| 46 | `  -1.label not in {<space>}<br>	∧ -1.reserved = if<br>	∧ -1.roles not in {STRING}<br>	∧ +1.reserved = (<br>	∧ +2.roles not in {EXPRESSION}<br>⇒ y = ␣<br>Confidence: 0.996. Support: 141.` |
| 47 | `  -1.label not in {<space>}<br>	∧ -1.reserved not in {;, if}<br>	∧ -1.roles not in {STRING}<br>	∧ +1.reserved = (<br>	∧ +2.roles not in {EXPRESSION}<br>⇒ y = ∅<br>Confidence: 0.984. Support: 2919.` |
| 48 | `  -1.label not in {<space>}<br>	∧ -1.reserved = {<br>	∧ -1.roles not in {STRING}<br>	∧ +1.reserved not in {(}<br>	∧ +2.internal_type = CommentLine<br>	∧ +2.roles not in {EXPRESSION}<br>⇒ y = ⏎⏎⇥⁺<br>Confidence: 0.930. Support: 165.` |
| 49 | `  -1.diff_col ≥ 13<br>	∧ -1.internal_type = CommentLine<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved not in {., ;, {}<br>	∧ -1.roles not in {STRING}<br>	∧ +1.reserved not in {(, ;}<br>	∧ +2.roles not in {EXPRESSION}<br>⇒ y = ⏎<br>Confidence: 0.971. Support: 1309.` |
| 50 | `  -1.diff_col ≥ 13<br>	∧ -1.internal_type not in {CommentLine}<br>	∧ -1.reserved not in {., ;, {}<br>	∧ +1.reserved not in {(, ;}<br>	∧ +1.roles in {LITERAL}<br>	∧ +2.roles not in {EXPRESSION}<br>⇒ y = "<br>Confidence: 0.999. Support: 955.` |
| 51 | `  -1.diff_col ≤ 12<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved not in {., ;, {}<br>	∧ -1.roles not in {STRING}<br>	∧ +1.roles in {COMMENT}<br>	∧ +2.roles not in {EXPRESSION}<br>⇒ y = ∅<br>Confidence: 0.997. Support: 866.` |
| 52 | `  -1.diff_col ≤ 12<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved not in {., ;, {}<br>	∧ -1.roles not in {STRING}<br>	∧ +1.reserved = ,<br>	∧ +1.roles not in {KEY}<br>	∧ +2.roles not in {EXPRESSION}<br>⇒ y = ∅<br>Confidence: 0.979. Support: 637.` |
| 53 | `  -1.diff_col ≤ 12<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved not in {., ;, {}<br>	∧ -1.roles in {KEY} and not in {STRING}<br>	∧ +1.reserved not in {,}<br>	∧ +1.roles not in {KEY}<br>	∧ +2.roles not in {EXPRESSION}<br>⇒ y = ∅<br>Confidence: 0.999. Support: 471.` |
| 54 | `  -1.diff_col ≤ 12<br>	∧ -1.label in {<newline>} and not in {<space>}<br>	∧ -1.reserved not in {., ;, {}<br>	∧ +1.reserved not in {(, ,, ;, }}<br>	∧ +2.roles not in {EXPRESSION}<br>⇒ y = "<br>Confidence: 0.998. Support: 217.` |
| 55 | `  -1.diff_col ≤ 12<br>	∧ -1.label not in {<newline>, <space>}<br>	∧ -1.reserved = (<br>	∧ -1.roles not in {KEY, STRING}<br>	∧ +1.reserved = )<br>	∧ +1.roles not in {KEY}<br>	∧ +2.roles not in {EXPRESSION}<br>⇒ y = ∅<br>Confidence: 0.997. Support: 2029.` |
| 56 | `  -1.diff_col ≤ 12<br>	∧ -1.label not in {<newline>, <space>}<br>	∧ -1.reserved not in {., {}<br>	∧ -1.roles not in {KEY, STRING}<br>	∧ -4.label not in {<newline>}<br>	∧ +1.reserved = )<br>	∧ +1.roles not in {COMMENT, KEY}<br>	∧ +2.roles not in {COMMENT, EXPRESSION}<br>	∧ ^1.internal_type not in {VariableDeclaration}<br>⇒ y = ␣<br>Confidence: 0.992. Support: 10635.` |
| 57 | `  •••start_col ≥ 5<br>	∧ -1.diff_col ≤ 12<br>	∧ -1.label not in {<newline>, <space>}<br>	∧ -1.reserved not in {., ;, {}<br>	∧ -1.roles not in {KEY, STRING}<br>	∧ -5.label not in {<newline>}<br>	∧ +1.reserved not in {(, ), ,, ;, }}<br>	∧ +1.roles not in {COMMENT, KEY}<br>	∧ +2.roles not in {COMMENT, EXPRESSION}<br>	∧ ^1.internal_type = ArrayExpression<br>⇒ y = ␣<br>Confidence: 0.945. Support: 425.` |
| 58 | `  •••start_col ≥ 5<br>	∧ -1.diff_col ≤ 12<br>	∧ -1.label not in {<newline>, <space>}<br>	∧ -1.reserved not in {., ;, {}<br>	∧ -1.roles in {IDENTIFIER} and not in {KEY, STRING}<br>	∧ -2.label not in {<-tab>}<br>	∧ -3.reserved not in {(}<br>	∧ +1.reserved not in {(, ), ,, ;, }}<br>	∧ +1.roles not in {COMMENT, KEY}<br>	∧ +1.length ≤ 1<br>	∧ +2.roles not in {COMMENT, EXPRESSION}<br>	∧ ^1.internal_type not in {ArrayExpression, VariableDeclaration}<br>⇒ y = ␣<br>Confidence: 0.992. Support: 713.` |
| 59 | `  •••start_col ≥ 5<br>	∧ -1.diff_col ≤ 12<br>	∧ -1.label not in {<newline>, <space>}<br>	∧ -1.reserved not in {., ;, {}<br>	∧ -1.roles not in {IDENTIFIER, KEY, STRING}<br>	∧ -2.label not in {<-tab>}<br>	∧ -3.reserved not in {(}<br>	∧ +1.reserved not in {(, ), ,, ;, }}<br>	∧ +1.roles not in {COMMENT, KEY}<br>	∧ +1.length ≤ 43<br>	∧ +2.roles not in {COMMENT, EXPRESSION}<br>	∧ ^1.internal_type not in {ArrayExpression, VariableDeclaration}<br>⇒ y = ␣<br>Confidence: 0.963. Support: 27635.` |
| 60 | `  •••start_col ≤ 4<br>	∧ -1.diff_col ≤ 12<br>	∧ -1.label not in {<newline>, <space>}<br>	∧ -1.reserved not in {., ;, {}<br>	∧ -1.roles not in {KEY, STRING}<br>	∧ +1.reserved not in {(, ), ,, ;, }}<br>	∧ +1.roles not in {COMMENT, KEY}<br>	∧ +2.roles not in {COMMENT, EXPRESSION}<br>	∧ ^1.internal_type not in {BlockStatement, VariableDeclaration}<br>	∧ ^2.roles in {STATEMENT}<br>⇒ y = ␣<br>Confidence: 0.991. Support: 168.` |
| 61 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<space>}<br>	∧ +1.reserved = =<br>	∧ +2.roles in {EXPRESSION}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ␣<br>Confidence: 0.999. Support: 923.` |
| 62 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<space>}<br>	∧ +1.internal_type = CommentLine<br>	∧ +1.reserved not in {=, {}<br>	∧ +1.length ≥ 2<br>	∧ +2.roles in {EXPRESSION}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.995. Support: 675.` |
| 63 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved not in {(, if}<br>	∧ +1.reserved = }<br>	∧ +1.length ≤ 1<br>	∧ +2.roles in {EXPRESSION}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ⏎⇥⁻<br>Confidence: 0.970. Support: 152.` |
| 64 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved not in {(, ,, :, if}<br>	∧ -1.roles in {EXPRESSION}<br>	∧ +1.reserved not in {=, {, }}<br>	∧ +1.length ≤ 1<br>	∧ +2.roles in {EXPRESSION}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.999. Support: 24374.` |
| 65 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved not in {(, ,, :, if}<br>	∧ -1.roles not in {EXPRESSION}<br>	∧ -2.length ≥ 3<br>	∧ +1.reserved = ,<br>	∧ +1.length ≤ 1<br>	∧ +2.roles in {EXPRESSION}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.998. Support: 245.` |
| 66 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved not in {(, ,, :, if}<br>	∧ -1.roles not in {EXPRESSION}<br>	∧ -2.length ≤ 2<br>	∧ +1.reserved not in {=, {, }}<br>	∧ +1.length ≤ 1<br>	∧ +2.roles in {EXPRESSION}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.988. Support: 7189.` |
| 67 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved not in {., ;}<br>	∧ +1.reserved = ;<br>	∧ +2.roles not in {EXPRESSION}<br>⇒ y = ∅<br>Confidence: 0.980. Support: 3830.` |
| 68 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved not in {., ;}<br>	∧ -2.length ≤ 1<br>	∧ +1.reserved = (<br>	∧ +2.roles not in {EXPRESSION}<br>⇒ y = ∅<br>Confidence: 0.996. Support: 2563.` |
| 69 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved = {<br>	∧ +1.reserved not in {(}<br>	∧ +2.roles in {COMMENT} and not in {EXPRESSION}<br>	∧ +3.roles not in {VALUE}<br>⇒ y = ⏎⏎⇥⁺<br>Confidence: 0.921. Support: 171.` |
| 70 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.label in {<newline>}<br>	∧ -1.reserved not in {., ;, {}<br>	∧ +1.reserved not in {(, ;}<br>	∧ +2.roles not in {EXPRESSION}<br>⇒ y = "<br>Confidence: 1.000. Support: 1337.` |
| 71 | `  -1.internal_type = CommentLine<br>	∧ -1.label not in {<newline>, <space>}<br>	∧ -1.reserved not in {., ;, {}<br>	∧ +1.reserved not in {(, ;}<br>	∧ +2.roles not in {EXPRESSION}<br>⇒ y = ⏎<br>Confidence: 0.960. Support: 1348.` |
| 72 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved not in {;, {}<br>	∧ +1.roles in {COMMENT}<br>	∧ +2.roles not in {EXPRESSION}<br>⇒ y = ∅<br>Confidence: 0.999. Support: 862.` |
| 73 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved not in {;, {}<br>	∧ +1.reserved = ,<br>	∧ +1.roles not in {KEY}<br>	∧ +2.roles not in {EXPRESSION}<br>⇒ y = ∅<br>Confidence: 0.988. Support: 694.` |
| 74 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved not in {;, {}<br>	∧ -1.roles in {KEY}<br>	∧ +1.reserved not in {,}<br>	∧ +1.roles not in {KEY}<br>	∧ +2.roles not in {EXPRESSION}<br>⇒ y = ∅<br>Confidence: 0.999. Support: 531.` |
| 75 | `  •••start_col ≤ 28<br>	∧ -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved not in {., {}<br>	∧ -1.roles not in {KEY}<br>	∧ +1.reserved = }<br>	∧ +1.roles not in {KEY}<br>	∧ +2.roles not in {EXPRESSION}<br>⇒ y = ⏎⇥⁻<br>Confidence: 0.938. Support: 682.` |
| 76 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved = (<br>	∧ -1.roles not in {KEY}<br>	∧ +1.reserved = )<br>	∧ +1.roles not in {KEY}<br>	∧ +2.roles not in {EXPRESSION}<br>	∧ +2.length ≤ 9<br>⇒ y = ∅<br>Confidence: 0.996. Support: 2008.` |
| 77 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.label not in {<newline>, <space>}<br>	∧ -1.reserved = (<br>	∧ -1.roles not in {KEY}<br>	∧ +1.reserved not in {(, ), ,, }}<br>	∧ +1.roles not in {COMMENT, KEY}<br>	∧ +2.roles not in {EXPRESSION}<br>	∧ +2.length ≤ 9<br>	∧ ^1.internal_type not in {VariableDeclaration}<br>⇒ y = ␣<br>Confidence: 0.987. Support: 10600.` |
| 78 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.label not in {<newline>, <space>}<br>	∧ -1.reserved not in {(, ., ;, {}<br>	∧ -5.label in {<newline>}<br>	∧ +1.reserved not in {(, ,, ;, }}<br>	∧ +2.roles not in {EXPRESSION}<br>	∧ +2.length ≤ 9<br>	∧ ^1.internal_type = ArrayExpression<br>⇒ y = ⏎<br>Confidence: 0.932. Support: 301.` |
| 79 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.label not in {<newline>, <space>}<br>	∧ -1.reserved not in {(, ., ;, {}<br>	∧ -1.roles not in {KEY}<br>	∧ -5.label not in {<newline>}<br>	∧ +1.reserved not in {(, ,, ;, }}<br>	∧ +1.roles not in {COMMENT, KEY}<br>	∧ +2.roles not in {EXPRESSION}<br>	∧ +2.length ≤ 9<br>	∧ ^1.internal_type = ArrayExpression<br>⇒ y = ␣<br>Confidence: 0.927. Support: 515.` |
| 80 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved not in {(, ., ;, {}<br>	∧ -1.roles not in {KEY}<br>	∧ -2.label not in {<-tab>}<br>	∧ -2.length ≤ 4<br>	∧ -4.label in {<newline>}<br>	∧ -5.diff_offset ≥ 7<br>	∧ +1.reserved not in {(, ,, ;, }}<br>	∧ +1.roles not in {KEY}<br>	∧ +1.length ≤ 43<br>	∧ +2.roles not in {EXPRESSION}<br>	∧ +2.length ≤ 9<br>	∧ ^1.internal_type not in {ArrayExpression, VariableDeclaration}<br>⇒ y = ␣<br>Confidence: 0.984. Support: 1825.` |
| 81 | `  -1.internal_type not in {CommentLine, Identifier, StringLiteral}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved not in {(, ., ;, {}<br>	∧ -1.roles not in {KEY}<br>	∧ -2.label not in {<-tab>}<br>	∧ -4.label not in {<newline>}<br>	∧ -5.diff_offset ≥ 7<br>	∧ +1.reserved not in {(, ,, ;, }}<br>	∧ +1.roles not in {KEY}<br>	∧ +1.length ≤ 43<br>	∧ +2.roles not in {EXPRESSION}<br>	∧ +2.length ≤ 9<br>	∧ ^1.internal_type not in {ArrayExpression, VariableDeclaration}<br>⇒ y = ␣<br>Confidence: 0.951. Support: 10720.` |
| 82 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved not in {(, ., ;, {}<br>	∧ -1.roles not in {KEY}<br>	∧ -2.label not in {<-tab>}<br>	∧ -4.label not in {<newline>}<br>	∧ -5.diff_offset ≥ 7<br>	∧ +1.reserved not in {(, ,, ;, }}<br>	∧ +1.roles not in {KEY}<br>	∧ +1.length ≤ 1<br>	∧ +2.roles not in {EXPRESSION}<br>	∧ +2.length ≤ 9<br>	∧ ^1.internal_type not in {ArrayExpression, VariableDeclaration}<br>⇒ y = ␣<br>Confidence: 0.990. Support: 12958.` |
| 83 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved not in {(, ., ;, {}<br>	∧ -1.roles not in {KEY}<br>	∧ -2.label not in {<-tab>}<br>	∧ -3.reserved not in {(}<br>	∧ -5.diff_offset ≤ 6<br>	∧ +1.reserved not in {(, ,, ;, }}<br>	∧ +1.roles not in {KEY}<br>	∧ +1.length ≤ 43<br>	∧ +2.roles not in {EXPRESSION}<br>	∧ +2.length ≤ 9<br>	∧ ^1.internal_type not in {ArrayExpression, VariableDeclaration}<br>⇒ y = ␣<br>Confidence: 0.933. Support: 2423.` |
| 84 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<space>}<br>	∧ +1.reserved not in {(}<br>	∧ +2.roles in {EXPRESSION}<br>	∧ ^1.roles in {OPERATOR}<br>⇒ y = ␣<br>Confidence: 0.987. Support: 3489.` |
| 85 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<space>}<br>	∧ +1.reserved not in {=, {}<br>	∧ +1.roles in {COMMENT}<br>	∧ +1.length ≥ 2<br>	∧ +2.roles in {EXPRESSION}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.998. Support: 657.` |
| 86 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved not in {(, ,, if}<br>	∧ +1.reserved = }<br>	∧ +1.length ≤ 1<br>	∧ +2.roles in {EXPRESSION}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ⏎⇥⁻<br>Confidence: 0.965. Support: 157.` |
| 87 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved not in {(, ,, if}<br>	∧ -1.roles in {EXPRESSION}<br>	∧ +1.reserved not in {=, {, }}<br>	∧ +1.length ≤ 1<br>	∧ +2.roles in {EXPRESSION}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.999. Support: 23958.` |
| 88 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved not in {(, ,, if}<br>	∧ -1.roles not in {EXPRESSION}<br>	∧ -2.label in {<-tab>}<br>	∧ -2.length ≥ 2<br>	∧ +1.reserved not in {=, {, }}<br>	∧ +1.length ≤ 1<br>	∧ +2.roles in {EXPRESSION}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.970. Support: 511.` |
| 89 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved not in {(, ,, if}<br>	∧ -1.roles not in {EXPRESSION}<br>	∧ -2.length ≤ 1<br>	∧ +1.reserved not in {=, {, }}<br>	∧ +1.length ≤ 1<br>	∧ +2.roles in {EXPRESSION}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.957. Support: 9113.` |
| 90 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved not in {., ;}<br>	∧ -2.label not in {<newline>}<br>	∧ +1.reserved = (<br>	∧ +2.roles not in {EXPRESSION}<br>⇒ y = ∅<br>Confidence: 0.990. Support: 2679.` |
| 91 | `  -1.diff_col ≤ 12<br>	∧ -1.internal_type not in {Identifier, StringLiteral}<br>	∧ -1.label not in {<newline>, <space>}<br>	∧ -1.reserved not in {(, ;, {}<br>	∧ -1.roles not in {KEY}<br>	∧ -2.label not in {<-tab>}<br>	∧ -3.reserved = (<br>	∧ -4.diff_offset ≤ 5<br>	∧ +1.reserved not in {,, }}<br>	∧ +1.roles not in {KEY}<br>	∧ +1.length ≥ 2<br>	∧ +2.roles not in {EXPRESSION}<br>	∧ ^1.internal_type not in {ArrayExpression}<br>⇒ y = ∅<br>Confidence: 0.991. Support: 171.` |
| 92 | `  -1.diff_col ≤ 12<br>	∧ -1.internal_type not in {Identifier, StringLiteral}<br>	∧ -1.label not in {<newline>, <space>}<br>	∧ -1.reserved not in {(, ., ;, {}<br>	∧ -1.roles not in {KEY}<br>	∧ -2.label not in {<-tab>}<br>	∧ -3.reserved not in {(}<br>	∧ +1.reserved not in {(, ,, ;, }}<br>	∧ +1.roles not in {KEY}<br>	∧ +1.length ≥ 2<br>	∧ +2.roles not in {EXPRESSION}<br>	∧ ^1.internal_type not in {ArrayExpression, VariableDeclaration}<br>⇒ y = ␣<br>Confidence: 0.936. Support: 13963.` |
| 93 | `  -1.diff_col ≤ 12<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<newline>, <space>}<br>	∧ -1.reserved not in {(, ., ;, {}<br>	∧ -1.roles not in {KEY}<br>	∧ -2.length ≤ 4<br>	∧ -4.label in {<newline>}<br>	∧ +1.reserved not in {(, ,, ;, }}<br>	∧ +1.roles not in {KEY}<br>	∧ +1.length ≤ 1<br>	∧ +2.roles not in {EXPRESSION}<br>	∧ ^1.internal_type not in {ArrayExpression, VariableDeclaration}<br>⇒ y = ␣<br>Confidence: 0.965. Support: 440.` |
| 94 | `  -1.label not in {<space>}<br>	∧ -1.roles not in {STRING}<br>	∧ +2.roles in {EXPRESSION}<br>	∧ ^1.roles in {OPERATOR}<br>⇒ y = ␣<br>Confidence: 0.965. Support: 3439.` |
| 95 | `  -1.label not in {<space>}<br>	∧ -1.roles not in {STRING}<br>	∧ +1.reserved not in {{}<br>	∧ +2.roles in {EXPRESSION}<br>	∧ ^1.internal_type = VariableDeclarator<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ␣<br>Confidence: 0.964. Support: 1077.` |
| 96 | `  -1.label not in {<space>}<br>	∧ -1.roles not in {STRING}<br>	∧ +1.reserved not in {{}<br>	∧ +1.roles in {COMMENT}<br>	∧ +1.length ≥ 2<br>	∧ +2.roles in {EXPRESSION}<br>	∧ ^1.internal_type not in {VariableDeclarator}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.998. Support: 699.` |
| 97 | `  -1.label not in {<space>}<br>	∧ -1.reserved not in {(, if}<br>	∧ -1.roles not in {STRING}<br>	∧ +1.reserved = }<br>	∧ +1.length ≤ 1<br>	∧ +2.roles in {EXPRESSION}<br>	∧ ^1.internal_type not in {VariableDeclarator}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ⏎⇥⁻<br>Confidence: 0.984. Support: 161.` |
| 98 | `  -1.label not in {<space>}<br>	∧ -1.reserved not in {(, ,, :, if}<br>	∧ -1.roles in {EXPRESSION} and not in {STRING}<br>	∧ +1.reserved not in {{, }}<br>	∧ +1.length ≤ 1<br>	∧ +2.roles in {EXPRESSION}<br>	∧ ^1.internal_type not in {VariableDeclarator}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.999. Support: 23987.` |
| 99 | `  -1.label not in {<space>}<br>	∧ -1.reserved not in {(, ,, :, if}<br>	∧ -1.roles not in {EXPRESSION, STRING}<br>	∧ -1.length ≤ 1<br>	∧ -2.length ≥ 2<br>	∧ +1.reserved not in {{, }}<br>	∧ +1.length ≤ 1<br>	∧ +2.roles in {EXPRESSION}<br>	∧ ^1.internal_type not in {VariableDeclarator}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.999. Support: 506.` |
| 100 | `  -1.label not in {<space>}<br>	∧ -1.reserved not in {(, ,, :, if}<br>	∧ -1.roles not in {EXPRESSION, STRING}<br>	∧ -2.length ≤ 1<br>	∧ +1.reserved not in {{, }}<br>	∧ +1.length ≤ 1<br>	∧ +2.roles in {EXPRESSION}<br>	∧ ^1.internal_type not in {VariableDeclarator}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.994. Support: 6911.` |
| 101 | `  -1.label not in {<space>}<br>	∧ -1.reserved not in {., ;}<br>	∧ -1.roles not in {STRING}<br>	∧ -3.diff_line = 0<br>	∧ +1.reserved = (<br>	∧ +2.roles not in {EXPRESSION}<br>⇒ y = ∅<br>Confidence: 0.993. Support: 2719.` |
| 102 | `  •••start_col ≤ 31<br>	∧ -1.diff_col ≤ 12<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved not in {., {}<br>	∧ -1.roles not in {KEY, STRING}<br>	∧ +1.reserved = }<br>	∧ +1.roles not in {COMMENT, KEY}<br>	∧ +2.roles not in {EXPRESSION}<br>⇒ y = ⏎⇥⁻<br>Confidence: 0.929. Support: 679.` |
| 103 | `  -1.diff_col ≤ 12<br>	∧ -1.label in {<newline>} and not in {<space>}<br>	∧ -1.reserved not in {., ;, {}<br>	∧ +1.reserved not in {(, ,, ;, }}<br>	∧ +2.roles not in {EXPRESSION}<br>	∧ +2.length ≤ 9<br>⇒ y = "<br>Confidence: 0.998. Support: 234.` |
| 104 | `  -1.diff_col ≤ 12<br>	∧ -1.label not in {<newline>, <space>}<br>	∧ -1.reserved = (<br>	∧ -1.roles not in {STRING}<br>	∧ +1.reserved = )<br>	∧ +1.roles not in {KEY}<br>	∧ +2.roles not in {EXPRESSION}<br>	∧ +2.length ≤ 9<br>⇒ y = ∅<br>Confidence: 0.998. Support: 1935.` |
| 105 | `  -1.diff_col ≤ 12<br>	∧ -1.label not in {<newline>, <space>}<br>	∧ -1.reserved = (<br>	∧ -1.roles not in {KEY, STRING}<br>	∧ +1.reserved not in {(, ), ,, }}<br>	∧ +1.roles not in {COMMENT, KEY}<br>	∧ +2.roles not in {EXPRESSION}<br>	∧ +2.length ≤ 9<br>	∧ ^1.internal_type not in {VariableDeclaration}<br>⇒ y = ␣<br>Confidence: 0.986. Support: 10713.` |
| 106 | `  -1.diff_col ≤ 12<br>	∧ -1.label not in {<newline>, <space>}<br>	∧ -1.reserved not in {(, ., ;, {}<br>	∧ -1.roles not in {KEY, STRING}<br>	∧ -5.label not in {<newline>}<br>	∧ +1.reserved not in {(, ,, ;, }}<br>	∧ +1.roles not in {KEY}<br>	∧ +2.roles not in {EXPRESSION}<br>	∧ +2.length ≤ 9<br>	∧ ^1.internal_type = ArrayExpression<br>⇒ y = ␣<br>Confidence: 0.945. Support: 498.` |
| 107 | `  -1.diff_col ≤ 12<br>	∧ -1.label not in {<newline>, <space>}<br>	∧ -1.reserved = }<br>	∧ -1.roles not in {KEY, STRING}<br>	∧ +1.reserved not in {,, }}<br>	∧ +1.roles not in {KEY}<br>	∧ +1.length ≥ 2<br>	∧ +2.roles not in {EXPRESSION}<br>	∧ +2.length ≤ 9<br>	∧ ^1.internal_type not in {ArrayExpression, VariableDeclaration}<br>	∧ ^1.roles not in {SCOPE}<br>⇒ y = ␣<br>Confidence: 0.924. Support: 190.` |
| 108 | `  -1.diff_col ≤ 12<br>	∧ -1.label not in {<newline>, <space>}<br>	∧ -1.reserved not in {(, ;, {, }}<br>	∧ -1.roles not in {IDENTIFIER, STRING}<br>	∧ -3.reserved = (<br>	∧ -5.diff_offset ≤ 6<br>	∧ +1.reserved not in {,, }}<br>	∧ +1.roles not in {KEY}<br>	∧ +1.length ≥ 2<br>	∧ +2.roles not in {EXPRESSION}<br>	∧ +2.length ≤ 9<br>	∧ ^1.internal_type not in {ArrayExpression}<br>⇒ y = ∅<br>Confidence: 0.963. Support: 175.` |
| 109 | `  -1.diff_col ≤ 12<br>	∧ -1.label not in {<newline>, <space>}<br>	∧ -1.reserved not in {(, ., ;, {, }}<br>	∧ -1.roles not in {IDENTIFIER, KEY, STRING}<br>	∧ -3.reserved not in {(}<br>	∧ +1.reserved not in {(, ,, ;, }}<br>	∧ +1.roles not in {KEY}<br>	∧ +1.length ≥ 2<br>	∧ +2.roles not in {EXPRESSION}<br>	∧ +2.length ≤ 9<br>	∧ ^1.internal_type not in {ArrayExpression, VariableDeclaration}<br>⇒ y = ␣<br>Confidence: 0.938. Support: 14176.` |
| 110 | `  -1.diff_col ≤ 12<br>	∧ -1.label not in {<newline>, <space>}<br>	∧ -1.reserved not in {(, ., ;, {}<br>	∧ -1.roles not in {KEY, STRING}<br>	∧ -4.label in {<newline>}<br>	∧ +1.reserved not in {(, ,, ;, }}<br>	∧ +1.roles not in {KEY}<br>	∧ +1.length ≤ 1<br>	∧ +2.roles not in {EXPRESSION}<br>	∧ +2.length ≤ 9<br>	∧ ^1.internal_type not in {ArrayExpression, VariableDeclaration}<br>⇒ y = ␣<br>Confidence: 0.956. Support: 461.` |
| 111 | `  -1.diff_col ≤ 12<br>	∧ -1.label not in {<newline>, <space>}<br>	∧ -1.reserved not in {(, ., ;, {}<br>	∧ -1.roles not in {KEY, STRING}<br>	∧ -4.label not in {<newline>}<br>	∧ +1.reserved not in {(, ,, ;, }}<br>	∧ +1.roles not in {KEY}<br>	∧ +1.length ≤ 1<br>	∧ +2.roles not in {EXPRESSION}<br>	∧ +2.length ≤ 9<br>	∧ ^1.internal_type not in {ArrayExpression, VariableDeclaration}<br>⇒ y = ␣<br>Confidence: 0.982. Support: 15331.` |
| 112 | `  -1.label not in {<space>}<br>	∧ -1.roles not in {STRING}<br>	∧ +1.reserved not in {=, {}<br>	∧ +1.roles in {COMMENT}<br>	∧ +1.length ≥ 2<br>	∧ +2.roles in {EXPRESSION}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.998. Support: 655.` |
| 113 | `  -1.label not in {<space>}<br>	∧ -1.reserved not in {(, ,, if}<br>	∧ -1.roles in {EXPRESSION} and not in {STRING}<br>	∧ +1.reserved not in {=, {, }}<br>	∧ +1.length ≤ 1<br>	∧ +2.roles in {EXPRESSION}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.998. Support: 24217.` |
| 114 | `  -1.label not in {<space>}<br>	∧ -1.reserved not in {(, ,, if}<br>	∧ -1.roles not in {EXPRESSION, STRING}<br>	∧ -2.label in {<-tab>}<br>	∧ -2.length ≥ 3<br>	∧ +1.reserved not in {=, {, }}<br>	∧ +1.length ≤ 1<br>	∧ +2.roles in {EXPRESSION}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.955. Support: 275.` |
| 115 | `  -1.label not in {<space>}<br>	∧ -1.reserved not in {(, ,, if}<br>	∧ -1.roles not in {EXPRESSION, STRING}<br>	∧ -2.length ≤ 2<br>	∧ +1.reserved not in {=, {, }}<br>	∧ +1.length ≤ 1<br>	∧ +2.roles in {EXPRESSION}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.956. Support: 9419.` |
| 116 | `  -1.label not in {<space>}<br>	∧ -1.reserved = {<br>	∧ -1.roles not in {STRING}<br>	∧ +1.reserved not in {(}<br>	∧ +2.roles in {COMMENT} and not in {EXPRESSION}<br>	∧ +3.roles not in {VALUE}<br>⇒ y = ⏎⏎⇥⁺<br>Confidence: 0.923. Support: 150.` |
| 117 | `  -1.diff_col ≤ 12<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved not in {., ;, {}<br>	∧ -1.roles not in {STRING}<br>	∧ +1.reserved = ,<br>	∧ +2.roles not in {EXPRESSION}<br>⇒ y = ∅<br>Confidence: 0.993. Support: 749.` |
| 118 | `  -1.diff_col ≤ 12<br>	∧ -1.label in {<newline>} and not in {<space>}<br>	∧ -1.reserved not in {., ;, {}<br>	∧ +1.reserved not in {(, ,, ;, }}<br>	∧ +2.roles not in {EXPRESSION}<br>	∧ +2.length ≤ 8<br>⇒ y = "<br>Confidence: 0.998. Support: 230.` |
| 119 | `  -1.diff_col ≤ 12<br>	∧ -1.label not in {<newline>, <space>}<br>	∧ -1.reserved = (<br>	∧ -1.roles not in {STRING}<br>	∧ +1.reserved = )<br>	∧ +1.roles not in {KEY}<br>	∧ +2.roles not in {EXPRESSION}<br>	∧ +2.length ≤ 8<br>⇒ y = ∅<br>Confidence: 0.998. Support: 1930.` |
| 120 | `  -1.diff_col ≤ 12<br>	∧ -1.label not in {<newline>, <space>}<br>	∧ -1.reserved = (<br>	∧ -1.roles not in {KEY, STRING}<br>	∧ +1.reserved not in {(, ), ,, }}<br>	∧ +1.roles not in {COMMENT, KEY}<br>	∧ +2.roles not in {EXPRESSION}<br>	∧ +2.length ≤ 8<br>	∧ ^1.internal_type not in {VariableDeclaration}<br>⇒ y = ␣<br>Confidence: 0.984. Support: 10414.` |
| 121 | `  -1.diff_col ≤ 12<br>	∧ -1.label not in {<newline>, <space>}<br>	∧ -1.reserved not in {(, ., ;, {}<br>	∧ -1.roles not in {KEY, STRING}<br>	∧ -5.label in {<newline>}<br>	∧ +1.reserved not in {(, ,, ;, }}<br>	∧ +2.roles not in {EXPRESSION}<br>	∧ +2.length ≤ 8<br>	∧ ^1.internal_type = ArrayExpression<br>⇒ y = ⏎<br>Confidence: 0.935. Support: 330.` |
| 122 | `  -1.diff_col ≤ 12<br>	∧ -1.label not in {<newline>, <space>}<br>	∧ -1.reserved not in {(, ., ;, {}<br>	∧ -1.roles not in {KEY, STRING}<br>	∧ -5.label not in {<newline>}<br>	∧ +1.reserved not in {(, ,, ;, }}<br>	∧ +1.roles not in {KEY}<br>	∧ +2.roles not in {EXPRESSION}<br>	∧ +2.length ≤ 8<br>	∧ ^1.internal_type = ArrayExpression<br>⇒ y = ␣<br>Confidence: 0.925. Support: 524.` |
| 123 | `  -1.diff_col ≤ 12<br>	∧ -1.label not in {<newline>, <space>}<br>	∧ -1.reserved = }<br>	∧ -1.roles not in {KEY, STRING}<br>	∧ +1.reserved not in {,, }}<br>	∧ +1.roles not in {KEY}<br>	∧ +1.length ≥ 2<br>	∧ +2.roles not in {EXPRESSION}<br>	∧ +2.length ≤ 8<br>	∧ ^1.internal_type not in {ArrayExpression, BlockStatement, VariableDeclaration}<br>⇒ y = ␣<br>Confidence: 0.949. Support: 185.` |
| 124 | `  -1.diff_col ≤ 12<br>	∧ -1.label not in {<newline>, <space>}<br>	∧ -1.reserved not in {(, ;, {, }}<br>	∧ -1.roles not in {IDENTIFIER, STRING}<br>	∧ -3.reserved = (<br>	∧ -4.diff_offset ≤ 5<br>	∧ +1.reserved not in {,, }}<br>	∧ +1.roles not in {KEY}<br>	∧ +1.length ≥ 2<br>	∧ +2.roles not in {EXPRESSION}<br>	∧ +2.length ≤ 8<br>	∧ ^1.internal_type not in {ArrayExpression}<br>⇒ y = ∅<br>Confidence: 0.981. Support: 185.` |
| 125 | `  -1.diff_col ≤ 12<br>	∧ -1.label not in {<newline>, <space>}<br>	∧ -1.reserved not in {(, ., ;, {, }}<br>	∧ -1.roles not in {IDENTIFIER, KEY, STRING}<br>	∧ -3.reserved not in {(}<br>	∧ +1.reserved not in {(, ,, ;, }}<br>	∧ +1.roles not in {KEY}<br>	∧ +1.length ≥ 2<br>	∧ +2.roles not in {EXPRESSION}<br>	∧ +2.length ≤ 8<br>	∧ ^1.internal_type not in {ArrayExpression, VariableDeclaration}<br>⇒ y = ␣<br>Confidence: 0.939. Support: 13711.` |
| 126 | `  -1.diff_col ≤ 12<br>	∧ -1.label not in {<newline>, <space>}<br>	∧ -1.reserved not in {(, ., ;, {}<br>	∧ -1.roles not in {KEY, STRING}<br>	∧ -2.length ≤ 3<br>	∧ -4.label in {<newline>}<br>	∧ +1.reserved not in {(, ,, ;, }}<br>	∧ +1.roles not in {KEY}<br>	∧ +1.length ≤ 1<br>	∧ +2.roles not in {EXPRESSION}<br>	∧ +2.length ≤ 8<br>	∧ ^1.internal_type not in {ArrayExpression, VariableDeclaration}<br>⇒ y = ␣<br>Confidence: 0.957. Support: 456.` |
| 127 | `  -1.diff_col ≤ 12<br>	∧ -1.label not in {<newline>, <space>}<br>	∧ -1.reserved not in {(, ., ;, {}<br>	∧ -1.roles not in {KEY, STRING}<br>	∧ -4.label not in {<newline>}<br>	∧ +1.reserved not in {(, ,, ;, }}<br>	∧ +1.roles not in {KEY}<br>	∧ +1.length ≤ 1<br>	∧ +2.roles not in {EXPRESSION}<br>	∧ +2.length ≤ 8<br>	∧ ^1.internal_type not in {ArrayExpression, VariableDeclaration}<br>⇒ y = ␣<br>Confidence: 0.983. Support: 15633.` |
| 128 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<space>}<br>	∧ +1.internal_type = CommentLine<br>	∧ +1.reserved not in {{}<br>	∧ +1.length ≥ 2<br>	∧ +2.roles in {EXPRESSION}<br>	∧ ^1.internal_type not in {VariableDeclarator}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.999. Support: 653.` |
| 129 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved not in {(, ,, if}<br>	∧ -1.roles in {EXPRESSION}<br>	∧ +1.reserved not in {{, }}<br>	∧ +1.length ≤ 1<br>	∧ +2.roles in {EXPRESSION}<br>	∧ ^1.internal_type not in {VariableDeclarator}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.999. Support: 23922.` |
| 130 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved not in {(, ,, if}<br>	∧ -1.roles not in {EXPRESSION}<br>	∧ -2.label in {<-tab>}<br>	∧ -2.length ≥ 2<br>	∧ +1.reserved not in {{, }}<br>	∧ +1.length ≤ 1<br>	∧ +2.roles in {EXPRESSION}<br>	∧ ^1.internal_type not in {VariableDeclarator}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.974. Support: 488.` |
| 131 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved not in {(, ,, if}<br>	∧ -1.roles not in {EXPRESSION}<br>	∧ -2.length ≤ 1<br>	∧ +1.reserved not in {{, }}<br>	∧ +1.length ≤ 1<br>	∧ +2.roles in {EXPRESSION}<br>	∧ ^1.internal_type not in {VariableDeclarator}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.960. Support: 9129.` |
| 132 | `  •••start_col ≥ 7<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved not in {., ;}<br>	∧ +1.reserved = (<br>	∧ +2.roles not in {EXPRESSION}<br>⇒ y = ∅<br>Confidence: 0.975. Support: 2753.` |
| 133 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.label not in {<newline>, <space>}<br>	∧ -1.reserved = (<br>	∧ -1.roles not in {KEY}<br>	∧ +1.reserved = )<br>	∧ +1.roles not in {KEY}<br>	∧ +2.roles not in {EXPRESSION}<br>⇒ y = ∅<br>Confidence: 0.997. Support: 1983.` |
| 134 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.label not in {<newline>, <space>}<br>	∧ -1.reserved = (<br>	∧ -1.roles not in {KEY}<br>	∧ +1.reserved not in {(, ), ,, }}<br>	∧ +1.roles not in {COMMENT, KEY}<br>	∧ +2.roles not in {COMMENT, EXPRESSION}<br>	∧ ^1.internal_type not in {VariableDeclaration}<br>⇒ y = ␣<br>Confidence: 0.984. Support: 10790.` |
| 135 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved not in {(, ;, {, }}<br>	∧ -1.roles not in {IDENTIFIER, KEY}<br>	∧ -3.reserved = (<br>	∧ -4.diff_offset ≤ 5<br>	∧ +1.reserved not in {,, }}<br>	∧ +1.roles not in {KEY}<br>	∧ +1.length ≥ 2<br>	∧ +2.roles not in {EXPRESSION}<br>	∧ ^1.internal_type not in {ArrayExpression}<br>⇒ y = ∅<br>Confidence: 0.998. Support: 224.` |
| 136 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved not in {(, ., ;, {, }}<br>	∧ -1.roles not in {IDENTIFIER, KEY}<br>	∧ -3.reserved not in {(}<br>	∧ +1.reserved not in {(, ,, ;, }}<br>	∧ +1.roles not in {KEY}<br>	∧ +1.length ≥ 2<br>	∧ +2.roles not in {EXPRESSION}<br>	∧ ^1.internal_type not in {ArrayExpression, VariableDeclaration}<br>⇒ y = ␣<br>Confidence: 0.940. Support: 13761.` |
| 137 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved not in {(, ., ;, {}<br>	∧ -1.roles not in {KEY}<br>	∧ -4.label in {<newline>}<br>	∧ +1.reserved not in {(, ,, ;, }}<br>	∧ +1.roles not in {KEY}<br>	∧ +1.length ≤ 1<br>	∧ +2.roles not in {EXPRESSION}<br>	∧ ^1.internal_type not in {ArrayExpression, VariableDeclaration}<br>⇒ y = ␣<br>Confidence: 0.952. Support: 510.` |
| 138 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved not in {(, ., ;, {}<br>	∧ -1.roles not in {KEY}<br>	∧ -4.label not in {<newline>}<br>	∧ +1.reserved not in {(, ,, ;, }}<br>	∧ +1.roles not in {KEY}<br>	∧ +1.length ≤ 1<br>	∧ +2.roles not in {EXPRESSION}<br>	∧ ^1.internal_type not in {ArrayExpression, VariableDeclaration}<br>⇒ y = ␣<br>Confidence: 0.983. Support: 15366.` |
| 139 | `  -1.label not in {<space>}<br>	∧ -1.reserved not in {(, if}<br>	∧ -1.roles not in {STRING}<br>	∧ +1.reserved = }<br>	∧ +1.length ≤ 1<br>	∧ +2.roles in {EXPRESSION}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ⏎⇥⁻<br>Confidence: 0.980. Support: 173.` |
| 140 | `  -1.label not in {<space>}<br>	∧ -1.reserved not in {., ;}<br>	∧ -1.roles not in {STRING}<br>	∧ -2.length ≤ 1<br>	∧ +1.reserved = (<br>	∧ +2.roles not in {EXPRESSION}<br>⇒ y = ∅<br>Confidence: 0.992. Support: 2732.` |
| 141 | `  -1.label in {<newline>} and not in {<space>}<br>	∧ -1.reserved not in {., ;, {}<br>	∧ +1.reserved not in {(, ;}<br>	∧ +2.roles not in {EXPRESSION}<br>⇒ y = "<br>Confidence: 1.000. Support: 1354.` |
| 142 | `  -1.internal_type = CommentLine<br>	∧ -1.label not in {<newline>, <space>}<br>	∧ -1.reserved not in {., ;, {}<br>	∧ -1.roles not in {STRING}<br>	∧ +1.reserved not in {(, ;}<br>	∧ +2.roles not in {EXPRESSION}<br>⇒ y = ⏎<br>Confidence: 0.971. Support: 1311.` |
| 143 | `  -1.internal_type not in {CommentLine}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved not in {;, {}<br>	∧ -1.roles not in {STRING}<br>	∧ +1.roles in {COMMENT}<br>	∧ +2.roles not in {EXPRESSION}<br>⇒ y = ∅<br>Confidence: 0.999. Support: 823.` |
| 144 | `  -1.internal_type not in {CommentLine}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved not in {;, {}<br>	∧ -1.roles not in {STRING}<br>	∧ +1.reserved = ,<br>	∧ +1.roles not in {KEY}<br>	∧ +2.roles not in {EXPRESSION}<br>⇒ y = ∅<br>Confidence: 0.991. Support: 685.` |
| 145 | `  -1.internal_type not in {CommentLine}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved not in {;, {}<br>	∧ -1.roles in {KEY} and not in {STRING}<br>	∧ +1.reserved not in {,}<br>	∧ +1.roles not in {KEY}<br>	∧ +2.roles not in {EXPRESSION}<br>⇒ y = ∅<br>Confidence: 0.999. Support: 543.` |
| 146 | `  -1.internal_type not in {CommentLine}<br>	∧ -1.label not in {<newline>, <space>}<br>	∧ -1.reserved = (<br>	∧ -1.roles not in {KEY, STRING}<br>	∧ +1.reserved = )<br>	∧ +1.roles not in {KEY}<br>	∧ +2.roles not in {EXPRESSION}<br>⇒ y = ∅<br>Confidence: 0.998. Support: 2102.` |
| 147 | `  -1.internal_type not in {CommentLine}<br>	∧ -1.label not in {<newline>, <space>}<br>	∧ -1.reserved = (<br>	∧ -1.roles not in {KEY, STRING}<br>	∧ +1.reserved not in {(, ), ,, }}<br>	∧ +1.roles not in {COMMENT, KEY}<br>	∧ +2.roles not in {COMMENT, EXPRESSION}<br>	∧ ^1.internal_type not in {VariableDeclaration}<br>⇒ y = ␣<br>Confidence: 0.988. Support: 10470.` |
| 148 | `  -1.internal_type not in {CommentLine}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved not in {(, ., ;, {}<br>	∧ -1.roles not in {KEY, STRING}<br>	∧ -5.label not in {<newline>}<br>	∧ +1.reserved not in {(, ,, ;, }}<br>	∧ +1.roles not in {KEY}<br>	∧ +2.roles not in {EXPRESSION}<br>	∧ ^1.internal_type not in {VariableDeclaration}<br>	∧ ^1.roles in {LIST}<br>⇒ y = ␣<br>Confidence: 0.931. Support: 501.` |
| 149 | `  -1.internal_type not in {CommentLine}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved not in {(, ., ;, {}<br>	∧ -1.roles not in {KEY, STRING}<br>	∧ -2.label in {<-tab>}<br>	∧ +1.reserved not in {(, ,, ;, }}<br>	∧ +1.roles not in {KEY}<br>	∧ +1.length ≥ 2<br>	∧ +2.roles not in {EXPRESSION}<br>	∧ ^1.internal_type not in {VariableDeclaration}<br>	∧ ^1.roles not in {BLOCK, LIST}<br>⇒ y = ␣<br>Confidence: 0.929. Support: 133.` |
| 150 | `  -1.internal_type not in {CommentLine, Identifier}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved not in {(, ., ;, {}<br>	∧ -1.roles not in {KEY, STRING}<br>	∧ -2.label not in {<-tab>}<br>	∧ -3.reserved = (<br>	∧ -4.diff_offset ≥ 6<br>	∧ +1.reserved not in {(, ,, ;, }}<br>	∧ +1.roles not in {KEY}<br>	∧ +1.length ≥ 2<br>	∧ +2.roles not in {EXPRESSION}<br>	∧ ^1.internal_type not in {VariableDeclaration}<br>	∧ ^1.roles not in {LIST}<br>⇒ y = ␣<br>Confidence: 0.939. Support: 483.` |
| 151 | `  -1.internal_type not in {CommentLine, Identifier}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved not in {(, ;, {}<br>	∧ -1.roles not in {KEY, STRING}<br>	∧ -2.label not in {<-tab>}<br>	∧ -3.reserved = (<br>	∧ -4.diff_offset ≤ 5<br>	∧ +1.reserved not in {,, }}<br>	∧ +1.roles not in {KEY}<br>	∧ +1.length ≥ 2<br>	∧ +2.roles not in {EXPRESSION}<br>	∧ ^1.roles not in {LIST}<br>⇒ y = ∅<br>Confidence: 0.986. Support: 176.` |
| 152 | `  -1.internal_type not in {CommentLine, Identifier}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved not in {(, ., ;, {}<br>	∧ -1.roles not in {KEY, STRING}<br>	∧ -2.label not in {<-tab>}<br>	∧ -3.reserved not in {(}<br>	∧ +1.reserved not in {(, ,, ;, }}<br>	∧ +1.roles not in {KEY}<br>	∧ +1.length ≥ 2<br>	∧ +2.roles not in {EXPRESSION}<br>	∧ ^1.internal_type not in {VariableDeclaration}<br>	∧ ^1.roles not in {LIST}<br>⇒ y = ␣<br>Confidence: 0.942. Support: 13800.` |
| 153 | `  -1.internal_type not in {CommentLine}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved not in {(, ., ;, {}<br>	∧ -1.roles not in {KEY, STRING}<br>	∧ -2.length ≤ 5<br>	∧ -4.label in {<newline>}<br>	∧ +1.reserved not in {(, ,, ;, }}<br>	∧ +1.roles not in {KEY}<br>	∧ +1.length ≤ 1<br>	∧ +2.roles not in {EXPRESSION}<br>	∧ ^1.internal_type not in {VariableDeclaration}<br>	∧ ^1.roles not in {LIST}<br>⇒ y = ␣<br>Confidence: 0.959. Support: 501.` |
| 154 | `  -1.internal_type not in {CommentLine}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved not in {(, ., ;, {}<br>	∧ -1.roles not in {KEY, STRING}<br>	∧ -4.label not in {<newline>}<br>	∧ +1.reserved not in {(, ,, ;, }}<br>	∧ +1.roles not in {KEY}<br>	∧ +1.length ≤ 1<br>	∧ +2.roles not in {EXPRESSION}<br>	∧ ^1.internal_type not in {VariableDeclaration}<br>	∧ ^1.roles not in {LIST}<br>⇒ y = ␣<br>Confidence: 0.984. Support: 15601.` |
| 155 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<space>}<br>	∧ +1.reserved not in {{}<br>	∧ +2.roles in {EXPRESSION}<br>	∧ ^1.roles not in {OPERATOR}<br>	∧ ^2.internal_type = VariableDeclaration<br>⇒ y = ␣<br>Confidence: 0.963. Support: 1008.` |
| 156 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<space>}<br>	∧ +1.internal_type = CommentLine<br>	∧ +1.reserved not in {{}<br>	∧ +1.length ≥ 2<br>	∧ +2.roles in {EXPRESSION}<br>	∧ ^1.roles not in {OPERATOR}<br>	∧ ^2.internal_type not in {VariableDeclaration}<br>⇒ y = ∅<br>Confidence: 0.998. Support: 698.` |
| 157 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved not in {if}<br>	∧ -2.roles not in {EXPRESSION}<br>	∧ +1.length ≤ 1<br>	∧ +2.roles in {EXPRESSION}<br>	∧ ^1.internal_type = ConditionalExpression<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ␣<br>Confidence: 0.966. Support: 220.` |
| 158 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved = (<br>	∧ -2.roles not in {EXPRESSION}<br>	∧ +1.length ≤ 1<br>	∧ +2.roles in {EXPRESSION}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ␣<br>Confidence: 0.996. Support: 142.` |
| 159 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved not in {(, if}<br>	∧ -2.roles not in {EXPRESSION}<br>	∧ -3.diff_col ≥ 5<br>	∧ -3.internal_type not in {StringLiteral}<br>	∧ -3.label in {<space>}<br>	∧ +1.reserved not in {{}<br>	∧ +1.length ≤ 1<br>	∧ +2.roles in {EXPRESSION}<br>	∧ ^1.roles not in {OPERATOR}<br>	∧ ^2.internal_type not in {VariableDeclaration}<br>⇒ y = ∅<br>Confidence: 0.997. Support: 171.` |
| 160 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved not in {(, if}<br>	∧ -1.roles in {EXPRESSION}<br>	∧ -2.roles not in {EXPRESSION}<br>	∧ -3.internal_type not in {StringLiteral}<br>	∧ -3.label not in {<space>}<br>	∧ +1.reserved not in {{}<br>	∧ +1.length ≤ 1<br>	∧ +2.roles in {EXPRESSION}<br>	∧ ^1.roles not in {OPERATOR}<br>	∧ ^2.internal_type not in {VariableDeclaration}<br>⇒ y = ∅<br>Confidence: 0.999. Support: 23923.` |
| 161 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved not in {(, if}<br>	∧ -1.roles not in {EXPRESSION}<br>	∧ -1.length ≤ 1<br>	∧ -2.roles not in {EXPRESSION}<br>	∧ -2.length ≥ 2<br>	∧ -3.internal_type not in {StringLiteral}<br>	∧ -3.label not in {<space>}<br>	∧ +1.reserved not in {{}<br>	∧ +1.length ≤ 1<br>	∧ +2.roles in {EXPRESSION}<br>	∧ ^1.roles not in {OPERATOR}<br>	∧ ^2.internal_type not in {VariableDeclaration}<br>⇒ y = ∅<br>Confidence: 0.954. Support: 570.` |
| 162 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved not in {(, if}<br>	∧ -1.roles not in {EXPRESSION}<br>	∧ -2.roles not in {EXPRESSION}<br>	∧ -2.length ≤ 1<br>	∧ -3.internal_type not in {StringLiteral}<br>	∧ -3.label not in {<space>}<br>	∧ +1.reserved not in {{}<br>	∧ +1.length ≤ 1<br>	∧ +2.roles in {EXPRESSION}<br>	∧ ^1.roles not in {OPERATOR}<br>	∧ ^2.internal_type not in {VariableDeclaration}<br>⇒ y = ∅<br>Confidence: 0.966. Support: 8902.` |
| 163 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved not in {;, {}<br>	∧ +1.reserved = ,<br>	∧ +2.roles not in {EXPRESSION}<br>⇒ y = ∅<br>Confidence: 0.987. Support: 725.` |
| 164 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.label not in {<newline>, <space>}<br>	∧ -1.reserved not in {(, ., ;, {, }}<br>	∧ -1.roles not in {IDENTIFIER, KEY}<br>	∧ -3.reserved = (<br>	∧ -5.diff_offset ≥ 7<br>	∧ +1.reserved not in {(, ,, ;, }}<br>	∧ +1.roles not in {COMMENT, KEY}<br>	∧ +1.length ≥ 2<br>	∧ +2.roles not in {EXPRESSION}<br>	∧ +2.length ≤ 9<br>	∧ ^1.internal_type not in {ArrayExpression, VariableDeclaration}<br>⇒ y = ␣<br>Confidence: 0.930. Support: 539.` |
| 165 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved not in {(, ;, {, }}<br>	∧ -1.roles not in {IDENTIFIER, KEY}<br>	∧ -3.reserved = (<br>	∧ -5.diff_offset ≤ 6<br>	∧ +1.reserved not in {,, }}<br>	∧ +1.roles not in {KEY}<br>	∧ +1.length ≥ 2<br>	∧ +2.roles not in {EXPRESSION}<br>	∧ +2.length ≤ 9<br>	∧ ^1.internal_type not in {ArrayExpression}<br>⇒ y = ∅<br>Confidence: 0.970. Support: 218.` |
| 166 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved not in {(, ., ;, {, }}<br>	∧ -1.roles not in {IDENTIFIER, KEY}<br>	∧ -3.reserved not in {(}<br>	∧ +1.reserved not in {(, ,, ;, }}<br>	∧ +1.roles not in {KEY}<br>	∧ +1.length ≥ 2<br>	∧ +2.roles not in {EXPRESSION}<br>	∧ +2.length ≤ 9<br>	∧ ^1.internal_type not in {ArrayExpression, VariableDeclaration}<br>⇒ y = ␣<br>Confidence: 0.940. Support: 13949.` |
| 167 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved not in {(, ., ;, {}<br>	∧ -1.roles not in {KEY}<br>	∧ -2.length ≤ 5<br>	∧ -4.label in {<newline>}<br>	∧ +1.reserved not in {(, ,, ;, }}<br>	∧ +1.roles not in {KEY}<br>	∧ +1.length ≤ 1<br>	∧ +2.roles not in {EXPRESSION}<br>	∧ +2.length ≤ 9<br>	∧ ^1.internal_type not in {ArrayExpression, VariableDeclaration}<br>⇒ y = ␣<br>Confidence: 0.951. Support: 463.` |
| 168 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved not in {(, ., ;, {}<br>	∧ -1.roles not in {KEY}<br>	∧ -4.label not in {<newline>}<br>	∧ +1.reserved not in {(, ,, ;, }}<br>	∧ +1.roles not in {KEY}<br>	∧ +1.length ≤ 1<br>	∧ +2.roles not in {EXPRESSION}<br>	∧ +2.length ≤ 9<br>	∧ ^1.internal_type not in {ArrayExpression, VariableDeclaration}<br>⇒ y = ␣<br>Confidence: 0.985. Support: 15651.` |
| 169 | `  -1.label not in {<space>}<br>	∧ -1.reserved not in {(, ,, if}<br>	∧ -1.roles in {EXPRESSION} and not in {STRING}<br>	∧ -3.label not in {<newline>}<br>	∧ +1.reserved not in {=, {, }}<br>	∧ +1.length ≤ 1<br>	∧ +2.roles in {EXPRESSION}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.999. Support: 23960.` |
| 170 | `  -1.label not in {<space>}<br>	∧ -1.reserved not in {(, ,, if}<br>	∧ -1.roles not in {EXPRESSION, STRING}<br>	∧ -1.length ≤ 2<br>	∧ -2.length ≥ 2<br>	∧ -3.label not in {<newline>}<br>	∧ +1.reserved not in {=, {, }}<br>	∧ +1.length ≤ 1<br>	∧ +2.roles in {EXPRESSION}<br>	∧ ^1.roles not in {CALLEE, OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.976. Support: 483.` |
| 171 | `  -1.label not in {<space>}<br>	∧ -1.reserved not in {(, ,, if}<br>	∧ -1.roles not in {EXPRESSION, STRING}<br>	∧ -2.length ≤ 1<br>	∧ -3.label not in {<newline>}<br>	∧ +1.reserved not in {=, {, }}<br>	∧ +1.length ≤ 1<br>	∧ +2.roles in {EXPRESSION}<br>	∧ ^1.roles not in {CALLEE, OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.985. Support: 7290.` |
| 172 | `  -1.diff_col ≥ 13<br>	∧ -1.internal_type not in {CommentLine}<br>	∧ -1.label in {<newline>}<br>	∧ -1.reserved not in {., ;, {}<br>	∧ +1.reserved not in {(, ;}<br>	∧ +2.roles not in {EXPRESSION}<br>⇒ y = "<br>Confidence: 0.999. Support: 937.` |
| 173 | `  -1.diff_col ≤ 12<br>	∧ -1.label not in {<newline>, <space>}<br>	∧ -1.reserved not in {(, ;, {}<br>	∧ -1.roles not in {IDENTIFIER, STRING}<br>	∧ -2.label not in {<-tab>}<br>	∧ -3.reserved = (<br>	∧ -4.diff_offset ≤ 5<br>	∧ +1.reserved not in {,, }}<br>	∧ +1.roles not in {KEY}<br>	∧ +1.length ≥ 2<br>	∧ +2.roles not in {EXPRESSION}<br>	∧ +2.length ≤ 9<br>	∧ ^1.internal_type not in {ArrayExpression}<br>⇒ y = ∅<br>Confidence: 0.982. Support: 191.` |
| 174 | `  -1.diff_col ≤ 12<br>	∧ -1.label not in {<newline>, <space>}<br>	∧ -1.reserved not in {(, ., ;, {}<br>	∧ -1.roles not in {IDENTIFIER, KEY, STRING}<br>	∧ -2.label not in {<-tab>}<br>	∧ -3.reserved not in {(}<br>	∧ +1.reserved not in {(, ,, ;, }}<br>	∧ +1.roles not in {KEY}<br>	∧ +1.length ≥ 2<br>	∧ +2.roles not in {EXPRESSION}<br>	∧ +2.length ≤ 9<br>	∧ ^1.internal_type not in {ArrayExpression, VariableDeclaration}<br>⇒ y = ␣<br>Confidence: 0.941. Support: 13854.` |
| 175 | `  -1.diff_col ≤ 12<br>	∧ -1.label not in {<newline>, <space>}<br>	∧ -1.reserved not in {(, ., ;, {}<br>	∧ -1.roles not in {KEY, STRING}<br>	∧ -2.length ≤ 5<br>	∧ -4.label in {<newline>}<br>	∧ +1.reserved not in {(, ,, ;, }}<br>	∧ +1.roles not in {KEY}<br>	∧ +1.length ≤ 1<br>	∧ +2.roles not in {EXPRESSION}<br>	∧ +2.length ≤ 9<br>	∧ ^1.internal_type not in {ArrayExpression, VariableDeclaration}<br>⇒ y = ␣<br>Confidence: 0.948. Support: 451.` |

### Note
All statistics are calculated with respect to the "analyze" config. This means that the rules were filtered by
`confidence_threshold` and `support_threshold` values.

<details>
    <summary>Machine-readable report</summary>
```json
{"javascript": {"avg_rule_len": 12.16, "max_conf": 0.9999549984931946, "max_support": 27635, "min_conf": 0.9210526347160339, "min_support": 133, "num_rules": 175}}
```
</details>
