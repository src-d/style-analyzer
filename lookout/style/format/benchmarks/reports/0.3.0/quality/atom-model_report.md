# Model report for file:///tmp/top-repos-quality-repos-agu98snj/atom HEAD 108b23210759a8c5b2f51ac99659be5dc31a7371

### Dump

```json
{'created_at': '2019-06-11 18:10:37',
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
 'size': '17.4 kB',
 'tags': [],
 'uuid': '947a4127-bf46-42f3-9d3a-fcf3bc9924c5',
 'vendor': 'source{d}',
 'version': [1]}
style.format.analyzer.FormatAnalyzer/[1] file:///tmp/top-repos-quality-repos-agu98snj/atom 108b23210759a8c5b2f51ac99659be5dc31a7371

# javascript
95 rules, avg.len. 10.2
## train
PPCR: 0.982063
### report
macro
{'f1-score': 0.7280215079522835,
 'precision': 0.7512374338035269,
 'recall': 0.7111357908267557,
 'support': 525400}
micro
{'f1-score': 0.9750932622763608,
 'precision': 0.9750932622763608,
 'recall': 0.9750932622763608,
 'support': 525400}
weighted
{'f1-score': 0.9735374069447599,
 'precision': 0.9727681771008666,
 'recall': 0.9750932622763608,
 'support': 525400}
### report_full
macro
{'f1-score': 0.6996093826506375,
 'precision': 0.7512374338035269,
 'recall': 0.6689200785429944,
 'support': 534996}
micro
{'f1-score': 0.9662692050894194,
 'precision': 0.9750932622763608,
 'recall': 0.9576034213339912,
 'support': 534996}
weighted
{'f1-score': 0.963463056542123,
 'precision': 0.9725777155517067,
 'recall': 0.9576034213339912,
 'support': 534996}
## test
PPCR: 0.975242
### report
macro
{'f1-score': 0.7196880492212835,
 'precision': 0.7303377471137491,
 'recall': 0.7126005090356092,
 'support': 88392}
micro
{'f1-score': 0.9607091139469636,
 'precision': 0.9607091139469636,
 'recall': 0.9607091139469636,
 'support': 88392}
weighted
{'f1-score': 0.9591403468992479,
 'precision': 0.9586557590501055,
 'recall': 0.9607091139469636,
 'support': 88392}
### report_full
macro
{'f1-score': 0.6920799659343717,
 'precision': 0.7303377471137491,
 'recall': 0.6667249616788349,
 'support': 90636}
micro
{'f1-score': 0.9486672475813839,
 'precision': 0.9607091139469636,
 'recall': 0.9369235182488195,
 'support': 90636}
weighted
{'f1-score': 0.9457129845624301,
 'precision': 0.9580914277313951,
 'recall': 0.9369235182488195,
 'support': 90636}
```

## javascript
### Summary
62 rules, avg.len. 9.2

| | |
|-|-|
|Min support|91|
|Max support|37426|
|Min confidence|0.9211438298225403|
|Max confidence|0.9998539686203003|

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
| 1 | `  -1.reserved = ,<br>	∧ ^1.internal_type = ArrayExpression<br>	∧ ^1.roles in {BINARY}<br>⇒ y = ⏎<br>Confidence: 1.000. Support: 37426.` |
| 2 | `  •••start_col ≥ 70<br>	∧ -1.reserved = ,<br>	∧ -2.reserved not in {}}<br>	∧ ^1.internal_type = ArrayExpression<br>	∧ ^1.roles not in {BINARY}<br>	∧ ^2.roles not in {INSTANCE}<br>⇒ y = ␣<br>Confidence: 0.976. Support: 530.` |
| 3 | `  •••start_col ≤ 69<br>	∧ -1.reserved = ,<br>	∧ -2.reserved not in {}}<br>	∧ -4.roles not in {LITERAL}<br>	∧ ^1.internal_type = ArrayExpression<br>	∧ ^1.roles not in {BINARY}<br>⇒ y = ␣<br>Confidence: 0.949. Support: 6169.` |
| 4 | `  -1.reserved not in {,}<br>	∧ ^1.internal_type = ArrayExpression<br>	∧ ^2.roles in {ASSIGNMENT}<br>⇒ y = '<br>Confidence: 1.000. Support: 18529.` |
| 5 | `  -1.internal_type = StringLiteral<br>	∧ -1.reserved not in {,}<br>	∧ ^1.internal_type = ArrayExpression<br>	∧ ^2.roles not in {ASSIGNMENT}<br>⇒ y = '<br>Confidence: 0.999. Support: 797.` |
| 6 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,}<br>	∧ +1.internal_type = StringLiteral<br>	∧ ^1.internal_type = ArrayExpression<br>	∧ ^2.roles not in {ASSIGNMENT}<br>⇒ y = '<br>Confidence: 0.995. Support: 468.` |
| 7 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,}<br>	∧ -5.label in {<-space>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ ^1.internal_type = ArrayExpression<br>	∧ ^2.roles not in {ASSIGNMENT}<br>⇒ y = ⏎␣⁺␣⁺<br>Confidence: 0.968. Support: 109.` |
| 8 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,}<br>	∧ -2.internal_type not in {StringLiteral}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ ^1.internal_type = ArrayExpression<br>	∧ ^2.roles not in {ASSIGNMENT}<br>⇒ y = ∅<br>Confidence: 0.967. Support: 5387.` |
| 9 | `  -1.internal_type = StringLiteral<br>	∧ ^1.internal_type not in {ArrayExpression}<br>	∧ ^1.roles in {QUALIFIED}<br>⇒ y = '<br>Confidence: 0.991. Support: 793.` |
| 10 | `  -1.internal_type not in {StringLiteral}<br>	∧ ^1.internal_type not in {ArrayExpression}<br>	∧ ^1.roles in {QUALIFIED}<br>⇒ y = ∅<br>Confidence: 0.997. Support: 6467.` |
| 11 | `  -1.internal_type = StringLiteral<br>	∧ ^1.internal_type not in {ArrayExpression}<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = '<br>Confidence: 0.956. Support: 10185.` |
| 12 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = (<br>	∧ +1.internal_type = StringLiteral<br>	∧ +1.length ≥ 2<br>	∧ ^1.internal_type not in {ArrayExpression}<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = '<br>Confidence: 0.949. Support: 6435.` |
| 13 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = (<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.length ≥ 2<br>	∧ ^1.internal_type not in {ArrayExpression}<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ∅<br>Confidence: 0.992. Support: 14340.` |
| 14 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = {<br>	∧ +1.roles not in {MAP}<br>	∧ +1.length ≥ 2<br>	∧ ^1.internal_type not in {ArrayExpression}<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ⏎␣⁺␣⁺<br>Confidence: 0.958. Support: 7242.` |
| 15 | `  •••start_col ≥ 16<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = )<br>	∧ +1.length ≥ 2<br>	∧ ^1.internal_type = IfStatement<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ␣<br>Confidence: 0.993. Support: 476.` |
| 16 | `  •••start_col ≥ 16<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = )<br>	∧ +1.length ≥ 2<br>	∧ ^1.internal_type not in {ArrayExpression, IfStatement}<br>	∧ ^1.roles in {EXPRESSION} and not in {QUALIFIED}<br>⇒ y = ␣<br>Confidence: 0.998. Support: 239.` |
| 17 | `  •••start_col ≤ 15<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = )<br>	∧ +1.length ≥ 2<br>	∧ +3.roles in {LITERAL}<br>	∧ ^1.internal_type not in {ArrayExpression}<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ⏎⏎<br>Confidence: 1.000. Support: 1341.` |
| 18 | `  •••start_col ≤ 15<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = )<br>	∧ +1.length ≥ 2<br>	∧ +3.reserved = (<br>	∧ +3.roles not in {LITERAL}<br>	∧ ^1.internal_type not in {ArrayExpression}<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ⏎⏎<br>Confidence: 0.944. Support: 116.` |
| 19 | `  -1.internal_type = CommentLine<br>	∧ -1.reserved not in {(, ), {}<br>	∧ +1.length ≥ 2<br>	∧ ^1.internal_type = File<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ⏎<br>Confidence: 0.989. Support: 4075.` |
| 20 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {(, ), {}<br>	∧ +1.roles in {COMMENT}<br>	∧ +1.length ≥ 2<br>	∧ ^1.internal_type = File<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ∅<br>Confidence: 1.000. Support: 3424.` |
| 21 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.label in {<space>}<br>	∧ -1.reserved not in {(, ), {}<br>	∧ +1.length ≥ 2<br>	∧ ^1.internal_type not in {ArrayExpression, File}<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = '<br>Confidence: 0.984. Support: 3483.` |
| 22 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved = }<br>	∧ +1.length ≥ 2<br>	∧ ^1.internal_type not in {ArrayExpression, File}<br>	∧ ^1.roles not in {QUALIFIED}<br>	∧ ^2.roles in {SCOPE}<br>⇒ y = ␣<br>Confidence: 0.988. Support: 465.` |
| 23 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved = }<br>	∧ +1.length ≥ 2<br>	∧ ^1.internal_type not in {ArrayExpression, File}<br>	∧ ^1.roles in {EXPRESSION} and not in {QUALIFIED}<br>	∧ ^2.roles not in {SCOPE}<br>⇒ y = ∅<br>Confidence: 0.998. Support: 244.` |
| 24 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved = }<br>	∧ +1.length ≥ 2<br>	∧ ^1.internal_type not in {ArrayExpression, File}<br>	∧ ^1.roles not in {EXPRESSION, QUALIFIED, STATEMENT}<br>	∧ ^2.roles not in {SCOPE}<br>⇒ y = ⏎⏎<br>Confidence: 0.977. Support: 1121.` |
| 25 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved not in {(, ), {, }}<br>	∧ +1.roles not in {MAP}<br>	∧ +1.length ≥ 2<br>	∧ ^1.internal_type not in {ArrayExpression, File}<br>	∧ ^1.roles in {LITERAL} and not in {QUALIFIED}<br>⇒ y = ∅<br>Confidence: 0.976. Support: 864.` |
| 26 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved = [<br>	∧ +1.roles not in {STRING}<br>	∧ +1.length ≥ 2<br>	∧ ^1.internal_type not in {File}<br>	∧ ^1.roles not in {LITERAL, QUALIFIED}<br>⇒ y = ∅<br>Confidence: 0.952. Support: 486.` |
| 27 | `  •••start_col ≤ 19<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved not in {(, ), [, {}<br>	∧ -1.length ≥ 3<br>	∧ +1.length ≥ 2<br>	∧ ^1.internal_type = BlockStatement<br>	∧ ^1.roles not in {LITERAL, QUALIFIED}<br>⇒ y = ␣<br>Confidence: 0.978. Support: 2821.` |
| 28 | `  •••start_col ≤ 19<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved not in {(, ), [, {}<br>	∧ -1.length ≤ 2<br>	∧ -2.reserved = (<br>	∧ +1.length ≥ 2<br>	∧ ^1.internal_type = BlockStatement<br>	∧ ^1.roles not in {LITERAL}<br>⇒ y = ∅<br>Confidence: 0.996. Support: 135.` |
| 29 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved not in {(, ), [, {, }}<br>	∧ -1.length ≥ 2<br>	∧ -2.diff_line = 0<br>	∧ -2.reserved = (<br>	∧ -4.diff_offset ≥ 6<br>	∧ +1.length ≥ 2<br>	∧ ^1.internal_type not in {ArrayExpression, BlockStatement, File}<br>	∧ ^1.roles not in {LITERAL, QUALIFIED}<br>⇒ y = ␣<br>Confidence: 0.930. Support: 717.` |
| 30 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved not in {(, ), [, {, }}<br>	∧ -1.length ≤ 1<br>	∧ -2.diff_line = 0<br>	∧ -2.reserved = (<br>	∧ -4.diff_offset ≥ 6<br>	∧ +1.length ≥ 2<br>	∧ ^1.internal_type not in {ArrayExpression, BlockStatement, File}<br>	∧ ^1.roles not in {LITERAL}<br>⇒ y = ∅<br>Confidence: 0.998. Support: 204.` |
| 31 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {', <space>}<br>	∧ -1.reserved not in {(, ), {, }}<br>	∧ -2.diff_line = 0<br>	∧ -2.reserved not in {(}<br>	∧ -4.diff_offset ≥ 6<br>	∧ +1.roles in {EXPRESSION}<br>	∧ +1.length ≥ 2<br>	∧ ^1.internal_type not in {ArrayExpression, BlockStatement, File}<br>	∧ ^1.roles not in {LITERAL, QUALIFIED}<br>⇒ y = ␣<br>Confidence: 0.976. Support: 18592.` |
| 32 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {', <space>}<br>	∧ -1.reserved not in {(, ), {, }}<br>	∧ -2.diff_line = 0<br>	∧ -2.reserved not in {(}<br>	∧ -4.diff_offset ≥ 6<br>	∧ +1.roles not in {EXPRESSION, IDENTIFIER}<br>	∧ +1.length ≥ 4<br>	∧ ^1.internal_type not in {ArrayExpression, BlockStatement, File}<br>	∧ ^1.roles in {EXPRESSION} and not in {LITERAL, QUALIFIED}<br>⇒ y = ␣<br>Confidence: 0.981. Support: 567.` |
| 33 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {', <space>}<br>	∧ -1.reserved not in {(, ), {, }}<br>	∧ -2.diff_line = 0<br>	∧ -2.reserved not in {(}<br>	∧ -4.diff_offset ≥ 6<br>	∧ +1.roles not in {EXPRESSION, IDENTIFIER}<br>	∧ +1.length ≥ 4<br>	∧ ^1.internal_type not in {ArrayExpression, BlockStatement, File}<br>	∧ ^1.roles in {DECLARATION} and not in {EXPRESSION, LITERAL, QUALIFIED}<br>⇒ y = ␣<br>Confidence: 0.982. Support: 137.` |
| 34 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {', <space>}<br>	∧ -1.reserved not in {(, ), {, }}<br>	∧ -2.diff_line = 0<br>	∧ -2.reserved not in {(}<br>	∧ -4.diff_offset ≥ 6<br>	∧ +1.roles not in {EXPRESSION, IDENTIFIER}<br>	∧ +1.length ≤ 3<br>	∧ ^1.internal_type not in {ArrayExpression, BlockStatement, File}<br>	∧ ^1.roles not in {LITERAL, QUALIFIED}<br>⇒ y = ␣<br>Confidence: 0.984. Support: 2147.` |
| 35 | `  •••start_col ≥ 4<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved not in {(, ), [, {, }}<br>	∧ -2.reserved = (<br>	∧ -4.diff_offset ≤ 5<br>	∧ +1.length ≥ 2<br>	∧ ^1.internal_type not in {ArrayExpression, BlockStatement, File}<br>	∧ ^1.roles not in {LITERAL}<br>⇒ y = ∅<br>Confidence: 0.960. Support: 136.` |
| 36 | `  -1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = =<br>	∧ +1.length ≤ 1<br>	∧ ^1.internal_type not in {ArrayExpression}<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ␣<br>Confidence: 0.995. Support: 10420.` |
| 37 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = (<br>	∧ +1.reserved = {<br>	∧ +1.length ≤ 1<br>	∧ ^1.internal_type not in {ArrayExpression}<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ∅<br>Confidence: 0.993. Support: 784.` |
| 38 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {(, [}<br>	∧ +1.reserved = {<br>	∧ +1.length ≤ 1<br>	∧ ^1.internal_type not in {ArrayExpression}<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ␣<br>Confidence: 0.993. Support: 8663.` |
| 39 | `  •••start_col ≤ 35<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.roles in {EXPRESSION}<br>	∧ +1.reserved = }<br>	∧ +1.length ≤ 1<br>	∧ ^1.internal_type not in {ArrayExpression}<br>	∧ ^1.roles in {DECLARATION} and not in {QUALIFIED}<br>⇒ y = ∅<br>Confidence: 0.936. Support: 164.` |
| 40 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.roles not in {EXPRESSION}<br>	∧ +1.reserved = }<br>	∧ +1.length ≤ 1<br>	∧ ^1.internal_type not in {ArrayExpression}<br>	∧ ^1.roles not in {EXPRESSION, QUALIFIED}<br>⇒ y = ⏎␣⁻␣⁻<br>Confidence: 0.943. Support: 6729.` |
| 41 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = ,<br>	∧ +1.reserved not in {{, }}<br>	∧ +1.length ≤ 1<br>	∧ ^1.internal_type not in {ArrayExpression}<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ␣<br>Confidence: 0.971. Support: 2714.` |
| 42 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = if<br>	∧ +1.reserved not in {{, }}<br>	∧ +1.length ≤ 1<br>	∧ ^1.internal_type not in {ArrayExpression}<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ␣<br>Confidence: 1.000. Support: 1869.` |
| 43 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, if}<br>	∧ -1.roles in {NAME}<br>	∧ +1.reserved not in {}}<br>	∧ +1.length ≤ 1<br>	∧ ^1.internal_type not in {ArrayExpression}<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ␣<br>Confidence: 0.985. Support: 1807.` |
| 44 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, if}<br>	∧ +1.reserved = )<br>	∧ +1.length ≤ 1<br>	∧ ^1.roles in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.989. Support: 318.` |
| 45 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, if}<br>	∧ +1.reserved = ]<br>	∧ +1.length ≤ 1<br>	∧ ^1.roles in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.996. Support: 141.` |
| 46 | `  -1.internal_type = Identifier<br>	∧ -1.reserved not in {,, if}<br>	∧ +1.reserved = (<br>	∧ +1.roles not in {STRING}<br>	∧ +1.length ≤ 1<br>	∧ ^1.roles in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.995. Support: 93.` |
| 47 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, if}<br>	∧ +1.reserved not in {(, ), }}<br>	∧ +1.roles not in {STRING}<br>	∧ +1.length ≤ 1<br>	∧ ^1.internal_type not in {ArrayExpression}<br>	∧ ^1.roles in {OPERATOR} and not in {QUALIFIED}<br>⇒ y = ␣<br>Confidence: 0.934. Support: 1391.` |
| 48 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = :<br>	∧ +1.reserved not in {}}<br>	∧ +1.length ≤ 1<br>	∧ ^1.internal_type not in {ArrayExpression}<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ␣<br>Confidence: 0.999. Support: 956.` |
| 49 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,}<br>	∧ +1.internal_type = StringLiteral<br>	∧ +1.reserved not in {=, {, }}<br>	∧ +1.length ≤ 1<br>	∧ ^1.internal_type not in {ArrayExpression}<br>	∧ ^1.roles not in {OPERATOR, QUALIFIED}<br>⇒ y = '<br>Confidence: 0.921. Support: 577.` |
| 50 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = function<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {}}<br>	∧ +1.length ≤ 1<br>	∧ ^1.internal_type not in {ArrayExpression}<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ␣<br>Confidence: 0.989. Support: 227.` |
| 51 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, if}<br>	∧ -3.roles in {EXPRESSION}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {}}<br>	∧ +1.length ≤ 1<br>	∧ ^1.internal_type not in {ArrayExpression}<br>	∧ ^1.roles not in {QUALIFIED}<br>	∧ ^2.internal_type = VariableDeclaration<br>⇒ y = ␣<br>Confidence: 0.983. Support: 268.` |
| 52 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, :, if}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {=, {}<br>	∧ +1.length ≤ 1<br>	∧ ^1.internal_type = CallExpression<br>	∧ ^1.roles in {CONDITION} and not in {OPERATOR}<br>	∧ ^2.internal_type not in {VariableDeclaration}<br>⇒ y = ∅<br>Confidence: 0.997. Support: 186.` |
| 53 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = const<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {}}<br>	∧ +1.length ≤ 1<br>	∧ ^1.internal_type not in {ArrayExpression}<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ␣<br>Confidence: 0.996. Support: 114.` |
| 54 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = return<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {}}<br>	∧ +1.length ≤ 1<br>	∧ ^1.internal_type not in {ArrayExpression}<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ␣<br>Confidence: 0.995. Support: 91.` |
| 55 | `  -1.reserved = (<br>	∧ +1.internal_type = Identifier<br>	∧ +1.reserved not in {=, {, }}<br>	∧ +1.length ≤ 1<br>	∧ ^1.internal_type not in {ArrayExpression}<br>	∧ ^1.roles not in {OPERATOR}<br>	∧ ^2.internal_type not in {VariableDeclaration}<br>⇒ y = ∅<br>Confidence: 0.991. Support: 169.` |
| 56 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, return}<br>	∧ -5.length ≥ 25<br>	∧ +1.internal_type not in {Identifier, StringLiteral}<br>	∧ +1.reserved not in {}}<br>	∧ +1.length ≤ 1<br>	∧ +4.reserved = ><br>	∧ ^1.internal_type not in {ArrayExpression}<br>	∧ ^1.roles not in {QUALIFIED}<br>⇒ y = ␣<br>Confidence: 0.979. Support: 216.` |
| 57 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, function, return}<br>	∧ +1.internal_type not in {Identifier, StringLiteral}<br>	∧ +1.reserved not in {=, {, }}<br>	∧ +1.length ≤ 1<br>	∧ +4.reserved = ><br>	∧ ^1.internal_type not in {ArrayExpression}<br>	∧ ^1.roles not in {OPERATOR}<br>	∧ ^2.internal_type not in {VariableDeclaration}<br>⇒ y = ∅<br>Confidence: 0.946. Support: 1130.` |
| 58 | `  -1.diff_col ≤ 2<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, function, return}<br>	∧ -3.reserved = }<br>	∧ +1.internal_type not in {Identifier, StringLiteral}<br>	∧ +1.reserved not in {=, {, }}<br>	∧ +1.length ≤ 1<br>	∧ ^1.internal_type not in {ArrayExpression}<br>	∧ ^1.roles not in {OPERATOR}<br>	∧ ^2.internal_type not in {VariableDeclaration}<br>⇒ y = ∅<br>Confidence: 0.939. Support: 468.` |
| 59 | `  •••start_col ≤ 12<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, function, return}<br>	∧ +1.internal_type not in {Identifier, StringLiteral}<br>	∧ +1.reserved not in {=, {, }}<br>	∧ +1.length ≤ 1<br>	∧ +3.internal_type = Identifier<br>	∧ ^1.internal_type not in {ArrayExpression}<br>	∧ ^1.roles not in {OPERATOR}<br>	∧ ^2.internal_type not in {VariableDeclaration}<br>⇒ y = ∅<br>Confidence: 0.968. Support: 264.` |
| 60 | `  •••start_col ≤ 12<br>	∧ -1.diff_col ≥ 3<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, :, function, return}<br>	∧ -1.roles in {EXPRESSION}<br>	∧ +1.internal_type not in {Identifier, StringLiteral}<br>	∧ +1.reserved not in {=, {, }}<br>	∧ +1.length ≤ 1<br>	∧ +3.internal_type not in {Identifier}<br>	∧ ^1.internal_type not in {ArrayExpression}<br>	∧ ^1.roles not in {OPERATOR}<br>	∧ ^2.internal_type not in {VariableDeclaration}<br>⇒ y = ∅<br>Confidence: 0.998. Support: 835.` |
| 61 | `  •••start_col ≤ 12<br>	∧ -1.diff_col ≤ 2<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, :, function, return}<br>	∧ +1.internal_type not in {Identifier, StringLiteral}<br>	∧ +1.reserved not in {=, {, }}<br>	∧ +1.length ≤ 1<br>	∧ +3.internal_type not in {Identifier}<br>	∧ ^1.internal_type not in {ArrayExpression}<br>	∧ ^1.roles not in {OPERATOR}<br>	∧ ^2.internal_type not in {VariableDeclaration}<br>⇒ y = ∅<br>Confidence: 0.990. Support: 4573.` |
| 62 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, if}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {=, {, }}<br>	∧ +1.length = 0<br>	∧ ^1.internal_type not in {ArrayExpression}<br>	∧ ^1.roles not in {OPERATOR, QUALIFIED}<br>⇒ y = ⏎<br>Confidence: 0.987. Support: 191.` |

### Note
All statistics are calculated with respect to the "analyze" config. This means that the rules were filtered by
`confidence_threshold` and `support_threshold` values.

<details>
    <summary>Machine-readable report</summary>
```json
{"javascript": {"avg_rule_len": 9.161290322580646, "max_conf": 0.9998539686203003, "max_support": 37426, "min_conf": 0.9211438298225403, "min_support": 91, "num_rules": 62}}
```
</details>
