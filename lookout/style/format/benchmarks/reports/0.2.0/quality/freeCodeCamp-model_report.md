# Model report for file:///tmp/top-repos-quality-repos-oj3n1zqw/freeCodeCamp HEAD cf65516cce60645a417e44c4fcea7418ca920572

### Dump

```json
{'created_at': '2019-04-08 13:53:01',
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
 'size': '22.6 kB',
 'tags': [],
 'uuid': '8adafe76-7d6c-437d-8be5-b5a45db24935',
 'vendor': 'source{d}',
 'version': [1]}
style.format.analyzer.FormatAnalyzer/[1] file:///tmp/top-repos-quality-repos-oj3n1zqw/freeCodeCamp cf65516cce60645a417e44c4fcea7418ca920572

# javascript
236 rules, avg.len. 12.5
## train
PPCR: 0.960984
### report
macro
{'f1-score': 0.6573591426849205,
 'precision': 0.6980179600356755,
 'recall': 0.6282940482922295,
 'support': 100517}
micro
{'f1-score': 0.9277535143309092,
 'precision': 0.9277535143309092,
 'recall': 0.9277535143309092,
 'support': 100517}
weighted
{'f1-score': 0.9234805536341264,
 'precision': 0.922230399969419,
 'recall': 0.9277535143309092,
 'support': 100517}
### report_full
macro
{'f1-score': 0.61762639565591,
 'precision': 0.6980179600356755,
 'recall': 0.5708637828432617,
 'support': 104598}
micro
{'f1-score': 0.9092947858518393,
 'precision': 0.9277535143309092,
 'recall': 0.8915562439052372,
 'support': 104598}
weighted
{'f1-score': 0.901693745779249,
 'precision': 0.9207565091651311,
 'recall': 0.8915562439052372,
 'support': 104598}
## test
PPCR: 0.962221
### report
macro
{'f1-score': 0.6546347357558234,
 'precision': 0.6934743669637086,
 'recall': 0.6267949765672955,
 'support': 23738}
micro
{'f1-score': 0.9289746398180133,
 'precision': 0.9289746398180133,
 'recall': 0.9289746398180133,
 'support': 23738}
weighted
{'f1-score': 0.9258329518913425,
 'precision': 0.9254069525770654,
 'recall': 0.9289746398180133,
 'support': 23738}
### report_full
macro
{'f1-score': 0.6178858955570286,
 'precision': 0.6934743669637086,
 'recall': 0.5732424924090523,
 'support': 24670}
micro
{'f1-score': 0.9110890761857544,
 'precision': 0.9289746398180133,
 'recall': 0.8938792055127686,
 'support': 24670}
weighted
{'f1-score': 0.9044685128065404,
 'precision': 0.9237712947403649,
 'recall': 0.8938792055127686,
 'support': 24670}
```

## javascript
### Summary
156 rules, avg.len. 12.5

| | |
|-|-|
|Min support|134|
|Max support|29403|
|Min confidence|0.9216216206550598|
|Max confidence|0.999721109867096|

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
| 1 | `  -1.internal_type = StringLiteral<br>⇒ y = '<br>Confidence: 0.967. Support: 3685.` |
| 2 | `  -1.internal_type not in {StringLiteral}<br>	∧ -2.label in {<space>}<br>	∧ -3.diff_offset ≥ 4<br>	∧ +1.internal_type = StringLiteral<br>⇒ y = ␣<br>Confidence: 0.976. Support: 1152.` |
| 3 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = :<br>	∧ -2.label not in {<space>}<br>	∧ +1.internal_type = StringLiteral<br>⇒ y = ␣<br>Confidence: 0.963. Support: 550.` |
| 4 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = ,<br>	∧ -2.label not in {<space>}<br>	∧ -5.label in {<newline>}<br>	∧ +1.internal_type = StringLiteral<br>⇒ y = ⏎<br>Confidence: 0.979. Support: 215.` |
| 5 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, :}<br>	∧ -2.label not in {<space>}<br>	∧ +1.internal_type = StringLiteral<br>⇒ y = '<br>Confidence: 0.945. Support: 3845.` |
| 6 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = {<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.roles not in {VALUE}<br>	∧ ^1.roles in {DECLARATION}<br>⇒ y = ⏎␣⁺␣⁺<br>Confidence: 0.923. Support: 1041.` |
| 7 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {{}<br>	∧ -2.label in {<space>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = }<br>	∧ ^1.roles in {DECLARATION, FUNCTION}<br>⇒ y = ␣<br>Confidence: 0.960. Support: 139.` |
| 8 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {{}<br>	∧ -1.roles not in {ARGUMENT}<br>	∧ -2.label in {<space>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {}}<br>	∧ ^1.roles in {DECLARATION, FUNCTION}<br>⇒ y = ∅<br>Confidence: 0.949. Support: 1335.` |
| 9 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = (<br>	∧ -2.label not in {<space>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ ^1.roles in {DECLARATION, FUNCTION}<br>⇒ y = ∅<br>Confidence: 0.990. Support: 435.` |
| 10 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {(, {}<br>	∧ -1.roles in {IDENTIFIER}<br>	∧ -2.label not in {<space>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = =<br>	∧ ^1.roles in {DECLARATION, FUNCTION}<br>⇒ y = ␣<br>Confidence: 0.997. Support: 170.` |
| 11 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {(, {}<br>	∧ -1.roles in {IDENTIFIER}<br>	∧ -2.label not in {<space>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {=}<br>	∧ ^1.roles in {DECLARATION, FUNCTION}<br>⇒ y = ∅<br>Confidence: 0.977. Support: 599.` |
| 12 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {{}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = ,<br>	∧ ^1.roles in {DECLARATION} and not in {FUNCTION}<br>⇒ y = ∅<br>Confidence: 0.998. Support: 299.` |
| 13 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {{}<br>	∧ -4.diff_line = 0<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {,}<br>	∧ ^1.roles in {DECLARATION} and not in {FUNCTION}<br>⇒ y = ␣<br>Confidence: 0.944. Support: 4267.` |
| 14 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = ;<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = }<br>	∧ ^1.roles not in {DECLARATION}<br>⇒ y = ⏎␣⁻␣⁻<br>Confidence: 0.940. Support: 1184.` |
| 15 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = ,<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ ^1.internal_type = ObjectExpression<br>	∧ ^1.roles not in {DECLARATION}<br>⇒ y = ⏎<br>Confidence: 0.930. Support: 1488.` |
| 16 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = ,<br>	∧ -2.reserved not in {)}<br>	∧ -5.diff_line = 0<br>	∧ -5.reserved = (<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +4.reserved not in {,}<br>	∧ ^1.internal_type not in {ObjectExpression}<br>	∧ ^1.roles not in {DECLARATION}<br>⇒ y = ␣<br>Confidence: 0.998. Support: 242.` |
| 17 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = :<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ ^1.roles not in {DECLARATION, OPERATOR}<br>⇒ y = ␣<br>Confidence: 0.981. Support: 1671.` |
| 18 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, :, ;}<br>	∧ -3.reserved = ;<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ ^1.roles in {EXPRESSION} and not in {DECLARATION, OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.978. Support: 691.` |
| 19 | `  -1.diff_col ≥ 2<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, :, ;}<br>	∧ -3.reserved = ;<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.length ≤ 18<br>	∧ ^1.roles not in {DECLARATION, EXPRESSION}<br>⇒ y = ␣<br>Confidence: 0.992. Support: 1982.` |
| 20 | `  -1.diff_col ≤ 1<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, :, ;}<br>	∧ -3.reserved = ;<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.length ≥ 2<br>	∧ ^1.roles not in {DECLARATION, EXPRESSION, OPERATOR, STATEMENT}<br>⇒ y = ⏎⏎<br>Confidence: 0.930. Support: 277.` |
| 21 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = {<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ ^1.roles in {INCOMPLETE} and not in {DECLARATION, OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.945. Support: 719.` |
| 22 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = {<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +2.reserved = }<br>	∧ ^1.roles not in {DECLARATION, INCOMPLETE}<br>⇒ y = ␣<br>Confidence: 0.966. Support: 426.` |
| 23 | `  -1.internal_type = CommentLine<br>	∧ -1.reserved not in {,, :, ;, {}<br>	∧ -3.reserved not in {;}<br>	∧ ^1.roles not in {DECLARATION, OPERATOR}<br>⇒ y = ⏎<br>Confidence: 0.931. Support: 613.` |
| 24 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {,, :, ;, {}<br>	∧ -3.reserved not in {;}<br>	∧ +1.internal_type not in {JSXIdentifier, StringLiteral}<br>	∧ ^1.internal_type = JSXOpeningElement<br>	∧ ^1.roles not in {DECLARATION, OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.925. Support: 369.` |
| 25 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {,, ;, {}<br>	∧ -3.reserved not in {;}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = {<br>	∧ ^1.internal_type not in {JSXOpeningElement}<br>	∧ ^1.roles in {IF} and not in {DECLARATION}<br>⇒ y = ␣<br>Confidence: 0.985. Support: 443.` |
| 26 | `  •••start_col ≤ 32<br>	∧ -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {,, :, ;, {}<br>	∧ -3.reserved not in {;}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = }<br>	∧ ^1.internal_type not in {JSXOpeningElement}<br>	∧ ^1.roles not in {DECLARATION, IF, INCOMPLETE, OPERATOR}<br>⇒ y = ⏎␣⁻␣⁻<br>Confidence: 0.936. Support: 447.` |
| 27 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {,, {}<br>	∧ -3.reserved = export<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {}}<br>	∧ ^1.internal_type not in {JSXOpeningElement}<br>	∧ ^1.roles not in {DECLARATION, IF}<br>⇒ y = ␣<br>Confidence: 0.999. Support: 384.` |
| 28 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {,, ;, {}<br>	∧ -3.reserved not in {;}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {}}<br>	∧ ^1.internal_type = AssignmentPattern<br>	∧ ^1.roles not in {DECLARATION, IF}<br>⇒ y = ␣<br>Confidence: 0.998. Support: 262.` |
| 29 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved = import<br>	∧ -3.reserved not in {export}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {}}<br>	∧ ^1.roles not in {DECLARATION, IF}<br>⇒ y = ␣<br>Confidence: 0.997. Support: 149.` |
| 30 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {,, :, ;, import, {}<br>	∧ -2.internal_type not in {StringLiteral}<br>	∧ -3.reserved not in {;, export}<br>	∧ -4.label not in {<-space>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {}}<br>	∧ +1.roles in {IDENTIFIER}<br>	∧ +2.reserved = ><br>	∧ ^1.internal_type not in {JSXOpeningElement}<br>	∧ ^1.roles not in {DECLARATION, IF, OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.999. Support: 808.` |
| 31 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {(, ), ,, :, ;, import, {}<br>	∧ -2.internal_type not in {StringLiteral}<br>	∧ -2.label in {<-space>}<br>	∧ -2.length ≥ 2<br>	∧ -3.reserved not in {;, export}<br>	∧ -4.label not in {<-space>}<br>	∧ -5.diff_offset ≥ 6<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = )<br>	∧ +2.reserved not in {>}<br>	∧ ^1.internal_type not in {JSXOpeningElement}<br>	∧ ^1.roles not in {DECLARATION, IF, OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.932. Support: 185.` |
| 32 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {(, ), ,, :, ;, import, {}<br>	∧ -2.internal_type not in {StringLiteral}<br>	∧ -2.length ≤ 1<br>	∧ -3.reserved not in {;, export}<br>	∧ -4.label not in {<-space>}<br>	∧ -5.diff_offset ≥ 6<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = )<br>	∧ +2.reserved not in {>}<br>	∧ ^1.internal_type not in {JSXOpeningElement}<br>	∧ ^1.roles not in {DECLARATION, IF, OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.964. Support: 1211.` |
| 33 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {(, ), ,, :, ;, [, import, {, }}<br>	∧ -2.internal_type not in {StringLiteral}<br>	∧ -3.reserved not in {;}<br>	∧ -4.label not in {<-space>, <newline>}<br>	∧ -5.diff_offset ≥ 6<br>	∧ +1.internal_type not in {Identifier, StringLiteral}<br>	∧ +1.reserved not in {), }}<br>	∧ +2.reserved not in {>}<br>	∧ ^1.internal_type not in {JSXOpeningElement}<br>	∧ ^1.roles not in {DECLARATION, IF, OPERATOR}<br>	∧ ^2.roles in {STATEMENT}<br>⇒ y = ∅<br>Confidence: 0.984. Support: 2595.` |
| 34 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {(, ), ,, :, ;, [, import, {, }}<br>	∧ -2.internal_type not in {StringLiteral}<br>	∧ -3.reserved not in {;}<br>	∧ -4.label not in {<-space>, <newline>}<br>	∧ -5.diff_offset ≥ 6<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {), }}<br>	∧ +2.reserved not in {>}<br>	∧ ^1.internal_type not in {JSXOpeningElement}<br>	∧ ^1.roles not in {DECLARATION, IF, OPERATOR}<br>	∧ ^2.roles not in {STATEMENT}<br>⇒ y = ∅<br>Confidence: 0.992. Support: 20379.` |
| 35 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {{}<br>	∧ -1.roles in {ARGUMENT}<br>	∧ -2.label in {<space>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = )<br>	∧ ^1.roles in {DECLARATION, FUNCTION}<br>⇒ y = ∅<br>Confidence: 0.998. Support: 202.` |
| 36 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {(, {}<br>	∧ -1.roles in {EXPRESSION}<br>	∧ -2.label not in {<space>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +2.reserved = ><br>	∧ ^1.roles in {DECLARATION, FUNCTION}<br>⇒ y = ␣<br>Confidence: 0.997. Support: 192.` |
| 37 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {(, {}<br>	∧ -1.roles in {EXPRESSION}<br>	∧ -2.label not in {<space>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +2.reserved not in {>}<br>	∧ ^1.roles in {DECLARATION, FUNCTION}<br>⇒ y = ∅<br>Confidence: 0.963. Support: 616.` |
| 38 | `  •••start_col ≥ 7<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = ;<br>	∧ -3.roles in {PATHNAME}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {export, }}<br>	∧ +3.length ≤ 2<br>	∧ ^1.internal_type = Program<br>	∧ ^1.roles not in {DECLARATION}<br>⇒ y = ⏎⏎<br>Confidence: 0.952. Support: 156.` |
| 39 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, :, ;}<br>	∧ -1.length ≥ 2<br>	∧ -3.reserved = ;<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.length ≤ 18<br>	∧ ^1.roles not in {DECLARATION, EXPRESSION}<br>⇒ y = ␣<br>Confidence: 0.992. Support: 2046.` |
| 40 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, :, ;}<br>	∧ -1.length ≤ 1<br>	∧ -3.reserved = ;<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.length ≥ 2<br>	∧ ^1.roles not in {DECLARATION, EXPRESSION, OPERATOR, STATEMENT}<br>⇒ y = ⏎⏎<br>Confidence: 0.952. Support: 321.` |
| 41 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {,, :, ;, {}<br>	∧ -3.reserved not in {;}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +2.reserved = =<br>	∧ ^1.roles in {EXPRESSION} and not in {DECLARATION, OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.938. Support: 735.` |
| 42 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {,, ;, {}<br>	∧ -3.reserved not in {;}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = {<br>	∧ +2.reserved not in {=}<br>	∧ ^1.roles in {IF} and not in {DECLARATION}<br>⇒ y = ␣<br>Confidence: 0.973. Support: 386.` |
| 43 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {,, :, ;, {}<br>	∧ -3.reserved not in {;, {}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {{}<br>	∧ +2.reserved not in {=}<br>	∧ ^1.internal_type not in {ConditionalExpression}<br>	∧ ^1.roles in {IF} and not in {DECLARATION, OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.923. Support: 1403.` |
| 44 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {,, ;, {}<br>	∧ -3.reserved not in {;}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {}}<br>	∧ +1.roles in {NAME}<br>	∧ +2.reserved not in {=}<br>	∧ ^1.roles not in {DECLARATION, IF}<br>⇒ y = ␣<br>Confidence: 0.956. Support: 239.` |
| 45 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {,, ;, {}<br>	∧ -3.reserved not in {;}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {}}<br>	∧ +2.reserved not in {=}<br>	∧ ^1.internal_type = AssignmentPattern<br>	∧ ^1.roles not in {DECLARATION, IF}<br>⇒ y = ␣<br>Confidence: 0.993. Support: 220.` |
| 46 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved = import<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {}}<br>	∧ +2.reserved not in {=}<br>	∧ ^1.roles not in {DECLARATION, IF}<br>⇒ y = ␣<br>Confidence: 0.997. Support: 156.` |
| 47 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {,, ;, {}<br>	∧ -2.reserved = =<br>	∧ -3.reserved not in {;}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {}}<br>	∧ +2.reserved not in {=}<br>	∧ ^1.roles not in {DECLARATION, IF}<br>⇒ y = ␣<br>Confidence: 0.942. Support: 147.` |
| 48 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {,, :, ;, import, {}<br>	∧ -1.roles in {EXPRESSION}<br>	∧ -2.reserved not in {=}<br>	∧ -3.reserved not in {;}<br>	∧ -5.reserved = ;<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {}}<br>	∧ +1.roles not in {NAME}<br>	∧ +2.reserved not in {=}<br>	∧ ^1.roles not in {DECLARATION, IF, OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.970. Support: 559.` |
| 49 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {,, ;, {}<br>	∧ -1.roles not in {EXPRESSION}<br>	∧ -1.length ≥ 3<br>	∧ -2.reserved not in {=}<br>	∧ -3.reserved not in {;}<br>	∧ -5.reserved = ;<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {}}<br>	∧ +2.reserved not in {=}<br>	∧ ^1.roles not in {DECLARATION, IF}<br>⇒ y = ␣<br>Confidence: 0.998. Support: 311.` |
| 50 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {,, :, ;, import, {}<br>	∧ -2.internal_type not in {StringLiteral}<br>	∧ -3.reserved not in {;}<br>	∧ -5.reserved not in {;}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {}}<br>	∧ +1.roles in {IDENTIFIER} and not in {NAME}<br>	∧ +2.reserved = ><br>	∧ ^1.roles not in {DECLARATION, IF, OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.999. Support: 770.` |
| 51 | `  •••start_col ≥ 23<br>	∧ -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {,, :, ;, import, {}<br>	∧ -2.internal_type not in {StringLiteral}<br>	∧ -2.reserved not in {=}<br>	∧ -3.reserved not in {;}<br>	∧ -5.reserved not in {;}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = )<br>	∧ +1.roles not in {NAME}<br>	∧ +2.reserved not in {=, >}<br>	∧ ^1.roles not in {DECLARATION, IF, OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.936. Support: 1785.` |
| 52 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {(, ), ,, :, ;, import, {}<br>	∧ -2.internal_type not in {StringLiteral}<br>	∧ -2.reserved not in {=}<br>	∧ -3.reserved not in {;}<br>	∧ -5.reserved not in {;}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {), }}<br>	∧ +1.roles not in {NAME}<br>	∧ +2.reserved not in {=, >}<br>	∧ +3.reserved = (<br>	∧ ^1.roles not in {DECLARATION, IF, OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.946. Support: 1889.` |
| 53 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {(, ,, :, ;, import, {}<br>	∧ -2.internal_type not in {StringLiteral}<br>	∧ -2.reserved not in {=}<br>	∧ -3.reserved not in {;}<br>	∧ -5.reserved not in {;}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = {<br>	∧ +1.roles not in {NAME}<br>	∧ +2.reserved not in {=, >}<br>	∧ +3.reserved not in {(}<br>	∧ ^1.roles not in {DECLARATION, IF, OPERATOR}<br>	∧ ^2.roles in {INCOMPLETE}<br>⇒ y = ∅<br>Confidence: 0.970. Support: 655.` |
| 54 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {(, ,, :, ;, import, {}<br>	∧ -2.internal_type not in {StringLiteral}<br>	∧ -2.reserved not in {=}<br>	∧ -3.reserved not in {;}<br>	∧ -5.reserved not in {;}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {), {, }}<br>	∧ +1.roles not in {NAME}<br>	∧ +2.reserved not in {=, >}<br>	∧ +3.reserved not in {(}<br>	∧ ^1.roles not in {DECLARATION, IF, OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.980. Support: 24943.` |
| 55 | `  -1.internal_type not in {StringLiteral}<br>	∧ -2.label in {<space>}<br>	∧ -3.diff_col ≥ 4<br>	∧ +1.internal_type = StringLiteral<br>⇒ y = ␣<br>Confidence: 0.978. Support: 1163.` |
| 56 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = ;<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = }<br>⇒ y = ⏎␣⁻␣⁻<br>Confidence: 0.942. Support: 1183.` |
| 57 | `  •••start_col ≥ 7<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = ;<br>	∧ -3.roles in {PATHNAME}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {export, }}<br>	∧ +3.length ≤ 1<br>	∧ ^2.roles in {FILE}<br>⇒ y = ⏎⏎<br>Confidence: 0.948. Support: 144.` |
| 58 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {;, {}<br>	∧ -2.label in {<space>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = }<br>	∧ ^1.roles in {DECLARATION, FUNCTION}<br>⇒ y = ␣<br>Confidence: 0.958. Support: 154.` |
| 59 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {;, {}<br>	∧ -2.label in {<space>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {}}<br>	∧ +2.reserved = ><br>	∧ ^1.roles in {DECLARATION, FUNCTION}<br>⇒ y = ␣<br>Confidence: 0.997. Support: 143.` |
| 60 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {;, {}<br>	∧ -2.label in {<space>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {}}<br>	∧ +2.reserved not in {>}<br>	∧ ^1.roles in {DECLARATION, FUNCTION}<br>⇒ y = ∅<br>Confidence: 0.956. Support: 1618.` |
| 61 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {(, ;, {}<br>	∧ -1.roles in {EXPRESSION}<br>	∧ -2.label not in {<space>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = =<br>	∧ ^1.roles in {DECLARATION, FUNCTION}<br>⇒ y = ␣<br>Confidence: 0.998. Support: 214.` |
| 62 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {(, ;, {}<br>	∧ -1.roles in {EXPRESSION}<br>	∧ -2.label not in {<space>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {=}<br>	∧ ^1.roles in {DECLARATION, FUNCTION}<br>⇒ y = ∅<br>Confidence: 0.968. Support: 634.` |
| 63 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {;, {}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = ,<br>	∧ ^1.roles in {DECLARATION} and not in {FUNCTION}<br>⇒ y = ∅<br>Confidence: 0.998. Support: 311.` |
| 64 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {;, {}<br>	∧ -3.label in {<newline>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {,}<br>	∧ ^1.roles in {DECLARATION} and not in {FUNCTION}<br>⇒ y = ⏎<br>Confidence: 0.990. Support: 151.` |
| 65 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {;, {}<br>	∧ -3.label not in {<newline>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {,}<br>	∧ ^1.roles in {DECLARATION} and not in {FUNCTION}<br>⇒ y = ␣<br>Confidence: 0.932. Support: 4369.` |
| 66 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = ,<br>	∧ -2.internal_type = Identifier<br>	∧ -2.reserved not in {)}<br>	∧ -5.diff_line = 0<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ ^1.internal_type not in {ObjectExpression}<br>	∧ ^1.roles not in {DECLARATION}<br>⇒ y = ␣<br>Confidence: 0.926. Support: 518.` |
| 67 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = ,<br>	∧ -2.internal_type not in {Identifier}<br>	∧ -2.reserved not in {)}<br>	∧ -5.diff_line = 0<br>	∧ -5.reserved = (<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ ^1.internal_type not in {ObjectExpression}<br>	∧ ^1.roles not in {DECLARATION}<br>⇒ y = ␣<br>Confidence: 0.986. Support: 176.` |
| 68 | `  -1.diff_offset ≥ 2<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, :, ;}<br>	∧ -3.reserved = ;<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.length ≤ 18<br>	∧ ^1.roles not in {DECLARATION, EXPRESSION}<br>⇒ y = ␣<br>Confidence: 0.987. Support: 2015.` |
| 69 | `  -1.diff_offset ≤ 1<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, :, ;}<br>	∧ -3.reserved = ;<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.length ≥ 2<br>	∧ ^1.roles not in {DECLARATION, EXPRESSION, OPERATOR, STATEMENT}<br>⇒ y = ⏎⏎<br>Confidence: 0.960. Support: 338.` |
| 70 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved = return<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {}}<br>	∧ +2.reserved not in {=}<br>	∧ ^1.roles not in {DECLARATION, IF}<br>⇒ y = ␣<br>Confidence: 0.997. Support: 161.` |
| 71 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {,, :, ;, import, return, {}<br>	∧ -2.internal_type not in {StringLiteral}<br>	∧ -3.reserved not in {;}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {function, }}<br>	∧ +1.roles in {IDENTIFIER}<br>	∧ +2.reserved = ><br>	∧ ^1.roles not in {DECLARATION, IF, OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.999. Support: 789.` |
| 72 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {,, :, ;, import, return, {}<br>	∧ -1.roles in {EXPRESSION}<br>	∧ -2.internal_type not in {StringLiteral}<br>	∧ -3.reserved not in {;}<br>	∧ -5.reserved = ;<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {function, }}<br>	∧ +2.reserved not in {=, >}<br>	∧ ^1.roles not in {DECLARATION, IF, OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.976. Support: 555.` |
| 73 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {,, ;, return, {}<br>	∧ -1.roles not in {EXPRESSION}<br>	∧ -1.length ≥ 3<br>	∧ -3.reserved not in {;}<br>	∧ -5.reserved = ;<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {}}<br>	∧ +2.reserved not in {=}<br>	∧ ^1.roles not in {DECLARATION, IF}<br>⇒ y = ␣<br>Confidence: 0.997. Support: 169.` |
| 74 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved = )<br>	∧ -2.internal_type not in {StringLiteral}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = ;<br>	∧ +1.roles not in {CALLEE}<br>	∧ +2.reserved not in {=, >}<br>	∧ ^1.roles not in {DECLARATION, IF, OPERATOR}<br>⇒ y = ∅<br>Confidence: 1.000. Support: 1793.` |
| 75 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved = )<br>	∧ -2.internal_type not in {StringLiteral}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = ,<br>	∧ +1.roles not in {CALLEE}<br>	∧ +2.reserved not in {=, >}<br>	∧ ^1.roles not in {DECLARATION, IF, OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.999. Support: 375.` |
| 76 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {), ,, :, ;, import, return, {}<br>	∧ -2.internal_type not in {StringLiteral}<br>	∧ -3.reserved not in {;}<br>	∧ -5.diff_offset ≥ 6<br>	∧ -5.reserved not in {;}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {function, }}<br>	∧ +1.roles not in {CALLEE}<br>	∧ +2.reserved not in {=, >}<br>	∧ ^1.roles not in {DECLARATION, IF, OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.968. Support: 29115.` |
| 77 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {), ,, :, ;, >, import, return, {}<br>	∧ -2.internal_type not in {StringLiteral}<br>	∧ -3.reserved not in {;}<br>	∧ -5.diff_offset ≤ 5<br>	∧ -5.reserved not in {;}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {function, }}<br>	∧ +1.roles not in {CALLEE}<br>	∧ +2.reserved not in {=, >}<br>	∧ ^1.roles not in {DECLARATION, IF, OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.945. Support: 589.` |
| 78 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {;, {}<br>	∧ -1.roles in {ARGUMENT}<br>	∧ -2.label in {<space>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = )<br>	∧ +2.reserved not in {)}<br>	∧ ^1.roles in {DECLARATION, FUNCTION}<br>⇒ y = ∅<br>Confidence: 0.997. Support: 185.` |
| 79 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {;, {}<br>	∧ -1.roles not in {ARGUMENT}<br>	∧ -2.label in {<space>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +2.reserved not in {)}<br>	∧ ^1.roles in {DECLARATION, FUNCTION}<br>⇒ y = ∅<br>Confidence: 0.923. Support: 1310.` |
| 80 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {;, {}<br>	∧ -3.reserved not in {=}<br>	∧ -4.diff_line = 0<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {,}<br>	∧ ^1.roles in {DECLARATION} and not in {FUNCTION}<br>⇒ y = ␣<br>Confidence: 0.964. Support: 4012.` |
| 81 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = ,<br>	∧ -2.reserved not in {)}<br>	∧ -2.roles in {ARGUMENT}<br>	∧ -5.diff_line = 0<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ ^1.internal_type not in {ObjectExpression}<br>	∧ ^1.roles not in {DECLARATION}<br>⇒ y = ␣<br>Confidence: 0.998. Support: 245.` |
| 82 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = ,<br>	∧ -2.reserved not in {)}<br>	∧ -2.roles not in {ARGUMENT}<br>	∧ -5.diff_line = 0<br>	∧ -5.reserved = (<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ ^1.internal_type not in {ObjectExpression}<br>	∧ ^1.roles not in {DECLARATION}<br>⇒ y = ␣<br>Confidence: 0.990. Support: 246.` |
| 83 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = ,<br>	∧ -2.reserved not in {)}<br>	∧ -2.roles in {MAP} and not in {ARGUMENT}<br>	∧ -5.diff_line = 0<br>	∧ -5.reserved not in {(}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ ^1.internal_type not in {ObjectExpression}<br>	∧ ^1.roles not in {DECLARATION}<br>⇒ y = ␣<br>Confidence: 0.922. Support: 185.` |
| 84 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, ;}<br>	∧ -4.diff_offset ≥ 7<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +2.roles in {EXPRESSION}<br>	∧ ^1.roles in {OPERATOR} and not in {DECLARATION}<br>⇒ y = ␣<br>Confidence: 0.951. Support: 1256.` |
| 85 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, ;}<br>	∧ -2.label in {<space>}<br>	∧ -4.diff_offset ≥ 7<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +2.roles not in {EXPRESSION}<br>	∧ ^1.roles in {OPERATOR} and not in {DECLARATION}<br>⇒ y = ␣<br>Confidence: 0.939. Support: 799.` |
| 86 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {,, :, ;, import, {}<br>	∧ -1.roles in {EXPRESSION}<br>	∧ -3.reserved not in {;}<br>	∧ -5.reserved = ;<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {}}<br>	∧ +1.roles not in {NAME}<br>	∧ +2.reserved not in {=}<br>	∧ ^1.roles not in {DECLARATION, IF, OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.969. Support: 532.` |
| 87 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {,, ;, {}<br>	∧ -1.roles not in {EXPRESSION}<br>	∧ -1.length ≥ 3<br>	∧ -3.reserved not in {;}<br>	∧ -5.reserved = ;<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {}}<br>	∧ +2.reserved not in {=}<br>	∧ ^1.roles not in {DECLARATION, IF}<br>⇒ y = ␣<br>Confidence: 0.999. Support: 352.` |
| 88 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved = )<br>	∧ -2.internal_type not in {StringLiteral}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = ;<br>	∧ +1.roles not in {CALLEE, NAME}<br>	∧ +2.reserved not in {=, >}<br>	∧ ^1.roles not in {DECLARATION, IF, OPERATOR}<br>⇒ y = ∅<br>Confidence: 1.000. Support: 1742.` |
| 89 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved = )<br>	∧ -2.internal_type not in {StringLiteral}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = ,<br>	∧ +1.roles not in {CALLEE, NAME}<br>	∧ +2.reserved not in {=, >}<br>	∧ ^1.roles not in {DECLARATION, IF, OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.999. Support: 354.` |
| 90 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {(, ), ,, :, ;, import, {}<br>	∧ -2.internal_type not in {StringLiteral}<br>	∧ -3.reserved not in {;}<br>	∧ -5.diff_offset ≥ 6<br>	∧ -5.reserved not in {;}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {}}<br>	∧ +1.roles in {COMMENT} and not in {CALLEE, NAME}<br>	∧ +2.reserved not in {;, =, >}<br>	∧ ^1.roles in {FILE} and not in {DECLARATION, IF, OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.999. Support: 572.` |
| 91 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {(, ), ,, :, ;, import, {}<br>	∧ -2.internal_type not in {StringLiteral}<br>	∧ -3.reserved not in {;}<br>	∧ -5.diff_offset ≥ 6<br>	∧ -5.reserved not in {;}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {}}<br>	∧ +1.roles not in {CALLEE, NAME}<br>	∧ +2.reserved not in {;, =, >}<br>	∧ ^1.roles not in {DECLARATION, IF, OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.983. Support: 24305.` |
| 92 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {), ,, :, ;, >, import, {}<br>	∧ -2.internal_type not in {StringLiteral}<br>	∧ -3.reserved not in {;}<br>	∧ -5.diff_offset ≤ 5<br>	∧ -5.reserved not in {;}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {}}<br>	∧ +1.roles not in {CALLEE, NAME}<br>	∧ +2.reserved not in {=, >}<br>	∧ ^1.roles not in {DECLARATION, IF, OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.936. Support: 619.` |
| 93 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {{}<br>	∧ -1.roles in {ARGUMENT}<br>	∧ -2.label in {<space>}<br>	∧ -4.roles in {FUNCTION}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +2.reserved not in {)}<br>	∧ ^1.roles in {DECLARATION, FUNCTION}<br>⇒ y = ∅<br>Confidence: 0.998. Support: 265.` |
| 94 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {{}<br>	∧ -1.roles in {ARGUMENT}<br>	∧ -2.label in {<space>}<br>	∧ -4.roles not in {FUNCTION}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +2.reserved not in {)}<br>	∧ ^1.roles in {DECLARATION, FUNCTION}<br>⇒ y = ␣<br>Confidence: 0.959. Support: 134.` |
| 95 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {{}<br>	∧ -4.diff_line ≥ 1<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {,}<br>	∧ +1.roles in {IMPORT}<br>	∧ ^1.roles in {DECLARATION} and not in {FUNCTION}<br>⇒ y = ⏎<br>Confidence: 0.939. Support: 139.` |
| 96 | `  •••start_col ≥ 6<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = ;<br>	∧ -3.roles in {PATHNAME}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {export, }}<br>	∧ +3.length ≤ 1<br>	∧ ^1.roles not in {DECLARATION}<br>	∧ ^2.roles in {FILE}<br>⇒ y = ⏎⏎<br>Confidence: 0.959. Support: 160.` |
| 97 | `  -1.diff_offset ≥ 2<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, :, ;}<br>	∧ -3.reserved = ;<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ ^1.roles not in {DECLARATION, EXPRESSION}<br>⇒ y = ␣<br>Confidence: 0.959. Support: 2208.` |
| 98 | `  •••start_col ≤ 32<br>	∧ -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {,, :, ;, {}<br>	∧ -3.reserved not in {;}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = }<br>	∧ +2.reserved not in {=}<br>	∧ ^1.roles not in {DECLARATION, IF, INCOMPLETE, OPERATOR}<br>⇒ y = ⏎␣⁻␣⁻<br>Confidence: 0.926. Support: 441.` |
| 99 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {,, {}<br>	∧ -3.reserved = export<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {}}<br>	∧ +2.reserved not in {=}<br>	∧ ^1.roles not in {DECLARATION, IF}<br>⇒ y = ␣<br>Confidence: 0.998. Support: 237.` |
| 100 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {,, :, ;, function, import, return, {}<br>	∧ -2.internal_type not in {StringLiteral}<br>	∧ -3.reserved not in {;}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {}}<br>	∧ +1.roles in {IDENTIFIER}<br>	∧ +2.reserved = ><br>	∧ ^1.roles not in {DECLARATION, IF, OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.999. Support: 824.` |
| 101 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved = )<br>	∧ -2.internal_type not in {StringLiteral}<br>	∧ -4.label not in {<-space>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = ;<br>	∧ +1.roles not in {CALLEE}<br>	∧ +2.reserved not in {=, >}<br>	∧ ^1.roles not in {DECLARATION, IF, OPERATOR}<br>⇒ y = ∅<br>Confidence: 1.000. Support: 1668.` |
| 102 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved = )<br>	∧ -2.internal_type not in {StringLiteral}<br>	∧ -4.label not in {<-space>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = ,<br>	∧ +1.roles not in {CALLEE}<br>	∧ +2.reserved not in {=, >}<br>	∧ ^1.roles not in {DECLARATION, IF, OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.999. Support: 348.` |
| 103 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {(, ), ,, :, ;, function, import, return, {}<br>	∧ -2.internal_type not in {StringLiteral}<br>	∧ -2.length ≤ 1<br>	∧ -3.reserved not in {;}<br>	∧ -4.label not in {<-space>}<br>	∧ -5.diff_offset ≥ 6<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = )<br>	∧ +1.roles not in {CALLEE}<br>	∧ +2.reserved not in {=, >}<br>	∧ ^1.roles not in {DECLARATION, IF, OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.961. Support: 1335.` |
| 104 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {(, ), ,, :, ;, function, import, return, {}<br>	∧ -2.internal_type not in {StringLiteral}<br>	∧ -3.reserved not in {;}<br>	∧ -4.label not in {<-space>}<br>	∧ -5.diff_offset ≥ 6<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {), }}<br>	∧ +1.roles not in {CALLEE}<br>	∧ +2.reserved not in {=, >}<br>	∧ ^1.roles not in {DECLARATION, IF, OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.981. Support: 24671.` |
| 105 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {), ,, :, ;, function, import, return, {}<br>	∧ -2.internal_type not in {StringLiteral}<br>	∧ -3.reserved not in {;}<br>	∧ -4.label not in {<-space>}<br>	∧ -5.diff_offset ≤ 5<br>	∧ -5.reserved not in {(}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {}}<br>	∧ +1.roles not in {CALLEE}<br>	∧ +2.reserved not in {=, >}<br>	∧ ^1.roles not in {DECLARATION, IF, OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.946. Support: 547.` |
| 106 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {;, {}<br>	∧ -2.label in {<space>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +2.reserved = ><br>	∧ ^1.roles in {DECLARATION, FUNCTION}<br>⇒ y = ␣<br>Confidence: 0.997. Support: 154.` |
| 107 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {;, {}<br>	∧ -2.label in {<space>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +2.reserved not in {), >}<br>	∧ ^1.roles in {DECLARATION, FUNCTION}<br>⇒ y = ∅<br>Confidence: 0.940. Support: 1536.` |
| 108 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {;, {}<br>	∧ -4.diff_line = 0<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {,}<br>	∧ ^1.roles in {DECLARATION} and not in {FUNCTION}<br>⇒ y = ␣<br>Confidence: 0.948. Support: 4138.` |
| 109 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {,, :, ;, import, return, {}<br>	∧ -1.roles in {EXPRESSION}<br>	∧ -2.internal_type not in {StringLiteral}<br>	∧ -3.reserved not in {;}<br>	∧ -5.reserved = ;<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {function, }}<br>	∧ +1.roles not in {CALLEE}<br>	∧ +2.reserved not in {=, >}<br>	∧ ^1.roles not in {DECLARATION, IF, OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.982. Support: 588.` |
| 110 | `  •••start_col ≥ 22<br>	∧ -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {,, :, ;, import, return, {}<br>	∧ -2.internal_type not in {StringLiteral}<br>	∧ -3.reserved not in {;}<br>	∧ -5.reserved not in {;}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = )<br>	∧ +1.roles not in {CALLEE}<br>	∧ +2.reserved not in {=, >}<br>	∧ ^1.roles not in {DECLARATION, IF, OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.928. Support: 1774.` |
| 111 | `  •••start_col ≤ 21<br>	∧ -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved = }<br>	∧ -2.internal_type not in {StringLiteral}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = )<br>	∧ +1.roles not in {CALLEE}<br>	∧ +2.reserved not in {=, >}<br>	∧ ^1.roles not in {DECLARATION, IF, OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.960. Support: 238.` |
| 112 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {,, :, ;, import, return}<br>	∧ -1.length ≥ 2<br>	∧ -2.internal_type not in {StringLiteral}<br>	∧ -3.reserved not in {;}<br>	∧ -5.reserved not in {;}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = ]<br>	∧ +1.roles not in {CALLEE}<br>	∧ +2.reserved not in {=, >}<br>	∧ ^1.roles not in {DECLARATION, IF, OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.923. Support: 267.` |
| 113 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved = )<br>	∧ -2.internal_type not in {StringLiteral}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {), function, }}<br>	∧ +1.roles not in {CALLEE}<br>	∧ +2.reserved not in {=, >}<br>	∧ +3.reserved = (<br>	∧ ^1.roles not in {DECLARATION, IF, OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.997. Support: 160.` |
| 114 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {), ,, :, ;, import, return, {}<br>	∧ -2.internal_type not in {StringLiteral}<br>	∧ -3.reserved not in {;}<br>	∧ -5.reserved not in {;}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {), function, {, }}<br>	∧ +1.roles not in {CALLEE}<br>	∧ +2.reserved not in {=, >}<br>	∧ +3.reserved = (<br>	∧ ^1.roles not in {DECLARATION, IF, OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.971. Support: 1752.` |
| 115 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {,, :, ;, import, return, {}<br>	∧ -2.internal_type not in {StringLiteral}<br>	∧ -3.reserved not in {;}<br>	∧ -5.reserved not in {;}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {), function, {, }}<br>	∧ +1.roles not in {CALLEE}<br>	∧ +2.reserved not in {=, >}<br>	∧ +3.reserved not in {(}<br>	∧ ^1.roles not in {DECLARATION, IF, OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.980. Support: 26082.` |
| 116 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {{}<br>	∧ -2.label in {<space>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +2.reserved = ><br>	∧ ^1.roles in {DECLARATION, FUNCTION}<br>⇒ y = ␣<br>Confidence: 0.997. Support: 163.` |
| 117 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {{}<br>	∧ -2.label in {<space>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +2.reserved not in {), >}<br>	∧ ^1.roles in {DECLARATION, FUNCTION}<br>⇒ y = ∅<br>Confidence: 0.933. Support: 1546.` |
| 118 | `  -1.diff_offset ≥ 2<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, :, ;}<br>	∧ -3.reserved = ;<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.length ≤ 19<br>	∧ ^1.roles not in {DECLARATION, EXPRESSION}<br>⇒ y = ␣<br>Confidence: 0.989. Support: 2087.` |
| 119 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {), ,, :, ;, import, return, {}<br>	∧ -2.internal_type not in {StringLiteral}<br>	∧ -3.reserved not in {;}<br>	∧ -5.diff_offset ≤ 5<br>	∧ -5.reserved not in {(, ;}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {function, }}<br>	∧ +1.roles not in {CALLEE}<br>	∧ +2.reserved not in {=, >}<br>	∧ ^1.roles not in {DECLARATION, IF, OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.942. Support: 558.` |
| 120 | `  -1.internal_type not in {StringLiteral}<br>	∧ -2.label in {<space>}<br>	∧ -4.length ≤ 6<br>	∧ +1.internal_type = StringLiteral<br>⇒ y = ␣<br>Confidence: 0.958. Support: 1212.` |
| 121 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {;, {}<br>	∧ -2.label in {<space>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = =<br>	∧ ^1.roles in {DECLARATION, FUNCTION}<br>⇒ y = ␣<br>Confidence: 0.997. Support: 150.` |
| 122 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {;, {}<br>	∧ -2.label in {<space>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {=, }}<br>	∧ ^1.roles in {DECLARATION, FUNCTION}<br>⇒ y = ∅<br>Confidence: 0.955. Support: 1593.` |
| 123 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {(, ;, {}<br>	∧ -1.roles in {EXPRESSION}<br>	∧ -2.label not in {<space>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +2.reserved = ><br>	∧ ^1.roles in {DECLARATION, FUNCTION}<br>⇒ y = ␣<br>Confidence: 0.997. Support: 185.` |
| 124 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {(, ;, {}<br>	∧ -1.roles in {EXPRESSION}<br>	∧ -2.label not in {<space>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +2.reserved not in {>}<br>	∧ ^1.roles in {DECLARATION, FUNCTION}<br>⇒ y = ∅<br>Confidence: 0.963. Support: 654.` |
| 125 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {(, ;, {}<br>	∧ -1.roles not in {EXPRESSION}<br>	∧ -2.label not in {<space>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.roles in {FUNCTION}<br>	∧ +1.length ≥ 3<br>	∧ ^1.roles in {DECLARATION, FUNCTION}<br>⇒ y = ␣<br>Confidence: 0.961. Support: 270.` |
| 126 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {(, ;, {}<br>	∧ -1.roles not in {EXPRESSION}<br>	∧ -1.length ≥ 4<br>	∧ -2.label not in {<space>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.roles not in {FUNCTION}<br>	∧ +1.length ≥ 3<br>	∧ ^1.roles in {DECLARATION, FUNCTION}<br>⇒ y = ␣<br>Confidence: 0.998. Support: 202.` |
| 127 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {(, ;, {}<br>	∧ -1.roles not in {EXPRESSION}<br>	∧ -2.label not in {<space>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.length ≤ 2<br>	∧ ^1.roles in {DECLARATION, FUNCTION}<br>⇒ y = ␣<br>Confidence: 0.971. Support: 1525.` |
| 128 | `  -1.diff_col ≥ 2<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, :, ;}<br>	∧ -3.reserved = ;<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.length ≤ 19<br>	∧ ^1.roles not in {DECLARATION, EXPRESSION}<br>⇒ y = ␣<br>Confidence: 0.992. Support: 2093.` |
| 129 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {,, :, ;, import, return, {}<br>	∧ -2.internal_type not in {StringLiteral}<br>	∧ -3.reserved not in {;, export}<br>	∧ -4.label not in {<-space>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {}}<br>	∧ +1.roles in {IDENTIFIER}<br>	∧ +2.reserved = ><br>	∧ ^1.roles not in {DECLARATION, IF, OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.999. Support: 802.` |
| 130 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved = )<br>	∧ -2.internal_type not in {StringLiteral}<br>	∧ -3.reserved not in {export}<br>	∧ -4.label not in {<-space>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = ;<br>	∧ +2.reserved not in {=, >}<br>	∧ ^1.roles not in {DECLARATION, IF, OPERATOR}<br>⇒ y = ∅<br>Confidence: 1.000. Support: 1748.` |
| 131 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved = )<br>	∧ -2.internal_type not in {StringLiteral}<br>	∧ -3.reserved not in {export}<br>	∧ -4.label not in {<-space>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = ,<br>	∧ +2.reserved not in {=, >}<br>	∧ ^1.roles not in {DECLARATION, IF, OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.999. Support: 365.` |
| 132 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {(, ), ,, :, ;, [, import, return, {}<br>	∧ -2.internal_type not in {StringLiteral}<br>	∧ -2.length ≤ 1<br>	∧ -3.reserved not in {;}<br>	∧ -4.label not in {<-space>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = )<br>	∧ +1.roles not in {CALLEE}<br>	∧ +2.reserved not in {=, >}<br>	∧ ^1.roles not in {DECLARATION, IF, OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.956. Support: 1241.` |
| 133 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {(, ), ,, :, ;, [, import, return, {}<br>	∧ -2.internal_type not in {StringLiteral}<br>	∧ -3.reserved not in {;}<br>	∧ -4.label not in {<-space>}<br>	∧ -5.diff_offset ≥ 6<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {), }}<br>	∧ +1.roles in {COMMENT} and not in {CALLEE}<br>	∧ +2.reserved not in {=, >}<br>	∧ ^1.internal_type = File<br>	∧ ^1.roles not in {DECLARATION, IF, OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.999. Support: 571.` |
| 134 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {(, ), ,, :, ;, [, import, return, {}<br>	∧ -2.internal_type not in {StringLiteral}<br>	∧ -3.reserved not in {;}<br>	∧ -4.label not in {<-space>}<br>	∧ -5.diff_offset ≥ 6<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {), }}<br>	∧ +1.roles not in {CALLEE}<br>	∧ +2.reserved not in {=, >}<br>	∧ ^1.roles not in {DECLARATION, IF, OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.984. Support: 23727.` |
| 135 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {(, ), ,, :, ;, >, [, import, return, {}<br>	∧ -2.internal_type not in {StringLiteral}<br>	∧ -3.reserved not in {;}<br>	∧ -4.label not in {<-space>}<br>	∧ -5.diff_offset ≤ 5<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {), }}<br>	∧ +1.roles not in {CALLEE}<br>	∧ +2.reserved not in {=, >}<br>	∧ ^1.roles not in {DECLARATION, IF, OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.966. Support: 515.` |
| 136 | `  •••start_col ≥ 7<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = ;<br>	∧ -3.roles in {PATHNAME}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {export, }}<br>	∧ +3.length ≤ 1<br>	∧ ^2.internal_type = File<br>⇒ y = ⏎⏎<br>Confidence: 0.957. Support: 152.` |
| 137 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {;, {}<br>	∧ -1.roles in {ARGUMENT}<br>	∧ -2.label in {<space>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = )<br>	∧ ^1.roles in {DECLARATION, FUNCTION}<br>⇒ y = ∅<br>Confidence: 0.997. Support: 177.` |
| 138 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {;, {}<br>	∧ -1.roles not in {ARGUMENT}<br>	∧ -2.label in {<space>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {}}<br>	∧ ^1.roles in {DECLARATION, FUNCTION}<br>⇒ y = ∅<br>Confidence: 0.938. Support: 1324.` |
| 139 | `  -1.diff_offset ≥ 4<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {(, ;, {}<br>	∧ -1.roles not in {EXPRESSION}<br>	∧ -2.label not in {<space>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.length ≥ 4<br>	∧ ^1.roles in {DECLARATION, FUNCTION}<br>⇒ y = ␣<br>Confidence: 0.997. Support: 170.` |
| 140 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {(, ;, {}<br>	∧ -1.roles not in {EXPRESSION}<br>	∧ -2.label not in {<space>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.length ≤ 3<br>	∧ ^1.roles in {DECLARATION, FUNCTION}<br>⇒ y = ␣<br>Confidence: 0.959. Support: 1605.` |
| 141 | `  -1.diff_col ≥ 2<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, :, ;}<br>	∧ -3.reserved = ;<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ ^1.roles not in {DECLARATION, EXPRESSION}<br>⇒ y = ␣<br>Confidence: 0.944. Support: 2118.` |
| 142 | `  -1.diff_offset ≥ 3<br>	∧ -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {,, ;, {}<br>	∧ -1.roles not in {EXPRESSION}<br>	∧ -3.reserved not in {;}<br>	∧ -5.reserved = ;<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {}}<br>	∧ +2.reserved not in {=}<br>	∧ ^1.roles not in {DECLARATION, IF}<br>⇒ y = ␣<br>Confidence: 0.998. Support: 325.` |
| 143 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved = )<br>	∧ -2.internal_type not in {StringLiteral}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = ;<br>	∧ +1.roles not in {NAME}<br>	∧ +2.reserved not in {=, >}<br>	∧ ^1.roles not in {DECLARATION, IF, OPERATOR}<br>⇒ y = ∅<br>Confidence: 1.000. Support: 1719.` |
| 144 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved = )<br>	∧ -2.internal_type not in {StringLiteral}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = ,<br>	∧ +1.roles not in {NAME}<br>	∧ +2.reserved not in {=, >}<br>	∧ ^1.roles not in {DECLARATION, IF, OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.999. Support: 377.` |
| 145 | `  •••start_col ≥ 26<br>	∧ -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {), ,, :, ;, import, {}<br>	∧ -2.internal_type not in {StringLiteral}<br>	∧ -3.reserved not in {;}<br>	∧ -5.diff_offset ≥ 6<br>	∧ -5.reserved not in {;}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {}}<br>	∧ +1.roles not in {CALLEE, NAME}<br>	∧ +2.reserved = ;<br>	∧ ^1.roles not in {DECLARATION, IF, OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.961. Support: 963.` |
| 146 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {), ,, :, ;, import, {}<br>	∧ -2.internal_type not in {StringLiteral}<br>	∧ -3.reserved not in {;}<br>	∧ -5.diff_offset ≥ 6<br>	∧ -5.reserved not in {;}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {}}<br>	∧ +1.roles not in {CALLEE, NAME}<br>	∧ +2.reserved not in {;, =, >}<br>	∧ ^1.roles not in {DECLARATION, IF, OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.972. Support: 27978.` |
| 147 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {), ,, :, ;, import, {}<br>	∧ -2.internal_type not in {StringLiteral}<br>	∧ -3.label not in {<space>}<br>	∧ -3.reserved not in {;}<br>	∧ -5.diff_offset ≤ 5<br>	∧ -5.reserved not in {;}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {}}<br>	∧ +1.roles not in {CALLEE, NAME}<br>	∧ +2.reserved not in {=, >}<br>	∧ ^1.roles not in {DECLARATION, IF, OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.926. Support: 546.` |
| 148 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {{}<br>	∧ -2.label in {<space>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = =<br>	∧ ^1.roles in {DECLARATION, FUNCTION}<br>⇒ y = ␣<br>Confidence: 0.997. Support: 166.` |
| 149 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {{}<br>	∧ -2.label in {<space>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {=, }}<br>	∧ ^1.roles in {DECLARATION, FUNCTION}<br>⇒ y = ∅<br>Confidence: 0.960. Support: 1621.` |
| 150 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {{}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {,}<br>	∧ +2.reserved not in {,}<br>	∧ ^1.roles in {DECLARATION} and not in {FUNCTION}<br>⇒ y = ␣<br>Confidence: 0.938. Support: 4355.` |
| 151 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, ;}<br>	∧ -3.diff_offset ≥ 5<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +2.roles in {EXPRESSION}<br>	∧ ^1.roles in {OPERATOR} and not in {DECLARATION}<br>⇒ y = ␣<br>Confidence: 0.958. Support: 1216.` |
| 152 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, ;}<br>	∧ -2.label in {<space>}<br>	∧ -3.diff_offset ≥ 5<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +2.roles not in {EXPRESSION}<br>	∧ ^1.roles in {OPERATOR} and not in {DECLARATION}<br>⇒ y = ␣<br>Confidence: 0.955. Support: 787.` |
| 153 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {,, :, ;, function, import, return, {}<br>	∧ -2.internal_type not in {StringLiteral}<br>	∧ -3.reserved not in {;}<br>	∧ -4.label in {<-space>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {}}<br>	∧ +2.reserved not in {=, >}<br>	∧ ^1.roles not in {DECLARATION, IF, MODULE, OPERATOR, STATEMENT}<br>⇒ y = ∅<br>Confidence: 0.928. Support: 564.` |
| 154 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved = )<br>	∧ -2.internal_type not in {StringLiteral}<br>	∧ -4.label not in {<-space>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = ;<br>	∧ +2.reserved not in {=, >}<br>	∧ ^1.roles not in {DECLARATION, IF, OPERATOR}<br>⇒ y = ∅<br>Confidence: 1.000. Support: 1656.` |
| 155 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved = )<br>	∧ -2.internal_type not in {StringLiteral}<br>	∧ -4.label not in {<-space>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = ,<br>	∧ +2.reserved not in {=, >}<br>	∧ ^1.roles not in {DECLARATION, IF, OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.999. Support: 405.` |
| 156 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {), ,, :, ;, function, import, return, {}<br>	∧ -2.internal_type not in {StringLiteral}<br>	∧ -3.reserved not in {;}<br>	∧ -4.label not in {<-space>}<br>	∧ -5.diff_offset ≥ 6<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {}}<br>	∧ +1.roles not in {CALLEE}<br>	∧ +2.reserved not in {=, >}<br>	∧ ^1.roles not in {DECLARATION, IF, OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.965. Support: 29403.` |

### Note
All statistics are calculated with respect to the "analyze" config. This means that the rules were filtered by
`confidence_threshold` and `support_threshold` values.

<details>
    <summary>Machine-readable report</summary>
```json
{"javascript": {"avg_rule_len": 12.512820512820513, "max_conf": 0.999721109867096, "max_support": 29403, "min_conf": 0.9216216206550598, "min_support": 134, "num_rules": 156}}
```
</details>
