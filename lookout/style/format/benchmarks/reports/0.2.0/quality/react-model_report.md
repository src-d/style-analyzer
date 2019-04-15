# Model report for file:///tmp/top-repos-quality-repos-6e5mbbl_/react HEAD 1034e26fe5e42ba07492a736da7bdf5bf2108bc6

### Dump

```json
{'created_at': '2019-04-09 00:22:21',
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
 'size': '22.0 kB',
 'tags': [],
 'uuid': 'edb8f900-d34f-409e-b75f-da59b4e9b0ed',
 'vendor': 'source{d}',
 'version': [1]}
style.format.analyzer.FormatAnalyzer/[1] file:///tmp/top-repos-quality-repos-6e5mbbl_/react 1034e26fe5e42ba07492a736da7bdf5bf2108bc6

# javascript
120 rules, avg.len. 12.0
## train
PPCR: 0.953672
### report
macro
{'f1-score': 0.7327875727654806,
 'precision': 0.7711706139932073,
 'recall': 0.7095738151576912,
 'support': 328602}
micro
{'f1-score': 0.9727421013870884,
 'precision': 0.9727421013870884,
 'recall': 0.9727421013870884,
 'support': 328602}
weighted
{'f1-score': 0.9712437223073707,
 'precision': 0.9717790085069717,
 'recall': 0.9727421013870884,
 'support': 328602}
### report_full
macro
{'f1-score': 0.6867912833271896,
 'precision': 0.7711706139932073,
 'recall': 0.6476721867466414,
 'support': 344565}
micro
{'f1-score': 0.9496751920400138,
 'precision': 0.9727421013870884,
 'recall': 0.9276769259791331,
 'support': 344565}
weighted
{'f1-score': 0.9443694052780301,
 'precision': 0.9705113594126098,
 'recall': 0.9276769259791331,
 'support': 344565}
## test
PPCR: 0.943562
### report
macro
{'f1-score': 0.7088456150136888,
 'precision': 0.749703791283711,
 'recall': 0.6871271836820407,
 'support': 78845}
micro
{'f1-score': 0.9644872851797831,
 'precision': 0.9644872851797831,
 'recall': 0.9644872851797831,
 'support': 78845}
weighted
{'f1-score': 0.9624437227801903,
 'precision': 0.9634888309910009,
 'recall': 0.9644872851797831,
 'support': 78845}
### report_full
macro
{'f1-score': 0.6640196137187058,
 'precision': 0.749703791283711,
 'recall': 0.6241096699736124,
 'support': 83561}
micro
{'f1-score': 0.9364801793037204,
 'precision': 0.9644872851797831,
 'recall': 0.9100537332008951,
 'support': 83561}
weighted
{'f1-score': 0.9295337172768833,
 'precision': 0.9606445195944826,
 'recall': 0.9100537332008951,
 'support': 83561}
```

## javascript
### Summary
90 rules, avg.len. 11.6

| | |
|-|-|
|Min support|96|
|Max support|61103|
|Min confidence|0.9215246438980103|
|Max confidence|0.9995682239532471|

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
| 1 | `  -1.internal_type = StringLiteral<br>	∧ -3.roles in {INCOMPLETE}<br>⇒ y = "<br>Confidence: 0.999. Support: 993.` |
| 2 | `  -1.internal_type = StringLiteral<br>	∧ -3.roles not in {INCOMPLETE}<br>⇒ y = '<br>Confidence: 0.995. Support: 12633.` |
| 3 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = :<br>	∧ +1.internal_type = StringLiteral<br>⇒ y = ␣<br>Confidence: 0.991. Support: 3944.` |
| 4 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = ,<br>	∧ -5.label in {<newline>}<br>	∧ +1.internal_type = StringLiteral<br>⇒ y = ⏎<br>Confidence: 0.951. Support: 1012.` |
| 5 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = ,<br>	∧ -5.label not in {<newline>}<br>	∧ +1.internal_type = StringLiteral<br>	∧ +2.reserved = :<br>⇒ y = ⏎<br>Confidence: 0.963. Support: 149.` |
| 6 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, :}<br>	∧ -2.roles in {INCOMPLETE}<br>	∧ +1.internal_type = StringLiteral<br>⇒ y = "<br>Confidence: 0.999. Support: 986.` |
| 7 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, :}<br>	∧ -2.label in {<space>}<br>	∧ -2.roles not in {INCOMPLETE}<br>	∧ -3.diff_col ≥ 4<br>	∧ +1.internal_type = StringLiteral<br>⇒ y = ␣<br>Confidence: 0.957. Support: 529.` |
| 8 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, :}<br>	∧ -2.label in {<space>}<br>	∧ -2.roles not in {INCOMPLETE}<br>	∧ -3.diff_col ≤ 3<br>	∧ -4.roles in {RIGHT}<br>	∧ +1.internal_type = StringLiteral<br>⇒ y = ⏎<br>Confidence: 0.967. Support: 227.` |
| 9 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, :}<br>	∧ -2.label in {<space>}<br>	∧ -2.roles not in {INCOMPLETE}<br>	∧ -3.diff_col ≤ 3<br>	∧ -4.internal_type = StringLiteral<br>	∧ -4.roles not in {RIGHT}<br>	∧ +1.internal_type = StringLiteral<br>⇒ y = ⏎␣⁺␣⁺<br>Confidence: 0.954. Support: 183.` |
| 10 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, :}<br>	∧ -2.label not in {<space>}<br>	∧ -2.roles not in {INCOMPLETE}<br>	∧ -3.reserved = .<br>	∧ +1.internal_type = StringLiteral<br>	∧ +2.reserved = )<br>⇒ y = '<br>Confidence: 0.997. Support: 1122.` |
| 11 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, :}<br>	∧ -2.label not in {<space>}<br>	∧ -2.roles not in {INCOMPLETE}<br>	∧ -3.reserved = .<br>	∧ -5.label not in {<newline>}<br>	∧ +1.internal_type = StringLiteral<br>	∧ +1.length ≥ 3<br>	∧ +2.reserved not in {)}<br>⇒ y = ⏎␣⁺␣⁺<br>Confidence: 0.926. Support: 682.` |
| 12 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, :}<br>	∧ -1.roles in {COMMENT}<br>	∧ -2.label not in {<space>}<br>	∧ -2.roles not in {INCOMPLETE}<br>	∧ -3.reserved not in {.}<br>	∧ +1.internal_type = StringLiteral<br>⇒ y = ⏎<br>Confidence: 0.998. Support: 227.` |
| 13 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = [<br>	∧ -2.label not in {<newline>, <space>}<br>	∧ -2.roles not in {INCOMPLETE}<br>	∧ -3.reserved not in {.}<br>	∧ +1.internal_type = StringLiteral<br>⇒ y = '<br>Confidence: 0.925. Support: 312.` |
| 14 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, :}<br>	∧ -2.label not in {<newline>, <space>}<br>	∧ -2.roles not in {INCOMPLETE}<br>	∧ -3.reserved not in {.}<br>	∧ +1.internal_type = StringLiteral<br>⇒ y = '<br>Confidence: 0.987. Support: 10761.` |
| 15 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = ,<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = }<br>⇒ y = ⏎␣⁻␣⁻<br>Confidence: 0.982. Support: 1504.` |
| 16 | `  •••start_col ≥ 30<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = ,<br>	∧ -2.roles not in {EXPRESSION}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {), }}<br>	∧ +3.length ≥ 4<br>	∧ +5.reserved = ,<br>⇒ y = ␣<br>Confidence: 0.925. Support: 179.` |
| 17 | `  •••start_col ≥ 30<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = ,<br>	∧ -2.roles not in {EXPRESSION}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {), }}<br>	∧ +3.roles in {EXPRESSION}<br>	∧ +3.length ≥ 2<br>	∧ +5.reserved not in {,}<br>⇒ y = ⏎<br>Confidence: 0.927. Support: 1906.` |
| 18 | `  •••start_col ≥ 30<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = ,<br>	∧ -5.label not in {<newline>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {}}<br>	∧ +3.length ≤ 1<br>	∧ ^1.internal_type = CallExpression<br>⇒ y = ␣<br>Confidence: 0.956. Support: 2587.` |
| 19 | `  •••start_col ≥ 30<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = ,<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = ]<br>	∧ +3.length ≤ 1<br>	∧ ^1.internal_type not in {CallExpression}<br>⇒ y = ⏎␣⁻␣⁻<br>Confidence: 0.970. Support: 186.` |
| 20 | `  •••start_col ≥ 30<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = ,<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {], }}<br>	∧ +3.length ≤ 1<br>	∧ +4.roles in {STRING}<br>	∧ ^1.internal_type not in {CallExpression}<br>⇒ y = ⏎<br>Confidence: 0.971. Support: 187.` |
| 21 | `  •••start_col ≥ 30<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = ,<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {), ], }}<br>	∧ +2.reserved not in {:}<br>	∧ +3.length ≤ 1<br>	∧ +4.roles not in {STRING}<br>	∧ ^1.internal_type not in {CallExpression}<br>⇒ y = ␣<br>Confidence: 0.925. Support: 458.` |
| 22 | `  •••start_col ≤ 29<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = ,<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {}}<br>	∧ ^1.roles in {SCOPE}<br>⇒ y = ⏎␣⁻␣⁻<br>Confidence: 0.971. Support: 540.` |
| 23 | `  •••start_col ≤ 29<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = ,<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = )<br>	∧ +3.reserved not in {=}<br>	∧ ^1.roles not in {SCOPE}<br>⇒ y = ⏎␣⁻␣⁻<br>Confidence: 0.934. Support: 113.` |
| 24 | `  •••start_col ≤ 29<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = ,<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = ]<br>	∧ +3.reserved not in {=}<br>	∧ ^1.roles not in {SCOPE}<br>⇒ y = ⏎␣⁻␣⁻<br>Confidence: 0.985. Support: 102.` |
| 25 | `  •••start_col ≤ 29<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = ,<br>	∧ -4.roles not in {IDENTIFIER}<br>	∧ -5.diff_offset ≥ 13<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {), ], }}<br>	∧ +3.reserved not in {=}<br>	∧ ^1.roles not in {SCOPE}<br>⇒ y = ⏎<br>Confidence: 0.951. Support: 3744.` |
| 26 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = ;<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = }<br>⇒ y = ⏎␣⁻␣⁻<br>Confidence: 0.989. Support: 4853.` |
| 27 | `  •••start_col ≤ 14<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = ;<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {}}<br>	∧ +5.reserved = (<br>⇒ y = ⏎⏎<br>Confidence: 0.962. Support: 990.` |
| 28 | `  •••start_col ≤ 14<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = ;<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {}}<br>	∧ +3.length = 0<br>	∧ +5.reserved not in {(}<br>⇒ y = ⏎<br>Confidence: 0.972. Support: 123.` |
| 29 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {(, ,, ;, {}<br>	∧ -3.diff_offset ≥ 5<br>	∧ -3.roles not in {RIGHT}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {(}<br>	∧ +2.roles in {EXPRESSION}<br>	∧ ^1.roles in {OPERATOR}<br>⇒ y = ␣<br>Confidence: 0.996. Support: 3476.` |
| 30 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {(, ,, ;, {}<br>	∧ -2.label in {<space>}<br>	∧ -3.diff_offset ≥ 5<br>	∧ -3.roles not in {RIGHT}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {(}<br>	∧ +2.roles not in {EXPRESSION}<br>	∧ ^1.roles in {OPERATOR}<br>⇒ y = ␣<br>Confidence: 0.985. Support: 2953.` |
| 31 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, ;, {}<br>	∧ -2.label not in {<space>}<br>	∧ -3.diff_offset ≥ 5<br>	∧ -3.roles not in {RIGHT}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = =<br>	∧ +2.roles not in {EXPRESSION}<br>	∧ ^1.roles in {OPERATOR}<br>⇒ y = ␣<br>Confidence: 0.998. Support: 239.` |
| 32 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = {<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.roles not in {MAP}<br>	∧ ^1.roles in {DECLARATION} and not in {OPERATOR}<br>⇒ y = ⏎␣⁺␣⁺<br>Confidence: 0.989. Support: 2775.` |
| 33 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = (<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = <<br>	∧ ^1.roles in {DECLARATION} and not in {OPERATOR}<br>⇒ y = ⏎␣⁺␣⁺<br>Confidence: 0.980. Support: 176.` |
| 34 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = (<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {<}<br>	∧ ^1.roles in {DECLARATION} and not in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.983. Support: 2706.` |
| 35 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {(, ,, ;, {}<br>	∧ -1.length ≥ 3<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +2.reserved = )<br>	∧ ^1.roles in {DECLARATION} and not in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.997. Support: 1245.` |
| 36 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {(, ,, ;, {}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +2.reserved not in {)}<br>	∧ +2.roles in {ARGUMENT}<br>	∧ ^1.roles in {DECLARATION} and not in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.980. Support: 1238.` |
| 37 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {(, ,, ;, {}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = )<br>	∧ +2.reserved not in {)}<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.roles in {DECLARATION} and not in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.987. Support: 1062.` |
| 38 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {(, ,, ;, {}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = ><br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.roles in {DECLARATION} and not in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.999. Support: 507.` |
| 39 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = <<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {), >}<br>	∧ +2.reserved not in {)}<br>	∧ ^1.roles in {DECLARATION} and not in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.997. Support: 453.` |
| 40 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {(, ,, ;, <, {}<br>	∧ -3.reserved not in {,}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = }<br>	∧ ^1.roles in {DECLARATION} and not in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.995. Support: 102.` |
| 41 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {(, ,, ;, <, {}<br>	∧ -3.reserved not in {,}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = ,<br>	∧ ^1.roles in {DECLARATION} and not in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.995. Support: 98.` |
| 42 | `  -1.diff_col ≤ 56<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {(, ,, ;, <, {}<br>	∧ -3.reserved not in {,}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {), ,, >, }}<br>	∧ +2.reserved not in {)}<br>	∧ +2.roles not in {ARGUMENT}<br>	∧ ^1.roles in {DECLARATION} and not in {INCOMPLETE}<br>⇒ y = ␣<br>Confidence: 0.977. Support: 11610.` |
| 43 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = :<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ ^1.roles not in {DECLARATION, OPERATOR}<br>	∧ ^2.roles not in {STATEMENT}<br>⇒ y = ␣<br>Confidence: 0.994. Support: 4789.` |
| 44 | `  -1.diff_offset ≥ 2<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, :, ;}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +2.reserved = =<br>	∧ ^1.roles not in {DECLARATION}<br>⇒ y = ␣<br>Confidence: 0.953. Support: 3946.` |
| 45 | `  -1.diff_offset ≤ 1<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, :, ;}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.roles in {LEFT}<br>	∧ +2.reserved = =<br>	∧ ^1.roles not in {BLOCK, DECLARATION, OPERATOR}<br>⇒ y = ⏎␣⁺␣⁺<br>Confidence: 0.982. Support: 139.` |
| 46 | `  -1.diff_offset ≤ 1<br>	∧ -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, :, ;}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.roles not in {LEFT}<br>	∧ +2.reserved = =<br>	∧ ^1.roles not in {BLOCK, DECLARATION, OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.976. Support: 2132.` |
| 47 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = {<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +2.reserved not in {=}<br>	∧ ^1.roles not in {DECLARATION, OPERATOR}<br>	∧ ^2.roles in {SCOPE}<br>⇒ y = ⏎␣⁺␣⁺<br>Confidence: 0.996. Support: 1837.` |
| 48 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, :, ;, {}<br>	∧ -1.roles in {EXPRESSION}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +2.reserved not in {=}<br>	∧ ^1.roles not in {DECLARATION, OPERATOR}<br>	∧ ^2.roles in {SCOPE}<br>⇒ y = ∅<br>Confidence: 0.989. Support: 2096.` |
| 49 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {,, :, ;, {}<br>	∧ -1.roles not in {EXPRESSION}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = ;<br>	∧ +2.reserved not in {=}<br>	∧ ^1.roles not in {DECLARATION, OPERATOR}<br>	∧ ^2.roles in {SCOPE}<br>⇒ y = ∅<br>Confidence: 0.997. Support: 165.` |
| 50 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved = (<br>	∧ -1.roles not in {EXPRESSION}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {)}<br>	∧ +2.reserved not in {=}<br>	∧ ^1.roles not in {DECLARATION, OPERATOR}<br>	∧ ^2.roles in {SCOPE}<br>⇒ y = ∅<br>Confidence: 0.939. Support: 107.` |
| 51 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.reserved not in {(, ,, :, ;, {}<br>	∧ -1.roles not in {EXPRESSION}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {), ;}<br>	∧ ^1.roles not in {DECLARATION}<br>	∧ ^2.roles in {SCOPE}<br>⇒ y = ␣<br>Confidence: 0.968. Support: 3891.` |
| 52 | `  -1.internal_type = CommentLine<br>	∧ -1.reserved not in {,, :, ;}<br>	∧ +2.reserved not in {=}<br>	∧ ^1.roles not in {DECLARATION, OPERATOR}<br>	∧ ^2.roles not in {SCOPE}<br>⇒ y = ⏎<br>Confidence: 0.930. Support: 1696.` |
| 53 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {,, :, ;}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.roles in {IDENTIFIER}<br>	∧ +2.reserved = ><br>	∧ ^1.roles not in {DECLARATION, OPERATOR}<br>	∧ ^2.roles not in {SCOPE}<br>⇒ y = ∅<br>Confidence: 0.997. Support: 2198.` |
| 54 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {,, :, ;}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = }<br>	∧ +1.roles not in {IDENTIFIER}<br>	∧ +2.reserved = ><br>	∧ ^1.roles not in {DECLARATION, OPERATOR}<br>	∧ ^2.roles not in {SCOPE}<br>⇒ y = ∅<br>Confidence: 0.984. Support: 335.` |
| 55 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {,, :, ;}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {}}<br>	∧ +1.roles not in {IDENTIFIER}<br>	∧ +2.reserved = ><br>	∧ ^1.roles not in {DECLARATION}<br>	∧ ^2.roles not in {SCOPE}<br>⇒ y = ␣<br>Confidence: 0.981. Support: 3153.` |
| 56 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {,, :, ;}<br>	∧ -1.length ≥ 4<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.roles in {NAME}<br>	∧ ^1.roles not in {DECLARATION}<br>	∧ ^2.roles not in {SCOPE}<br>⇒ y = ␣<br>Confidence: 1.000. Support: 1087.` |
| 57 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved = if<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ ^1.roles not in {DECLARATION}<br>	∧ ^2.roles not in {SCOPE}<br>⇒ y = ␣<br>Confidence: 1.000. Support: 1158.` |
| 58 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved = {<br>	∧ -3.reserved = )<br>	∧ -5.diff_offset ≥ 6<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +2.reserved not in {=, >}<br>	∧ ^1.roles not in {DECLARATION, OPERATOR}<br>	∧ ^2.roles not in {SCOPE}<br>⇒ y = ⏎␣⁺␣⁺<br>Confidence: 0.998. Support: 228.` |
| 59 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved = {<br>	∧ -3.reserved not in {)}<br>	∧ -5.diff_offset ≥ 6<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.roles not in {NAME}<br>	∧ +2.reserved not in {=, >}<br>	∧ +4.internal_type not in {Identifier}<br>	∧ +5.roles in {EXPRESSION}<br>	∧ ^1.roles in {INCOMPLETE} and not in {BODY, DECLARATION, OPERATOR}<br>	∧ ^2.roles not in {SCOPE}<br>⇒ y = ∅<br>Confidence: 0.992. Support: 412.` |
| 60 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved = {<br>	∧ -3.reserved not in {)}<br>	∧ -5.diff_offset ≥ 6<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.roles not in {NAME}<br>	∧ +2.reserved not in {=, >}<br>	∧ +4.internal_type not in {Identifier}<br>	∧ +5.roles not in {EXPRESSION}<br>	∧ ^1.roles not in {BODY, DECLARATION, OPERATOR}<br>	∧ ^2.roles not in {SCOPE}<br>⇒ y = ∅<br>Confidence: 0.978. Support: 2871.` |
| 61 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved = {<br>	∧ -5.diff_offset ≤ 5<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +2.reserved not in {=, >}<br>	∧ ^1.roles not in {DECLARATION, OPERATOR}<br>	∧ ^2.roles not in {SCOPE}<br>⇒ y = ⏎␣⁺␣⁺<br>Confidence: 0.969. Support: 1290.` |
| 62 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {,, ;, if, {}<br>	∧ -2.reserved = =<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ ^1.roles not in {DECLARATION}<br>	∧ ^2.roles not in {SCOPE}<br>⇒ y = ␣<br>Confidence: 0.949. Support: 1440.` |
| 63 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {,, :, ;, if, {}<br>	∧ -2.label in {<-space>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = if<br>	∧ +1.length ≥ 2<br>	∧ +2.reserved not in {=, >}<br>	∧ ^1.roles not in {DECLARATION, OPERATOR}<br>	∧ ^2.roles not in {SCOPE}<br>⇒ y = ⏎<br>Confidence: 0.961. Support: 816.` |
| 64 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {,, :, ;, {}<br>	∧ -2.label in {<-space>}<br>	∧ -2.reserved not in {=}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = }<br>	∧ +1.length ≤ 1<br>	∧ +2.reserved not in {=, >}<br>	∧ ^1.roles not in {DECLARATION, OPERATOR}<br>	∧ ^2.roles not in {SCOPE}<br>⇒ y = ⏎␣⁻␣⁻<br>Confidence: 0.952. Support: 1325.` |
| 65 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {,, :, ;, {}<br>	∧ -2.label in {<-space>}<br>	∧ -2.reserved not in {=}<br>	∧ -3.reserved = )<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = )<br>	∧ +1.length ≤ 1<br>	∧ +2.reserved not in {=, >}<br>	∧ ^1.roles not in {DECLARATION, OPERATOR}<br>	∧ ^2.roles not in {SCOPE}<br>⇒ y = ⏎␣⁻␣⁻<br>Confidence: 0.998. Support: 270.` |
| 66 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {,, :, ;, if, {}<br>	∧ -2.label in {<-space>}<br>	∧ -2.reserved not in {=}<br>	∧ -3.reserved = )<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {), }}<br>	∧ +1.roles not in {NAME}<br>	∧ +1.length ≤ 1<br>	∧ +2.reserved not in {=, >}<br>	∧ ^1.roles not in {DECLARATION, OPERATOR}<br>	∧ ^2.roles not in {SCOPE}<br>⇒ y = ∅<br>Confidence: 0.999. Support: 670.` |
| 67 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {,, :, ;, if, {}<br>	∧ -2.label in {<-space>}<br>	∧ -2.reserved not in {=}<br>	∧ -3.reserved not in {)}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {}}<br>	∧ +1.roles not in {NAME}<br>	∧ +1.length ≤ 1<br>	∧ +2.reserved not in {=, >}<br>	∧ ^1.roles not in {DECLARATION, OPERATOR}<br>	∧ ^2.roles not in {SCOPE}<br>⇒ y = ∅<br>Confidence: 0.978. Support: 5407.` |
| 68 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved = return<br>	∧ -2.label not in {<-space>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ ^1.roles not in {DECLARATION}<br>	∧ ^2.roles not in {SCOPE}<br>⇒ y = ␣<br>Confidence: 0.982. Support: 631.` |
| 69 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved = let<br>	∧ -2.label not in {<-space>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ ^1.roles not in {DECLARATION}<br>	∧ ^2.roles not in {SCOPE}<br>⇒ y = ␣<br>Confidence: 0.999. Support: 425.` |
| 70 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {,, ;, return, {}<br>	∧ -2.label not in {<-space>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ ^1.roles not in {DECLARATION}<br>	∧ ^2.internal_type = ClassBody<br>	∧ ^2.roles not in {SCOPE}<br>⇒ y = ␣<br>Confidence: 0.943. Support: 427.` |
| 71 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {,, ;, return, {}<br>	∧ -2.label not in {<-space>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.roles in {BODY}<br>	∧ ^1.roles not in {DECLARATION}<br>	∧ ^2.roles not in {SCOPE}<br>⇒ y = ␣<br>Confidence: 0.994. Support: 265.` |
| 72 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {,, ;, return, {}<br>	∧ -2.label in {<newline>} and not in {<-space>}<br>	∧ -3.diff_offset ≤ 12<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ ^1.roles not in {DECLARATION}<br>	∧ ^2.roles in {FILE} and not in {SCOPE}<br>⇒ y = ␣<br>Confidence: 0.994. Support: 405.` |
| 73 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved = )<br>	∧ -2.label not in {<-space>}<br>	∧ -2.reserved not in {=}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +2.reserved not in {=, >}<br>	∧ +4.reserved = if<br>	∧ ^1.roles not in {DECLARATION, OPERATOR}<br>	∧ ^2.internal_type not in {ClassBody}<br>	∧ ^2.roles not in {SCOPE}<br>⇒ y = ⏎␣⁻␣⁻<br>Confidence: 0.922. Support: 223.` |
| 74 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {), ,, :, ;, return, {}<br>	∧ -2.label not in {<-space>}<br>	∧ -2.reserved not in {=}<br>	∧ -5.diff_line = 0<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.roles not in {NAME}<br>	∧ +2.reserved not in {=, >}<br>	∧ +4.reserved = if<br>	∧ ^1.roles not in {CONDITION, DECLARATION, OPERATOR}<br>	∧ ^2.internal_type not in {ClassBody}<br>	∧ ^2.roles not in {FILE, SCOPE}<br>⇒ y = ∅<br>Confidence: 0.983. Support: 377.` |
| 75 | `  -1.internal_type not in {CommentLine}<br>	∧ -1.reserved not in {,, :, ;, {}<br>	∧ -2.label not in {<-space>}<br>	∧ -2.reserved not in {=}<br>	∧ +1.internal_type = DirectiveLiteral<br>	∧ +2.reserved not in {=, >}<br>	∧ ^1.roles not in {DECLARATION, OPERATOR}<br>	∧ ^2.internal_type not in {ClassBody}<br>	∧ ^2.roles not in {SCOPE}<br>⇒ y = '<br>Confidence: 0.995. Support: 103.` |
| 76 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved = )<br>	∧ -2.label not in {<-space>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = {<br>	∧ ^1.roles not in {DECLARATION}<br>	∧ ^2.roles not in {SCOPE}<br>⇒ y = ␣<br>Confidence: 0.998. Support: 274.` |
| 77 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {), ,, :, ;, return, {}<br>	∧ -2.label not in {<-space>}<br>	∧ -2.reserved not in {=}<br>	∧ -2.roles in {IDENTIFIER}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = {<br>	∧ +1.roles not in {NAME}<br>	∧ +2.reserved not in {=, >}<br>	∧ ^1.roles not in {CONDITION, DECLARATION, OPERATOR}<br>	∧ ^2.internal_type not in {ClassBody}<br>	∧ ^2.roles not in {FILE, SCOPE}<br>⇒ y = ∅<br>Confidence: 0.996. Support: 1610.` |
| 78 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {), ,, :, ;, return, {}<br>	∧ -2.label not in {<-space>}<br>	∧ -2.reserved not in {=}<br>	∧ -2.roles not in {IDENTIFIER}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = {<br>	∧ +1.roles not in {NAME}<br>	∧ +2.reserved not in {=, >}<br>	∧ ^1.internal_type = JSXElement<br>	∧ ^1.roles not in {CONDITION, DECLARATION, OPERATOR}<br>	∧ ^2.internal_type not in {ClassBody}<br>	∧ ^2.roles not in {FILE, SCOPE}<br>⇒ y = ∅<br>Confidence: 0.998. Support: 234.` |
| 79 | `  -1.diff_col ≥ 2<br>	∧ -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {), ,, ;, return, {}<br>	∧ -2.label not in {<-space>}<br>	∧ -2.roles not in {IDENTIFIER}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = {<br>	∧ ^1.internal_type not in {JSXElement}<br>	∧ ^1.roles not in {DECLARATION}<br>	∧ ^2.roles not in {SCOPE}<br>⇒ y = ␣<br>Confidence: 0.966. Support: 133.` |
| 80 | `  -1.diff_col ≥ 4<br>	∧ -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {,, ;, return, {}<br>	∧ -2.label not in {<-space>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.roles in {CALLEE}<br>	∧ ^1.roles not in {DECLARATION}<br>	∧ ^2.roles not in {SCOPE}<br>⇒ y = ␣<br>Confidence: 0.996. Support: 119.` |
| 81 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {,, :, ;}<br>	∧ -2.internal_type not in {JSXIdentifier}<br>	∧ -2.label not in {<-space>}<br>	∧ -2.reserved not in {}}<br>	∧ -2.roles in {ARGUMENT}<br>	∧ -5.label in {<newline>}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = )<br>	∧ +1.roles not in {CALLEE}<br>	∧ +2.reserved not in {=, >}<br>	∧ ^1.roles not in {DECLARATION, OPERATOR}<br>	∧ ^2.internal_type not in {ClassBody}<br>	∧ ^2.roles not in {SCOPE}<br>⇒ y = ⏎␣⁻␣⁻<br>Confidence: 0.984. Support: 96.` |
| 82 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {,, :, ;, let, return}<br>	∧ -2.label not in {<-space>}<br>	∧ -2.reserved not in {=, }}<br>	∧ -2.roles in {IDENTIFIER} and not in {ARGUMENT}<br>	∧ -4.diff_line ≥ 1<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = )<br>	∧ +1.roles not in {CALLEE, NAME}<br>	∧ +2.reserved not in {=, >}<br>	∧ ^1.roles not in {CONDITION, DECLARATION, OPERATOR}<br>	∧ ^2.internal_type not in {ClassBody}<br>	∧ ^2.roles not in {FILE, SCOPE}<br>⇒ y = ∅<br>Confidence: 0.996. Support: 117.` |
| 83 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {,, :, ;, let, return}<br>	∧ -2.label not in {<-space>}<br>	∧ -2.reserved not in {=, }}<br>	∧ -2.roles not in {ARGUMENT}<br>	∧ -4.diff_line = 0<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved = )<br>	∧ +1.roles not in {CALLEE, NAME}<br>	∧ +2.reserved not in {=, >}<br>	∧ ^1.roles not in {CONDITION, DECLARATION, OPERATOR}<br>	∧ ^2.internal_type not in {ClassBody}<br>	∧ ^2.roles not in {FILE, SCOPE}<br>⇒ y = ∅<br>Confidence: 0.986. Support: 8894.` |
| 84 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved = (<br>	∧ -2.label not in {<-space>}<br>	∧ -2.reserved not in {=}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {)}<br>	∧ +1.roles not in {CALLEE, NAME}<br>	∧ +2.reserved not in {=, >}<br>	∧ +3.length ≥ 13<br>	∧ ^1.roles not in {CONDITION, DECLARATION, OPERATOR}<br>	∧ ^2.internal_type not in {ClassBody}<br>	∧ ^2.roles in {CALL} and not in {FILE, SCOPE}<br>⇒ y = ∅<br>Confidence: 0.958. Support: 132.` |
| 85 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved = (<br>	∧ -2.label not in {<-space>}<br>	∧ -2.reserved not in {=}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {)}<br>	∧ +1.roles not in {CALLEE, NAME}<br>	∧ +2.reserved not in {=, >}<br>	∧ +3.length ≤ 12<br>	∧ ^1.roles not in {CONDITION, DECLARATION, OPERATOR}<br>	∧ ^2.internal_type not in {ClassBody}<br>	∧ ^2.roles not in {FILE, SCOPE}<br>⇒ y = ∅<br>Confidence: 0.963. Support: 9092.` |
| 86 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {(, ,, :, ;, let, return, {}<br>	∧ -2.label not in {<-space>}<br>	∧ -2.reserved not in {=}<br>	∧ +1.internal_type not in {StringLiteral}<br>	∧ +1.reserved not in {), {}<br>	∧ +1.roles not in {CALLEE, NAME}<br>	∧ +2.reserved not in {=, >}<br>	∧ ^1.roles not in {CONDITION, DECLARATION, OPERATOR}<br>	∧ ^2.internal_type not in {ClassBody}<br>	∧ ^2.roles in {EXPRESSION} and not in {FILE, SCOPE}<br>⇒ y = ∅<br>Confidence: 0.997. Support: 61103.` |
| 87 | `  -1.diff_col ≤ 2<br>	∧ -1.internal_type not in {CommentLine}<br>	∧ -1.reserved not in {(, ,, :, ;, let, return, {}<br>	∧ -2.label not in {<-space>}<br>	∧ -2.reserved not in {=}<br>	∧ +1.internal_type = Identifier<br>	∧ +1.reserved not in {), {}<br>	∧ +1.roles not in {CALLEE, NAME}<br>	∧ +2.reserved not in {=, >}<br>	∧ ^1.roles not in {CONDITION, DECLARATION, OPERATOR}<br>	∧ ^2.internal_type not in {ClassBody}<br>	∧ ^2.roles not in {EXPRESSION, FILE, SCOPE}<br>⇒ y = ∅<br>Confidence: 0.983. Support: 1420.` |
| 88 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {(, ,, :, ;, let, return, {}<br>	∧ -2.label in {<newline>} and not in {<-space>}<br>	∧ -2.reserved not in {=}<br>	∧ -3.diff_col = 0<br>	∧ -3.length ≥ 20<br>	∧ -4.length ≤ 21<br>	∧ +1.internal_type not in {Identifier, StringLiteral}<br>	∧ +1.reserved not in {), {}<br>	∧ +1.roles not in {CALLEE, NAME}<br>	∧ +2.reserved not in {=, >}<br>	∧ ^1.roles not in {CONDITION, DECLARATION, OPERATOR, STATEMENT}<br>	∧ ^2.internal_type not in {ClassBody}<br>	∧ ^2.roles not in {EXPRESSION, FILE, SCOPE}<br>⇒ y = ∅<br>Confidence: 0.999. Support: 577.` |
| 89 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {(, ,, :, ;, let, return, {}<br>	∧ -2.label in {<newline>} and not in {<-space>}<br>	∧ -2.reserved not in {=}<br>	∧ -3.length ≤ 19<br>	∧ -4.length ≤ 21<br>	∧ +1.internal_type not in {Identifier, StringLiteral}<br>	∧ +1.reserved not in {), {}<br>	∧ +1.roles not in {CALLEE, NAME}<br>	∧ +2.reserved not in {=, >}<br>	∧ ^1.roles not in {CONDITION, DECLARATION, OPERATOR, STATEMENT}<br>	∧ ^2.internal_type not in {ClassBody}<br>	∧ ^2.roles not in {EXPRESSION, FILE, SCOPE}<br>⇒ y = ∅<br>Confidence: 0.990. Support: 4284.` |
| 90 | `  -1.internal_type not in {CommentLine, StringLiteral}<br>	∧ -1.reserved not in {(, ,, :, ;, let, return, {}<br>	∧ -2.label not in {<-space>, <newline>}<br>	∧ -2.reserved not in {=}<br>	∧ +1.internal_type not in {Identifier, StringLiteral}<br>	∧ +1.reserved not in {), {}<br>	∧ +1.roles not in {CALLEE, NAME}<br>	∧ +2.reserved not in {=, >}<br>	∧ ^1.roles not in {CONDITION, DECLARATION, OPERATOR}<br>	∧ ^2.internal_type not in {ClassBody}<br>	∧ ^2.roles not in {EXPRESSION, FILE, SCOPE}<br>⇒ y = ∅<br>Confidence: 0.992. Support: 31650.` |

### Note
All statistics are calculated with respect to the "analyze" config. This means that the rules were filtered by
`confidence_threshold` and `support_threshold` values.

<details>
    <summary>Machine-readable report</summary>
```json
{"javascript": {"avg_rule_len": 11.644444444444444, "max_conf": 0.9995682239532471, "max_support": 61103, "min_conf": 0.9215246438980103, "min_support": 96, "num_rules": 90}}
```
</details>
