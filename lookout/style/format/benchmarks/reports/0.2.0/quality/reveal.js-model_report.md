# Model report for file:///tmp/top-repos-quality-repos-6adtmc4j/reveal.js HEAD 0b3e7839ebf4ed8b6c180aca0abafa28c67aee6d

### Dump

```json
{'created_at': '2019-04-08 13:10:08',
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
 'size': '20.0 kB',
 'tags': [],
 'uuid': '0dbeb93a-7bb0-4630-bd86-20158ce4392d',
 'vendor': 'source{d}',
 'version': [1]}
style.format.analyzer.FormatAnalyzer/[1] file:///tmp/top-repos-quality-repos-6adtmc4j/reveal.js 0b3e7839ebf4ed8b6c180aca0abafa28c67aee6d

# javascript
39 rules, avg.len. 10.9
## train
PPCR: 0.828594
### report
macro
{'f1-score': 0.7859987188414619,
 'precision': 0.8246493166522445,
 'recall': 0.7649339216736821,
 'support': 43304}
micro
{'f1-score': 0.9580870127470903,
 'precision': 0.9580870127470903,
 'recall': 0.9580870127470903,
 'support': 43304}
weighted
{'f1-score': 0.9571598257671093,
 'precision': 0.9581584447336557,
 'recall': 0.9580870127470903,
 'support': 43304}
### report_full
macro
{'f1-score': 0.6411359705042906,
 'precision': 0.8246493166522445,
 'recall': 0.5710311497470325,
 'support': 52262}
micro
{'f1-score': 0.8682795136345562,
 'precision': 0.9580870127470903,
 'recall': 0.7938655237074739,
 'support': 52262}
weighted
{'f1-score': 0.8524765413059251,
 'precision': 0.950810551395723,
 'recall': 0.7938655237074739,
 'support': 52262}
## test
PPCR: 0.815168
### report
macro
{'f1-score': 0.6397583654887888,
 'precision': 0.6354380019230728,
 'recall': 0.6863175820336447,
 'support': 8803}
micro
{'f1-score': 0.9541065545836647,
 'precision': 0.9541065545836647,
 'recall': 0.9541065545836647,
 'support': 8803}
weighted
{'f1-score': 0.9546262013189685,
 'precision': 0.9582186763689367,
 'recall': 0.9541065545836647,
 'support': 8803}
### report_full
macro
{'f1-score': 0.5038987366407286,
 'precision': 0.6354380019230728,
 'recall': 0.4501200010356058,
 'support': 10799}
micro
{'f1-score': 0.8569533721048873,
 'precision': 0.9541065545836647,
 'recall': 0.7777571997407168,
 'support': 10799}
weighted
{'f1-score': 0.8431325349255021,
 'precision': 0.9419728050859159,
 'recall': 0.7777571997407168,
 'support': 10799}
```

## javascript
### Summary
21 rules, avg.len. 9.8

| | |
|-|-|
|Min support|91|
|Max support|7187|
|Min confidence|0.9318181872367859|
|Max confidence|0.9995073676109314|

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
| 1 | `  +1.reserved not in {)}<br>	∧ ^1.internal_type = MemberExpression<br>⇒ y = ∅<br>Confidence: 0.995. Support: 7187.` |
| 2 | `  -1.internal_type not in {StringLiteral}<br>	∧ -1.label not in {<space>}<br>	∧ -2.reserved not in {[}<br>	∧ +1.reserved not in {)}<br>	∧ ^1.internal_type not in {MemberExpression}<br>	∧ ^1.roles in {OPERATOR}<br>⇒ y = ␣<br>Confidence: 0.944. Support: 4975.` |
| 3 | `  -1.internal_type = Identifier<br>	∧ +1.reserved = =<br>	∧ ^1.internal_type not in {MemberExpression}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ␣<br>Confidence: 0.999. Support: 477.` |
| 4 | `  -1.internal_type = Identifier<br>	∧ -2.reserved = (<br>	∧ -3.reserved not in {(}<br>	∧ +1.reserved = )<br>	∧ ^1.internal_type not in {MemberExpression}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.998. Support: 258.` |
| 5 | `  -1.internal_type = Identifier<br>	∧ -2.reserved not in {(}<br>	∧ -3.reserved not in {(}<br>	∧ -4.label not in {<space>}<br>	∧ -5.reserved = (<br>	∧ +1.reserved = )<br>	∧ ^1.internal_type not in {MemberExpression}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.995. Support: 101.` |
| 6 | `  -1.internal_type = Identifier<br>	∧ +1.reserved not in {), =}<br>	∧ ^1.internal_type not in {ConditionalExpression, MemberExpression}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.981. Support: 4181.` |
| 7 | `  -1.internal_type not in {Identifier, StringLiteral}<br>	∧ +1.reserved = ;<br>	∧ ^1.internal_type not in {MemberExpression}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.999. Support: 1839.` |
| 8 | `  -1.internal_type = CommentLine<br>	∧ -1.reserved not in {{}<br>	∧ +1.reserved not in {;, }}<br>	∧ ^1.internal_type not in {MemberExpression}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ⏎<br>Confidence: 0.986. Support: 898.` |
| 9 | `  -1.internal_type not in {CommentLine, Identifier}<br>	∧ -1.reserved not in {{}<br>	∧ +1.reserved not in {;, }}<br>	∧ +1.roles in {COMMENT}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 1.000. Support: 1015.` |
| 10 | `  -1.internal_type not in {CommentLine, Identifier}<br>	∧ -1.reserved = ;<br>	∧ +1.reserved not in {;, }}<br>	∧ +1.roles not in {COMMENT}<br>	∧ ^1.internal_type not in {File, MemberExpression}<br>	∧ ^1.roles in {FOR} and not in {OPERATOR}<br>⇒ y = ␣<br>Confidence: 0.984. Support: 91.` |
| 11 | `  -1.internal_type not in {CommentLine, Identifier}<br>	∧ -1.reserved not in {;, {}<br>	∧ -1.roles in {STRING}<br>	∧ -3.reserved = (<br>	∧ +1.reserved not in {;, }}<br>	∧ +1.roles not in {COMMENT}<br>	∧ ^1.internal_type not in {MemberExpression}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = '<br>Confidence: 0.999. Support: 341.` |
| 12 | `  -1.internal_type not in {CommentLine, Identifier}<br>	∧ -1.label in {<space>}<br>	∧ -1.reserved not in {;, {}<br>	∧ -1.roles not in {STRING}<br>	∧ -2.reserved = (<br>	∧ +1.reserved not in {;, }}<br>	∧ +1.roles not in {COMMENT}<br>	∧ ^1.internal_type not in {MemberExpression}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = '<br>Confidence: 0.999. Support: 403.` |
| 13 | `  -1.internal_type not in {CommentLine, Identifier}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved not in {{}<br>	∧ -1.roles not in {STRING}<br>	∧ +1.reserved = ,<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.993. Support: 507.` |
| 14 | `  -1.internal_type not in {CommentLine, Identifier}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved = (<br>	∧ -1.roles not in {STRING}<br>	∧ -2.diff_col ≥ 3<br>	∧ -2.internal_type not in {Identifier}<br>	∧ +1.reserved not in {,, }}<br>	∧ +1.roles not in {COMMENT}<br>	∧ +1.length ≥ 2<br>	∧ ^1.internal_type not in {MemberExpression}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ␣<br>Confidence: 0.996. Support: 386.` |
| 15 | `  -1.internal_type not in {CommentLine, Identifier}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved = (<br>	∧ -1.roles not in {STRING}<br>	∧ +1.reserved = )<br>	∧ +1.length ≤ 1<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.999. Support: 473.` |
| 16 | `  -1.internal_type not in {CommentLine, Identifier}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved = (<br>	∧ -1.roles not in {STRING}<br>	∧ -2.diff_col ≤ 2<br>	∧ +1.reserved not in {), ,, }}<br>	∧ +1.length ≤ 1<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.932. Support: 110.` |
| 17 | `  •••start_col ≥ 7<br>	∧ -1.internal_type not in {CommentLine, Identifier}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved not in {(, ;, {}<br>	∧ -1.roles in {COMMENT} and not in {STRING}<br>	∧ +1.reserved not in {,, ;, }}<br>	∧ +1.roles not in {KEY}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ∅<br>Confidence: 0.997. Support: 148.` |
| 18 | `  •••start_col ≥ 7<br>	∧ -1.internal_type not in {CommentLine, Identifier}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved not in {{}<br>	∧ -1.roles not in {NUMBER, STRING}<br>	∧ -2.roles in {STRING}<br>	∧ -3.diff_col ≥ 4<br>	∧ +1.reserved = )<br>	∧ +1.roles not in {COMMENT, KEY}<br>	∧ +2.roles not in {ARGUMENT, COMMENT}<br>	∧ +3.roles not in {ASSIGNMENT}<br>	∧ ^1.internal_type not in {MemberExpression}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ␣<br>Confidence: 0.998. Support: 286.` |
| 19 | `  •••start_col ≥ 7<br>	∧ -1.internal_type not in {CommentLine, Identifier}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved not in {(, ;, {, }}<br>	∧ -1.roles not in {NUMBER, STRING}<br>	∧ -3.diff_col ≥ 4<br>	∧ +1.reserved not in {(, ), ,, ;, }}<br>	∧ +1.roles not in {COMMENT, KEY}<br>	∧ +2.roles not in {ARGUMENT, COMMENT}<br>	∧ +3.roles not in {ASSIGNMENT}<br>	∧ ^1.internal_type not in {MemberExpression}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ␣<br>Confidence: 0.953. Support: 4111.` |
| 20 | `  •••start_col ≤ 6<br>	∧ -1.diff_offset ≥ 2<br>	∧ -1.internal_type not in {CommentLine, Identifier}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved = var<br>	∧ -1.roles not in {STRING}<br>	∧ +1.reserved not in {,, }}<br>	∧ +1.roles not in {COMMENT}<br>	∧ +1.length ≥ 2<br>	∧ ^1.internal_type not in {MemberExpression}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ␣<br>Confidence: 0.996. Support: 127.` |
| 21 | `  •••start_col ≤ 6<br>	∧ -1.diff_offset ≥ 2<br>	∧ -1.internal_type not in {CommentLine, Identifier}<br>	∧ -1.label not in {<space>}<br>	∧ -1.reserved not in {(, ;, {}<br>	∧ -1.roles not in {STRING}<br>	∧ -1.length ≤ 2<br>	∧ +1.reserved not in {,, ;, }}<br>	∧ +1.roles not in {COMMENT}<br>	∧ +1.length ≥ 2<br>	∧ ^1.internal_type not in {MemberExpression}<br>	∧ ^1.roles not in {OPERATOR}<br>⇒ y = ⏎<br>Confidence: 0.980. Support: 122.` |

### Note
All statistics are calculated with respect to the "analyze" config. This means that the rules were filtered by
`confidence_threshold` and `support_threshold` values.

<details>
    <summary>Machine-readable report</summary>
```json
{"javascript": {"avg_rule_len": 9.80952380952381, "max_conf": 0.9995073676109314, "max_support": 7187, "min_conf": 0.9318181872367859, "min_support": 91, "num_rules": 21}}
```
</details>
