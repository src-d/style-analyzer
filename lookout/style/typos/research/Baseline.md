# Baseline model

## Model description

1. For every token generate list of closest on edit distance and most frequent words from "vocabulary" with SymSpell lookout algorithm

2. Take the most frequent candidates for edit distances 0, 1, 2

3. Build classifier on candidates.
* Features: edit distance between token and candidate, frequency of candidate and frequency of token.
* Labels: whether correction is accurate.
* Model: RandomForestClassifier from sklearn library.

4. For every token rank candidates based on pred_proba from classifier

5. If the first suggestion equals to token then token is considered correct, otherwise suggest all options in the order of ranking.


## Datasets creation

1. Randomly pick tokens from the dataset of identifiers statistics.
1. Augmented with random edit typos (insertion, deletion, substitution) with given probability.
1. Split on train and test in given proportions


## Vocabulary

Tokens from splitted identifiers from GitHub repos with known frequencies and vector embeddings.


## Results of model suggestions for small datasets (~11k train, ~4k test)

METRICS        |DETECTION SCORE|TOP1 SCORE CORR|TOP2 SCORE CORR|TOP3 SCORE CORR|TOP1 SCORE ALL |TOP2 SCORE ALL |TOP3 SCORE ALL 
---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------
Accuracy       |         0.860 |         0.709 |         0.786 |         0.788 |         0.776 |         0.806 |         0.807 
Precision      |         0.944 |         0.926 |         1.000 |         1.000 |         0.928 |         1.000 |         1.000 
Recall         |         0.758 |         0.751 |         0.774 |         0.775 |         0.585 |         0.602 |         0.603 
F1             |         0.841 |         0.830 |         0.872 |         0.873 |         0.717 |         0.751 |         0.752 



## Results on whole dataset of identifiers (~400k train, ~44k test)

SCORE | DETECTION | TOP SUGGESTION | TOP2 SUGGESTIONS | TOP3 SUGGESTIONS
----- | --------- | -------------- | ---------------- | ----------------
Accuracy | 0.888 | 0.735 | 0.740 | 0.740
Precision | 0.999 | 0.999 | 1.000 | 1.000
Recall | 0.776 | 0.474 | 0.480 | 0.480
F1 | 0.874 | 0.643 | 0.648 | 0.648


## Top10 corrections for tokens from the whole dataset

Identifier      | Correction | Correction probability
--------------- | ---------- | ----------------------
aaaaaaaaaaaaaaa | aaaaaaaaaaaaaaaa | 1.000
enbutton   | nbutton  | 1.000
stopit  | stopbit  | 1.000
bxf     | buf  | 0.900
carlos  | carlo | 0.900
jazz    | jazzy | 0.900
prepath  | prepatch | 0.900
vttablet | ttablet  | 0.900
