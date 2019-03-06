# Tools for training and testing Typos corrector

Typos corrector is a model for correcting misspellings inside tokens, accounting for their contexts.
All tokens belong to some domain, here the domain is code identifiers (it can also be common natural language, science, etc.).

## Correction process

- Receive a dataset of tokens to check. Their context is used if provided.

- For every given token:

1. **Candidates generation** - generate a set of tokens, considered as possible corrections by the model.

For every candidate a list of features is generated. Features are based on frequency, embedding, similarity and closeness to the
checked token, etc.

For more information on features and generator parameters look at docs for `CandidatesGenerator` in `lookout/style/typos/generation.py`.

2. **Candidates ranking** - for every checked token correction candidates are ranked based on their features.

XGBoost is used for logistic regression. For every candidate label 1 means that it is the correction of the typo, 0 means that it is not.
Candidates are ranked base on the probability of getting 1, obtained from the XGBoost model.

For more information look at docs for `CandidatesRanker` in `lookout/style/typos/ranking.py`.

- If the most probable correction is the checked token itself, model concludes that the token spelled correctly.
Otherwise several most probable candidates are returned as suggestions for correction

## Model requirements

- Vocabulary of *possibly correct* tokens is used for corrections lookup.

All correction candidates will come from this vocabulary. If the analysed token is inside the vocabulary,
it will still be checked by the model (in-word misspelling problem is solved).

For example, it can be N most frequent tokens from the domain, or some dictionary of grammatically correct words.

- Information about token frequencies in the domain is used for candidates ranking. It is important for correction quality.

- Model uses fasttext embeddings tuned for the domain. They are important for ranking candidates based on the token's context.
Script for obtaining such embeddings will be published later.

## Used data

### Data for building the model

- File with vocabulary: tokens considered correct by the model with their frequencies.

Vocabulary for code identifiers tokens is available at
`lookout/style/typos/16k_vocabulary.csv`

- File with frequencies (weights) for tokens (not only from the vocabulary, as many as possible).

Frequencies for code identifiers' tokens are available at
`lookout/style/typos/research/all_frequencies.csv`

- File with pretrained fasttext model.

Pretrained 10-dimensional vectorizer is available at
`lookout/style/typos/research/test_ft.bin`.

### Input

pandas.DataFrame with rows containing:
- A token to check.
- Its context (not obligatory).
- The correction of the checked token (only for training). Correct token must belong to the vocabulary used by the model.

Example of input data for training:

| |typo | token_split | identifier|
|---|----|-------------|-----------|
|0 | tipo | tipo check | typo |
|1 | function | function name | function |

### Correction output

Python dictionary:

- Keys - row indices in the input dataframe for corrected tokens.
- Values - lists of correction candidates with weights.

Example of output:

```
{0: [(typo, 0.99), (type, 0.56)]}
```

## Training

Corrector can be trained on a pandas.DataFrame or on its csv dump. 

Example:

```
import pandas
from lookout.style.typos.corrector import TyposCorrector

corrector = TyposCorrector()
corrector.initialize_generator("lookout/style/typos/16k_vocabulary.csv", "lookout/style/typos/research/all_frequencies.csv",
"lookout/style/typos/research/test_ft.bin")
corrector.train("train_data.csv")
corrector.save("corrector.asdf")
```

## Testing

Corrector can be run on a pandas.DataFrame or on its csv dump. Accuracy, precision, recall and f1 for top-k first suggestions on all tokens (ALL) or only corrected ones (CORR) are used as quality metrics. Function `print_scores` from `lookout/style/typos/research/dev_utils` calculates them all.

Example:

```
import pandas
from lookout.style.typos.corrector import TyposCorrector
from lookout.style.typos.research.dev_utils import print_scores

corrector = TyposCorrector().load("corrector.asdf")
test_data = pandas.read_csv("test_data.csv", index_col=0)
suggestions = corrector.suggest(test_data)
with open("scores.txt", "w") as out:
    print_scores(test_data, suggestions, out)
```

Example of scores output:

METRICS        |DETECTION SCORE|TOP1 SCORE CORR|TOP2 SCORE CORR|TOP3 SCORE CORR|TOP1 SCORE ALL |TOP2 SCORE ALL |TOP3 SCORE ALL 
---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------
Accuracy       |         0.954 |         0.878 |         0.934 |         0.950 |         0.910 |         0.935 |         0.942 
Precision      |         0.998 |         0.997 |         1.000 |         1.000 |         0.997 |         1.000 |         1.000 
Recall         |         0.908 |         0.880 |         0.934 |         0.950 |         0.819 |         0.868 |         0.882 
F1             |         0.951 |         0.935 |         0.966 |         0.974 |         0.899 |         0.929 |         0.937 

## Data preparation

Several scripts for data preparation are available:

1. Data filtering - in the given dataset leave only identifiers which have all their tokens inside the given vocabulary.

2. Pick subset - pick a fixed size portion of rows from the given dataset. Train-test split can also be done.

3. Create typos - corrupt tokens in the given dataset with given probabilities.

All of them can be run through the `preprocessing.py` script. For more information run:

```
python3 preprocessing.py --help
```





