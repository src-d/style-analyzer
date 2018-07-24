Baseline model ready. Short description:

1. For every token generate list of closest on edit distance and most frequent words from "vocabulary" with SymSpell lookout algorithm

2. Take the most frequent candidates for edit distances 0, 1, 2

3. Build classifier on candidates. 
Features: edit distance between token and candidate, frequency of candidate and frequency of token. 
Labels: whether correction is accurate. 
Model: RandomForestClassifier from sklearn library.

4. For every token rank candidates based on pred_proba from classifier

5. If the first suggestion equals to token then token is considered correct, otherwise suggest all options in the order of ranking.


DATASET:
15k randomly picked tokens from the dataset of identifiers statistics, augmented with random edit typos (insertion, deletion, substitution) with 50% probability. Split on train and test datasets in proportion 7:3.

---------------------------------------------------------------------------

Results of model suggestions when trained and tested on small datasets:

---------------------------------------------------------------------------
DETECTION SCORE

{'tn': 2303, 'fn': 543, 'tp': 1734, 'fp': 94}
Accuracy: 0.863714163457424
Precision: 0.9485776805251641
Recall: 0.761528326745718
F1: 0.8448233861144945

FIRST SUGGESTION SCORE

{'tn': 2303, 'fn': 944, 'tp': 1333, 'fp': 94}
Accuracy: 0.7779204107830552
Precision: 0.9341275402943238
Recall: 0.585419411506368
F1: 0.7197624190064795

FIRST TWO SUGGESTIONS SCORE

{'tn': 2396, 'fn': 763, 'tp': 1514, 'fp': 1}
Accuracy: 0.8365425759520753
Precision: 0.9993399339933994
Recall: 0.6649099692577953
F1: 0.7985232067510549

FIRST THREE SUGGESTIONS SCORE

{'tn': 2397, 'fn': 760, 'tp': 1517, 'fp': 0}
Accuracy: 0.8373983739837398
Precision: 1.0
Recall: 0.6662274923144489
F1: 0.7996837111228255

---------------------------------------------------------------------------

Results on corrupted whole dataset of identifiers (~400k train, ~44k test):

---------------------------------------------------------------------------
DETECTION SCORE

{'fp': 9, 'fn': 4921, 'tp': 17072, 'tn': 22078}
Accuracy: 0.8881578947368421
Precision: 0.9994730987647094
Recall: 0.7762469876778975
F1: 0.8738291446998004

FIRST SUGGESTION SCORE

{'fp': 9, 'fn': 11562, 'tp': 10431, 'tn': 22078}
Accuracy: 0.7375
Precision: 0.9991379310344828
Recall: 0.47428727322329833
F1: 0.6432337434094904

FIRST TWO SUGGESTIONS SCORE

{'fp': 0, 'fn': 11428, 'tp': 10565, 'tn': 22087}
Accuracy: 0.7407441016333939
Precision: 1.0
Recall: 0.4803801209475742
F1: 0.6489956385527366

FIRST THREE SUGGESTIONS SCORE

{'fp': 0, 'fn': 11428, 'tp': 10565, 'tn': 22087}
Accuracy: 0.7407441016333939
Precision: 1.0
Recall: 0.4803801209475742
F1: 0.6489956385527366
