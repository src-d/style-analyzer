import random

def rand_bool(true_prob):
    """
    Returns True with probability true_prob
    """
    return random.uniform(0, 1) < true_prob 

def read_frequencies(file):
    """
    Read token frequencies from file.
    """
    frequencies = {}
    with open(file, 'r') as f:
        for line in f:
            split = line.split()
            frequencies[split[0]] = int(split[1])
    return frequencies

def suggest_corrections(candidates, pred_proba):
    """
    Suggest typos corrections base on candidates and correctness probabilities.
    
    candidates: dataframe with columns 'id', 'typo', 'candidate' and indexed by
                range(len(pred_proba)).
                
    Returns suggestions: {id : [(candidate, correct_prob)]}, candidates are sorted 
                         by correct_prob in a descending order.
    """
    suggestions = {}
    id = -1
    typo = ''
    corrections = []
    for index in range(len(pred_proba)):
        if candidates.loc[index, 'id'] != id:
            if id != -1:
                suggestions[id] = list(sorted(corrections, key=lambda x:-x[1]))
                if suggestions[id][0] == typo:
                    siggestions[id] = [(typo, 1.0)]

            id = candidates.loc[index, 'id']
            typo = candidates.loc[index, 'typo']
            corrections = []

        corrections.append((candidates.loc[index, 'candidate'], pred_proba[index]))

    suggestions[id] = list(sorted(corrections, key=lambda x:-x[1]))  
    return suggestions

def correct(suggestions):
    """
    Returns the first suggestion for every typo.
    """
    corrections = {}
    for id in suggestions.keys():
        corrections[id] = suggestions[id][0][0]
    return corrections

def detection_score(typos, suggestions):
    """
    Calculates score of solution for typo detection problem.
    
    typos: DataFrame which indexed by 'id' and has columns 'typo', 'corrupted'.
    
    suggestions: {id : [(candidate, correct_prob)]}, candidates are sorted 
                 by correct_prob in a descending order .
    """
    scores = {'tp': 0, 'fp' : 0, 'tn' : 0, 'fn' : 0}
    for id in typos.index:
        if typos.loc[id, 'corrupted']:
            if suggestions[id][0][0] != typos.loc[id, 'typo']:
                scores['tp'] += 1
            else:
                scores['fn'] += 1
        else:
            if suggestions[id][0][0] == typos.loc[id, 'typo']:
                scores['tn'] += 1
            else:
                scores['fp'] += 1
    return scores

def first_k_set(corrections, k):
    first_k = set()
    for correction, prob in corrections[:k]:
        first_k.add(correction)
    return first_k

def score_at_k(typos, suggestions, k):
    """
    Calculates score of solution for typo correction problem. 
    The suggestions for typo correction are considered correct 
    if there is a right one among the first k.
    
    typos: DataFrame which is indexed by 'id' and 
            has columns 'typo', 'corrupted'.
    
    suggestions: {id : [(candidate, correct_prob)]}, 
                  candidates inside one suggestions list are 
                  sorted by correct_prob in a descending order .
    """
    scores = {'tp': 0, 'fp' : 0, 'tn' : 0, 'fn' : 0}
    for id in typos.index:
        if typos.loc[id, 'corrupted']:
            if typos.loc[id, 'identifier'] in first_k_set(suggestions[id], k):
                scores['tp'] += 1
            else:
                scores['fn'] += 1
        else:
            if typos.loc[id, 'identifier'] in first_k_set(suggestions[id], k):
                scores['tn'] += 1
            else:
                scores['fp'] += 1
    return scores

def correction_score(typos, corrections):
    """
    Equal to score_at_k(typos, corrections, 1).
    """
    assert typos.shape[0] == corrections.shape[0]
    scores = {'tp': 0, 'fp' : 0, 'tn' : 0, 'fn' : 0}
    for id in typos.index:
        if typos.loc[id, 'corrupted']:
            if corrections[id] == typos.loc[id, 'identifier']:
                scores['tp'] += 1
            else:
                scores['fn'] += 1
        else:
            if corrections[id]== typos.loc[id, 'identifier']:
                scores['tn'] += 1
            else:
                scores['fp'] += 1
    return scores
                
def accuracy(score):
    return (score['tp'] + score['tn']) / sum(score.values())

def precision(score):
    return score['tp'] / (score['tp'] + score['fp']) 

def recall(score):
    return score['tp'] / (score['tp'] + score['fn']) 
            
def f1(score):
    return 2 / (1 / precision(score) + 1 / recall(score))

def print_score_metrics(score, file=None):
    print(score, file=file)
    print('Accuracy:', accuracy(score), file=file)
    print('Precision:', precision(score), file=file)
    print('Recall:', recall(score), file=file)
    print('F1:', f1(score), file=file)
    
def print_suggestion_results(typos, suggestions, file=None):
    print('DETECTION SCORE\n', file=file)
    print_score_metrics(detection_score(typos, suggestions), file=file)
    print('\nFIRST SUGGESTION SCORE\n', file=file)
    print_score_metrics(score_at_k(typos, suggestions, 1), file=file)
    print('\nFIRST TWO SUGGESTIONS SCORE\n', file=file)
    print_score_metrics(score_at_k(typos, suggestions, 2), file=file)
    print('\nFIRST THREE SUGGESTIONS SCORE\n', file=file)
    print_score_metrics(score_at_k(typos, suggestions, 3), file=file)