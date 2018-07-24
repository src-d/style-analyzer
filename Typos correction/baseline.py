import pandas
import numpy
import itertools
import random
import pickle

from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from multiprocessing import Pool, cpu_count

import pysymspell.symspell as symspell
from typos_functions import *

MAX_DISTANCE = 2

class Baseline():
    """
    Typos correction model, based on SymSpell lookout algorithm
    
    https://github.com/wolfgarbe/SymSpell
    
    and simple Random Forest classifier, based on token frequencies
    and edit distance between typo and candidate.
    
    Requires file containing tokens frequencies in a format 'token, frequency'.
    
    Training data: dataframe indexed by 'id' and containing columns 'identifier', 'typo'.
    Testing data: dataframe indexed by 'id' and containing column 'typo'.
    
    """
    def __init__(self, pretrained_file=None, frequencies_file=None):
        if pretrained_file is not None:
            self = pickle.load(pretrained_file)
        else:
            self.checker = symspell.SymSpell(max_dictionary_edit_distance=MAX_DISTANCE)
            self.checker.load_dictionary(frequencies_file)
            self.frequencies = read_frequencies(frequencies_file)
           
    def fit(self, train_file, cand_train_file=None):
        train_df = pandas.read_pickle(train_file)
        self.identifiers = train_df.identifier.copy()
        
        if cand_train_file is None:
            self.candidates = self._create_candidates(train_df, 'cand_' + train_file)
        else:
            self.candidates = pandas.read_pickle(cand_train_file)
            
        self.train_matrix = self._create_matrix(self.candidates)
        self.train_labels = self._create_labels()
        self.model = RandomForestClassifier()
        
        self.model.fit(self.train_matrix, self.train_labels)
        
    def dump(self, dump_file):
        with open(dump_file, 'wb') as f:
            pickle.dump(self, f)
        
    def suggest(self, test_file, cand_test_file=None):
        test_df = pandas.read_pickle(test_file)
        if cand_test_file is None:
            test_candidates = self._create_candidates(test_df, 'cand_' + test_file)
        else:
            test_candidates = pandas.read_pickle(cand_test_file)
            
        test_matrix = self._create_matrix(test_candidates)
        test_proba = self.model.predict_proba(test_matrix)
        return suggest_corrections(test_candidates, test_proba[:, 1])
    
    def correct(self, test_file, cand_file=None):
        return correct(self.suggest(test_file, cand_file))
            
    def _freq(self, token):
        try:
            return self.frequencies[token]
        except KeyError:
            return 0
                
    def _create_candidates(self, data, cand_file):
        candidates = pandas.DataFrame(columns=['id', 'typo', 'candidate', 'typo_freq', 'cand_freq', 'distance'])
        for id, row in tqdm(data.iterrows()):
            last_dist = -1
            typo = row.typo
            for suggestion in self.checker.lookup(typo, 2, MAX_DISTANCE):
                if suggestion.distance != last_dist:
                    candidate = suggestion.term
                    candidates = candidates.append(pandas.DataFrame([[id, typo, candidate, self._freq(typo), 
                                                                      self._freq(candidate), suggestion.distance]], 
                                                                    columns=candidates.columns), ignore_index=True)
                    last_dist = suggestion.distance

            if last_dist == -1:
                candidates = candidates.append(pandas.DataFrame([[id, typo, typo, self._freq(typo), self._freq(typo), 0]], 
                                                                columns=candidates.columns), ignore_index=True)
        candidates.to_pickle(cand_file)
        return candidates
        
        
    def _create_labels(self):
        labels = []
        for ind, row in self.candidates.iterrows():
            labels.append(int(row.candidate == self.identifiers[row.id]))
            
        return numpy.array(labels)
    
    def _create_matrix(self, candidates):
        return numpy.array(candidates.loc[:, ['typo_freq', 'cand_freq', 'distance']], dtype='int')  


def baseline(args):
    baseline = Baseline(args.pretrained_file, args.frequencies_file)
    
    if args.train_file is not None:
        baseline.fit(args.train_file, args.cand_train_file)
        
    if args.dump_file is not None:
        baseline.dump(args.dump_file)
        
    if args.test_file is not None:
        suggestions = baseline.suggest(args.test_file, args.cand_test_file)
        with open(args.out_file, 'w') as out_file:
            print_suggestion_results(pandas.read_pickle(args.test_file), 
                                     suggestions, out_file)