from multiprocessing import Pool
from os import path
import pickle

import numpy
import pandas
from sklearn.ensemble import RandomForestClassifier

from lookout.style.typos.research.dev_utils import (correct as typos_functions_correct,
                                                    print_suggestion_results,
                                                    read_frequencies,
                                                    suggest_corrections)
import lookout.style.typos.symspell as symspell

MAX_DISTANCE = 2


class Baseline:
    """
    Typos correction model, based on SymSpell lookout algorithm

    https://github.com/wolfgarbe/SymSpell

    and simple Random Forest classifier, based on token frequencies
    and edit distance between typo and candidate.

    Requires file containing tokens frequencies in a format "token, frequency".

    Training data: dataframe indexed by "id" and containing columns "identifier", "typo".
    Testing data: dataframe indexed by "id" and containing column "typo".

    """

    def __init__(self, frequencies_file):
        self.checker = symspell.SymSpell(max_dictionary_edit_distance=MAX_DISTANCE)
        self.checker.load_dictionary(frequencies_file)
        self.frequencies = read_frequencies(frequencies_file)

    def fit(self, train_file, cand_train_file=None):
        train_df = pandas.read_pickle(train_file)
        self.identifiers = train_df.identifier.copy()

        if cand_train_file is None:
            self.candidates = self._create_candidates(train_df, "cand_" + train_file)
        else:
            self.candidates = pandas.read_pickle(cand_train_file)

        self.train_matrix = self._create_matrix(self.candidates)
        self.train_labels = self._create_labels()
        self.model = RandomForestClassifier()

        self.model.fit(self.train_matrix, self.train_labels)

    def dump(self, dump_file):
        with open(dump_file, "wb") as f:
            pickle.dump(self, f)

    def suggest(self, test_file, cand_test_file=None):
        test_df = pandas.read_pickle(test_file)
        if cand_test_file is None:
            path_split = path.split(test_file)
            test_candidates = self._create_candidates(test_df, path.join(path_split[0],
                                                                         "cand_" + path_split[1]))
        else:
            test_candidates = pandas.read_pickle(cand_test_file)

        test_matrix = self._create_matrix(test_candidates)
        test_proba = self.model.predict_proba(test_matrix)
        return suggest_corrections(test_candidates, test_proba[:, 1])

    def correct(self, test_file, cand_file=None):
        return typos_functions_correct(self.suggest(test_file, cand_file))

    def _freq(self, token):
        try:
            return self.frequencies[token]
        except KeyError:
            return 0

    def _lookup_corrections(self, typo_info):
        id, typo = typo_info
        candidates = pandas.DataFrame(columns=["id", "typo", "candidate", "typo_freq",
                                               "cand_freq", "distance"])
        last_dist = -1
        for suggestion in self.checker.lookup(typo, 2, MAX_DISTANCE):
            if suggestion.distance != last_dist:
                candidate = suggestion.term
                candidates = candidates.append(pandas.DataFrame([[id, typo, candidate,
                                                                  self._freq(typo),
                                                                  self._freq(candidate),
                                                                  suggestion.distance]],
                                                                columns=candidates.columns),
                                               ignore_index=True)
                last_dist = suggestion.distance

        if last_dist == -1:
            candidates = candidates.append(pandas.DataFrame([[id, typo, typo, self._freq(typo),
                                                              self._freq(typo), 0]],
                                                            columns=candidates.columns),
                                           ignore_index=True)
        return candidates

    def _create_candidates(self, data, cand_file):
        typos = list(data.typo.to_dict().items())
        with Pool(32) as pool:
            candidates_list = pool.map(self._lookup_corrections, typos)

        all_candidates = pandas.DataFrame(columns=["id", "typo", "candidate", "typo_freq",
                                                   "cand_freq", "distance"])
        for candidates in candidates_list:
            all_candidates = all_candidates.append(candidates, ignore_index=True)

        all_candidates.to_pickle(cand_file)
        return all_candidates

    def _create_labels(self):
        labels = []
        for ind, row in self.candidates.iterrows():
            labels.append(int(row.candidate == self.identifiers[row.id]))

        return numpy.array(labels)

    def _create_matrix(self, candidates):
        return numpy.array(candidates.loc[:, ["typo_freq", "cand_freq", "distance"]],
                           dtype="int")


def baseline(args):
    if args.pretrained_file is not None:
        with open(args.pretrained_file, "rb") as f:
            baseline = pickle.load(f)
    else:
        baseline = Baseline(args.frequencies_file)

    if args.train_file is not None:
        baseline.fit(args.train_file, args.cand_train_file)

    if args.dump_file is not None:
        baseline.dump(args.dump_file)

    if args.test_file is not None:
        suggestions = baseline.suggest(args.test_file, args.cand_test_file)
        with open(args.out_file, "w") as out_file:
            print_suggestion_results(pandas.read_pickle(args.test_file), suggestions, out_file)
