from itertools import chain
from multiprocessing import Pool
import pickle
from typing import NamedTuple, List, Set, Union

from gensim.models import FastText
from modelforge import merge_strings, split_strings
import numpy
import pandas
from scipy.spatial.distance import cosine
from tqdm import tqdm

from lookout.style.typos.symspell import EditDistance, SymSpell
from lookout.style.typos.utils import (add_context_info, collect_embeddings, read_frequencies,
                                       read_vocabulary, CANDIDATE_COLUMN, ID_COLUMN, TYPO_COLUMN)


TypoInfo = NamedTuple("TypoInfo", [("index", int),
                                   ("typo", str),
                                   ("before", list),
                                   ("after", list)])


class CandidatesGenerator:
    """
    Look for candidates for correction of typos and generates features
    for them. Candidates are generated in three ways:
    1. Closest on cosine distance of embeddings to the given token.
    2. Closest on cosine distance to the compound vector of token context.
    3. Closest on the edit distance and most frequent tokens from vocabulary.
    """
    DEFAULT_RADIUS = 3
    DEFAULT_MAX_DISTANCE = 2
    DEFAULT_NEIGHBORS_NUMBER = 10
    DEFAULT_TAKEN_FOR_DISTANCE = 5

    def __init__(self):
        self.checker = None
        self.fasttext = None
        self.neighbors_number = self.DEFAULT_NEIGHBORS_NUMBER
        self.taken_for_distance = self.DEFAULT_TAKEN_FOR_DISTANCE
        self.max_distance = self.DEFAULT_MAX_DISTANCE
        self.radius = self.DEFAULT_RADIUS
        self.tokens = []
        self.frequencies = {}

    def construct(self, vocabulary_file: str, frequencies_file: str, embeddings_file: str,
                  neighbors_number: int = DEFAULT_NEIGHBORS_NUMBER,
                  taken_for_distance: int = DEFAULT_TAKEN_FOR_DISTANCE,
                  max_distance: int = DEFAULT_MAX_DISTANCE, radius: int = DEFAULT_RADIUS,
                  max_corrected_length: int = 12) -> None:
        """
        Construct correction candidates generator.
        :param vocabulary_file: Text file used to generate vocabulary of corrections candidates.
                                First token in every line split is added to the vocabulary.
        :param frequencies_file: File with lines "token count".
        :param embeddings_file: Dump of fasttext model.
        :param neighbors_number: Number of neighbors of context and typo embeddings
                                 to consider as candidates.
        :param taken_for_distance: Number of the most frequent tokens among tokens on
                                   equal edit distance from the typo to consider as candidates.
        :param max_distance: Maximum edit distance for symspell lookup for candidates.
        :param radius: Maximum edit distance from typo allowed for candidates.
        :param max_corrected_length: Maximum length of prefix in which symspell lookup
                                     for typos is conducted
        """
        self.checker = SymSpell(max_dictionary_edit_distance=max_distance,
                                prefix_length=max_corrected_length)
        self.checker.load_dictionary(vocabulary_file)

        self.fasttext = FastText.load_fasttext_format(embeddings_file)

        self.neighbors_number = neighbors_number
        self.taken_for_distance = taken_for_distance
        self.max_distance = max_distance
        self.radius = radius

        self.tokens = read_vocabulary(vocabulary_file)
        self.frequencies = read_frequencies(frequencies_file)

    def generate_candidates(self, data: pandas.DataFrame, threads_number: int,
                            save_candidates_file: str = None,
                            start_pool_size: int = 64) -> pandas.DataFrame:
        """
        Generates candidates for typos inside data.
        :param data: DataFrame, containing column TYPO_COLUMN.
        :param threads_number: Number of threads for multiprocessing.
        :param save_candidates_file: File to save candidates to.
        :param start_pool_size: Length of data, starting from which multiprocessing is desired.
        :return: DataFrame containing candidates for corrections
                 and features for their ranking for each typo.
        """
        data = add_context_info(data)

        typos = [TypoInfo(index, data.loc[index].typo, data.loc[index].before,
                          data.loc[index].after)
                 for i, index in enumerate(data.index)]

        if len(typos) > start_pool_size:
            with Pool(min(threads_number, len(typos))) as pool:
                candidates = list(tqdm(pool.imap(self._lookup_corrections_for_token, typos,
                                                 chunksize=min(256, 1 + len(typos) //
                                                               threads_number)),
                                       total=len(typos)))
        else:
            candidates = list(map(self._lookup_corrections_for_token, typos))

        candidates = pandas.DataFrame(list(chain.from_iterable(candidates)))
        candidates.columns = ([ID_COLUMN, TYPO_COLUMN, CANDIDATE_COLUMN] +
                              list(range(len(candidates.columns) - 3)))
        candidates[ID_COLUMN] = candidates[ID_COLUMN].astype(data.index.dtype)

        if save_candidates_file is not None:
            candidates.to_pickle(save_candidates_file)

        return candidates

    def _lookup_corrections_for_token(self, typo_info: TypoInfo) -> List[List[Union[str, float]]]:
        candidates = []
        candidate_tokens = self._get_candidate_tokens(typo_info)

        typo_vec = self._vec(typo_info.typo)

        dist_calc = EditDistance(typo_info.typo, "damerau")
        for candidate in set(candidate_tokens):
            candidate_vec = self.fasttext.wv[candidate]
            dist = dist_calc.damerau_levenshtein_distance(candidate, self.radius)

            if dist < 0:
                continue
            candidates.append(self._generate_features(typo_info, dist, typo_vec,
                                                      candidate, candidate_vec))

        return candidates

    def _get_candidate_tokens(self, typo_info: TypoInfo) -> Set[str]:
        candidate_tokens = []

        last_dist = -1
        taken_for_dist = 0
        if self.taken_for_distance > 0:
            for suggestion in self.checker.lookup(typo_info.typo, 2, self.max_distance):
                if suggestion.distance != last_dist:
                    taken_for_dist = 0
                    last_dist = suggestion.distance
                if taken_for_dist >= self.taken_for_distance:
                    continue
                candidate_tokens.append(suggestion.term)
                taken_for_dist += 1

        if self.neighbors_number > 0:
            typo_neighbors = self._closest(self._vec(typo_info.typo), self.neighbors_number)
            candidate_tokens.extend(typo_neighbors)

            if len(typo_info.before + typo_info.after) > 0:
                context_neighbors = self._closest(self._compound_vec(typo_info.before +
                                                                     typo_info.after),
                                                  self.neighbors_number)
                candidate_tokens.extend(context_neighbors)

        candidate_tokens = set([candidate for candidate in candidate_tokens if candidate in self.tokens])
        if not len(candidate_tokens):
            candidate_tokens.add(typo_info.typo)
        return candidate_tokens

    def _generate_features(self, typo_info: TypoInfo, dist: int, typo_vec: numpy.ndarray,
                           candidate: str, candidate_vec: numpy.ndarray
                           ) -> List[Union[str, float]]:
        """
        Features for correction candidate.
        :param typo_info: instance of TypoInfo class.
        :param dist: edit distance from candidate to typo.
        :param candidate: candidate token.
        :param candidate_vec: candidate token embedding.
        :return: index, typo and candidate tokens, frequencies info,
                 cosine distances between embeggings and contexts,
                 edit distance between the tokens, embeddings of
                 the tokens and contexts.
        """
        before_vec = self._compound_vec(typo_info.before)
        after_vec = self._compound_vec(typo_info.after)
        context_vec = self._compound_vec(typo_info.before + typo_info.after)
        return ([typo_info.index, typo_info.typo, candidate,
                 self._freq(typo_info.typo),
                 self._freq(candidate),
                 self._freq_relation(typo_info.typo, candidate),
                 self._cos(typo_vec, before_vec),
                 self._cos(typo_vec, after_vec),
                 self._cos(typo_vec, context_vec),
                 self._cos(candidate_vec, before_vec),
                 self._cos(candidate_vec, after_vec),
                 self._cos(candidate_vec, context_vec),
                 self._cos(typo_vec, candidate_vec),
                 dist] + list(before_vec) + list(after_vec) +
                list(typo_vec) + list(candidate_vec) + list(context_vec))

    def _vec(self, token: str) -> numpy.ndarray:
        return self.fasttext.wv[token]

    def _freq(self, token: str) -> float:
        return float(self.frequencies.get(token, 0))

    @staticmethod
    def _cos(first_vec: numpy.ndarray, second_vec: numpy.ndarray) -> float:
        if numpy.linalg.norm(first_vec) * numpy.linalg.norm(second_vec) != 0:
            return cosine(first_vec, second_vec)
        return 1.0

    def _closest(self, item: Union[numpy.ndarray, str], quantity: int) -> List[str]:
        return [token for token, _ in self.fasttext.wv.most_similar([item], topn=quantity)]

    def _freq_relation(self, first_token: str, second_token: str) -> float:
        return -numpy.log((1.0 * self._freq(first_token) + 1e-5) /
                          (1.0 * self._freq(second_token) + 1e-5))

    def _compound_vec(self, split: List[str]) -> numpy.ndarray:
        compound_vec = numpy.zeros(self.fasttext.wv["a"].shape)
        if len(split) == 0:
            return compound_vec
        else:
            for token in split:
                compound_vec += self.fasttext.wv[token]
        return compound_vec

    def _generate_tree(self) -> dict:
        tree = self.__dict__.copy()
        tree["checker"] = self.checker.__dict__.copy()
        tree["tokens"] = merge_strings(self.tokens)
        tree["fasttext"] = pickle.dumps(self.fasttext)
        return tree

    def _load_tree(self, tree: dict) -> None:
        self.__dict__.update(tree)

        self.tokens = split_strings(self.tokens)
        self.fasttext = pickle.loads(self.fasttext)
        self.checker = SymSpell(max_dictionary_edit_distance=self.max_distance)
        checker_tree = tree["checker"]
        checker_tree["_deletes"] = {int(h): deletes
                                    for h, deletes in checker_tree["_deletes"].items()}
        self.checker.__dict__.update(checker_tree)

    def __str__(self):
        return ("Vocabulary_size %d. \n"
                "Neighbors number %d. \n"
                "Maximum distance for search %d. \n"
                "Maximum distance allowed %d. \n"
                "Taken for distance %d.") % (len(self.tokens), self.neighbors_number,
                                             self.max_distance, self.radius,
                                             self.taken_for_distance)


def get_candidates_features(candidates: pandas.DataFrame) -> numpy.ndarray:
    return candidates.drop(columns=[ID_COLUMN, TYPO_COLUMN,
                                    CANDIDATE_COLUMN]).as_matrix().astype(float)


def get_candidates_tokens(candidates: pandas.DataFrame) -> pandas.DataFrame:
    return candidates[[ID_COLUMN, TYPO_COLUMN, CANDIDATE_COLUMN]]
