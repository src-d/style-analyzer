"""Generation of the typo correction candidates. Contains features extraction and serialization."""
from itertools import chain
from multiprocessing import Pool
from typing import Any, Iterable, List, Mapping, NamedTuple, Optional, Set, Tuple, Union

from gensim.models.fasttext import load_facebook_vectors
from gensim.models.keyedvectors import FastTextKeyedVectors, Vocab
from modelforge import merge_strings, Model, split_strings
import numpy
import pandas
from scipy.spatial.distance import cosine
from sourced.ml.core.models.license import DEFAULT_LICENSE
from tqdm import tqdm

from lookout.style.common import merge_dicts
from lookout.style.typos.config import DEFAULT_CORRECTOR_CONFIG
from lookout.style.typos.symspell import EditDistance, SymSpell
from lookout.style.typos.utils import add_context_info, Columns, read_frequencies, read_vocabulary


TypoInfo = NamedTuple("TypoInfo", (("index", int),
                                   ("typo", str),
                                   ("before", str),
                                   ("after", str)))

Features = NamedTuple("Features", (("index", int),
                                   ("typo", str),
                                   ("candidate", str),
                                   ("vec", numpy.ndarray)))


class CandidatesGenerator(Model):
    """
    Looks for candidates for correction of typos and generates features \
    for them. Candidates are generated in three ways: \
    1. Closest by cosine distance of embeddings to the given token. \
    2. Closest by cosine distance to the compound vector of token context. \
    3. Closest by the edit distance and most frequent tokens from vocabulary.
    """

    NAME = "candidates_generator"
    VENDOR = "source{d}"
    DESCRIPTION = "Model that generates candidates to fix typos."
    LICENSE = DEFAULT_LICENSE
    NO_COMPRESSION = ("/wv/vectors/",)

    def __init__(self, **kwargs):
        """Initialize a new instance of CandidatesGenerator."""
        super().__init__(**kwargs)
        self.checker = None
        self.wv = None
        self.tokens = set()
        self.frequencies = {}
        self.min_freq = 0
        self.config = DEFAULT_CORRECTOR_CONFIG["generation"]

    def construct(self, vocabulary_file: str, frequencies_file: str, embeddings_file: str,
                  config: Optional[Mapping[str, Any]] = None) -> None:
        """
        Construct correction candidates generator.

        :param vocabulary_file: Text file used to generate vocabulary of correction \
                                candidates. First token in every line split is added \
                                to the vocabulary.
        :param frequencies_file: Path to the text file with frequencies. Each line must \
                                 be two values separated with a whitespace: "token count".
        :param embeddings_file: Path to the dump of FastText model.
        :param config: Candidates generation configuration, options:
                       neighbors_number: Number of neighbors of context and typo embeddings \
                                         to consider as candidates (int).
                       edit_dist_number: Number of the most frequent tokens among tokens on \
                                         equal edit distance from the typo to consider as \
                                         candidates (int).
                       max_distance: Maximum edit distance for symspell lookup for candidates \
                                    (int).
                       radius: Maximum edit distance from typo allowed for candidates (int).
                       max_corrected_length: Maximum length of prefix in which symspell lookup \
                                             for typos is conducted (int).
                       start_pool_size: Length of data, starting from which multiprocessing is \
                                        desired (int).
                       chunksize: Max size of a chunk for one process during multiprocessing (int).
                       set_min_freq: True to set the frequency of the unknown tokens to the \
                                     minimum frequency in the vocabulary. It is set to zero \
                                     otherwise.
        """
        self.set_config(config)
        self.checker = SymSpell(max_dictionary_edit_distance=self.config["max_distance"],
                                prefix_length=self.config["max_corrected_length"])
        self.checker.load_dictionary(vocabulary_file)
        self.wv = load_facebook_vectors(embeddings_file)
        self.tokens = set(read_vocabulary(vocabulary_file))
        self.frequencies = read_frequencies(frequencies_file)
        if self.config["set_min_freq"]:
            self.min_freq = min(self.frequencies.values())

    def set_config(self, config: Optional[Mapping[str, Any]] = None) -> None:
        """
        Update candidates generation config.

        :param config: Candidates generation configuration, options:
                       neighbors_number: Number of neighbors of context and typo embeddings \
                                         to consider as candidates (int).
                       edit_dist_number: Number of the most frequent tokens among tokens at \
                                         equal edit distance from the typo to consider as \
                                         candidates (int).
                       max_distance: Maximum edit distance for symspell lookup for candidates \
                                    (int).
                       radius: Maximum edit distance from typo allowed for candidates (int).
                       max_corrected_length: Maximum length of prefix in which symspell lookup \
                                             for typos is conducted (int).
                       start_pool_size: Length of data, starting from which multiprocessing is \
                                        desired (int).
                       chunksize: Max size of a chunk for one process during multiprocessing (int).
        """
        if config is None:
            config = {}
        self.config = merge_dicts(self.config, config)

    def expand_vocabulary(self, additional_tokens: Iterable[str]) -> None:
        """
        Add given tokens to the generator's vocabulary.

        :param additional_tokens: Tokens to add to the vocabulary.
        """
        self.tokens.update(additional_tokens)

    def generate_candidates(self, data: pandas.DataFrame, processes_number: int,
                            save_candidates_file: Optional[str] = None) -> pandas.DataFrame:
        """
        Generate candidates for typos inside data.

        :param data: DataFrame which contains column Columns.Token.
        :param processes_number: Number of processes for multiprocessing.
        :param save_candidates_file: File to save candidates to.
        :return: DataFrame containing candidates for corrections \
                 and features for their ranking for each typo.
        """
        data = add_context_info(data)
        typos = [TypoInfo(index, token, before, after) for index, token, before, after in
                 zip(data.index, data[Columns.Token], data[Columns.Before], data[Columns.After])]
        if len(typos) > self.config["start_pool_size"] and processes_number > 1:
            with Pool(min(processes_number, len(typos))) as pool:
                candidates = list(tqdm(pool.imap(
                    self._lookup_corrections_for_token, typos,
                    chunksize=min(self.config["chunksize"], 1 + len(typos) // processes_number)),
                                       total=len(typos)))
        else:
            candidates = [self._lookup_corrections_for_token(t) for t in typos]
        candidates = pandas.DataFrame(list(chain.from_iterable(candidates)))
        candidates.columns = [Columns.Id, Columns.Token, Columns.Candidate, Columns.Features]
        candidates.loc[:, Columns.Id] = candidates[Columns.Id].astype(data.index.dtype)
        if save_candidates_file is not None:
            candidates.to_csv(save_candidates_file, compression="xz")
        return candidates

    def dump(self) -> str:
        """
        Represent the candidates generator.
        """
        return "\n".join((
            "Vocabulary_size %d." % len(self.tokens),
            "Neighbors number %d." % self.config["neighbors_number"],
            "Maximum distance for search %d." % self.config["max_distance"],
            "Maximum distance allowed %d." % self.config["radius"],
            "Token for distance %d." % self.config["edit_dist_number"],
        ))

    def __eq__(self, other: "CandidatesGenerator") -> bool:
        def compare(first, second) -> bool:
            if isinstance(first, numpy.ndarray) or isinstance(second, numpy.ndarray):
                if (first != second).any():
                    return False
            if isinstance(first, dict):
                assert isinstance(second, dict)
                for key, val in first.items():
                    val2 = second[key]
                    if hasattr(val, "__dict__"):
                        if type(val) != type(val2):
                            return False
                        if val.__dict__ != val2.__dict__:
                            return False
                    elif val != val2:
                        return False
                return True
            if first != second:
                return False
            return True

        for key in vars(self):
            if key == "_source":
                continue
            origin = getattr(self, key)
            peak = getattr(other, key)
            if key in ("checker", "wv"):
                for key2 in vars(origin):
                    if not compare(getattr(origin, key2), getattr(peak, key2)):
                        return False
            elif not compare(origin, peak):
                return False
            return True

    def _lookup_corrections_for_token(self, typo_info: TypoInfo) -> List[Features]:
        candidates = []
        candidate_tokens = self._get_candidate_tokens(typo_info)
        typo_vec = self._vec(typo_info.typo)
        for candidate, dist in candidate_tokens:
            if dist < 0:
                continue
            candidate_vec = self._vec(candidate)
            candidates.append(self._generate_features(typo_info, dist, typo_vec,
                                                      candidate, candidate_vec))
        return candidates

    def _get_candidate_tokens(self, typo_info: TypoInfo) -> Set[Tuple[str, int]]:
        candidate_tokens = set()
        last_dist = -1
        edit_candidates_count = 0
        dist_calc = EditDistance(typo_info.typo, "damerau")
        if self.config["edit_dist_number"] > 0:
            for suggestion in self.checker.lookup(typo_info.typo, 2, self.config["max_distance"]):
                if suggestion.distance != last_dist:
                    edit_candidates_count = 0
                    last_dist = suggestion.distance
                if edit_candidates_count >= self.config["edit_dist_number"]:
                    continue
                candidate_tokens.add((suggestion.term, suggestion.distance))
                edit_candidates_count += 1
        if self.config["neighbors_number"] > 0:
            typo_neighbors = self._closest(self._vec(typo_info.typo),
                                           self.config["neighbors_number"])
            candidate_tokens |= set((
                candidate,
                dist_calc.damerau_levenshtein_distance(candidate, self.config["radius"]))
                for candidate in typo_neighbors if candidate in self.tokens)
            if len(typo_info.before + typo_info.after) > 0:
                context_neighbors = self._closest(
                    self._compound_vec("%s %s" % (typo_info.before, typo_info.after)),
                    self.config["neighbors_number"])
                candidate_tokens |= set([(
                    candidate,
                    dist_calc.damerau_levenshtein_distance(candidate, self.config["radius"]))
                    for candidate in context_neighbors if candidate in self.tokens])
        candidate_tokens.add((typo_info.typo, 0))
        return candidate_tokens

    def _generate_features(self, typo_info: TypoInfo, dist: int, typo_vec: numpy.ndarray,
                           candidate: str, candidate_vec: numpy.ndarray,
                           ) -> Features:
        """
        Compile features for a single correction candidate.

        :param typo_info: instance of TypoInfo class.
        :param dist: edit distance from candidate to typo.
        :param typo_vec: embedding of the original token.
        :param candidate: candidate token.
        :param candidate_vec: embedding of the candidate token.
        :return: index, typo and candidate tokens, frequencies info, \
                 cosine distances between embeggings and contexts, \
                 edit distance between the tokens, \
                 embeddings of the tokens and contexts.
        """
        context = "%s %s" % (typo_info.before, typo_info.after)
        before_vec = self._compound_vec(typo_info.before)
        after_vec = self._compound_vec(typo_info.after)
        context_vec = self._compound_vec(context)
        return Features(typo_info.index, typo_info.typo, candidate, numpy.concatenate((
            (
                self._freq(typo_info.typo),
                self._freq(candidate),
                self._freq_relation(typo_info.typo, candidate),
                self._cos(typo_vec, before_vec),
                self._cos(typo_vec, after_vec),
                self._cos(typo_vec, context_vec),
                self._cos(candidate_vec, before_vec),
                self._cos(candidate_vec, after_vec),
                self._cos(candidate_vec, context_vec),
                self._avg_cos(typo_vec, typo_info.before),
                self._avg_cos(typo_vec, typo_info.after),
                self._avg_cos(typo_vec, context),
                self._avg_cos(candidate_vec, typo_info.before),
                self._avg_cos(candidate_vec, typo_info.after),
                self._avg_cos(candidate_vec, context),
                self._min_cos(typo_vec, typo_info.before),
                self._min_cos(typo_vec, typo_info.after),
                self._min_cos(typo_vec, context),
                self._min_cos(candidate_vec, typo_info.before),
                self._min_cos(candidate_vec, typo_info.after),
                self._min_cos(candidate_vec, context),
                self._cos(typo_vec, candidate_vec),
                dist,
                float(dist > 0 or candidate in self.tokens),
            ),
            before_vec,
            after_vec,
            typo_vec,
            candidate_vec,
            context_vec),
        ).astype(numpy.float32))

    def _vec(self, token: str) -> numpy.ndarray:
        return self.wv[token]

    def _freq(self, token: str) -> float:
        return float(self.frequencies.get(token, self.min_freq))

    @staticmethod
    def _cos(first_vec: numpy.ndarray, second_vec: numpy.ndarray) -> float:
        if numpy.linalg.norm(first_vec) * numpy.linalg.norm(second_vec) != 0:
            return cosine(first_vec, second_vec)
        return 1.0

    def _min_cos(self, typo_vec: numpy.ndarray, context: str) -> float:
        if not len(context.split()):
            return 1.0
        return min([2.0] + [self._cos(typo_vec, self._vec(token)) for token in context.split()])

    def _avg_cos(self, typo_vec: numpy.ndarray, context: str) -> float:
        if not len(context.split()):
            return 1.0
        return sum([self._cos(typo_vec, self._vec(token)) for token in
                    context.split()]) / len(context.split())

    def _closest(self, item: Union[numpy.ndarray, str], quantity: int) -> List[str]:
        return [token for token, _ in self.wv.most_similar([item], topn=quantity)]

    def _freq_relation(self, first_token: str, second_token: str) -> float:
        return -numpy.log((1.0 * self._freq(first_token) + 1e-5) /
                          (1.0 * self._freq(second_token) + 1e-5))

    def _compound_vec(self, text: str) -> numpy.ndarray:
        split = text.split()
        compound_vec = numpy.zeros(self.wv.vector_size)
        for token in split:
            compound_vec += self.wv[token]
        return compound_vec

    def _generate_tree(self) -> dict:
        tree = self.__dict__.copy()

        class DummyModel(Model):
            NAME = "dummy"
            VENDOR = "dummy"
            DESCRIPTION = "dummy"
        for key in vars(DummyModel()):
            del tree[key]
        freqkeys = [""] * len(self.frequencies)
        freqvals = numpy.zeros(len(self.frequencies), dtype=numpy.uint32)
        for i, (key, val) in enumerate(sorted(self.frequencies.items())):
            freqkeys[i] = key
            freqvals[i] = val
        tree["frequencies"] = {"keys": merge_strings(freqkeys), "vals": freqvals}
        tree["checker"] = self.checker.__dict__.copy()
        delstrs = set()
        delindexes = numpy.zeros(len(self.checker._deletes), dtype=numpy.uint32)
        dellengths = numpy.zeros_like(delindexes)
        for i, (key, dss) in enumerate(self.checker._deletes.items()):
            delindexes[i] = key
            dellengths[i] = len(dss)
            for ds in dss:
                delstrs.add(ds)
        delstrs = sorted(delstrs)
        delstrs_map = {s: i for i, s in enumerate(delstrs)}
        deldata = numpy.zeros(sum(dellengths), dtype=numpy.uint32)
        offset = 0
        for di in delindexes:
            dss = self.checker._deletes[di]
            for j, ds in enumerate(dss):
                deldata[offset + j] = delstrs_map[ds]
            offset += len(dss)
        tree["checker"]["_deletes"] = {
            "strings": merge_strings(delstrs),
            "indexes": delindexes,
            "lengths": dellengths,
            "data": deldata,
        }
        wordvals = numpy.zeros(len(self.checker._words), dtype=numpy.uint32)
        for key, val in self.checker._words.items():
            wordvals[delstrs_map[key]] = val
        tree["checker"]["_words"] = wordvals
        tree["tokens"] = merge_strings(sorted(self.tokens))
        vocab_strings = [""] * len(self.wv.vocab)
        vocab_counts = numpy.zeros(len(vocab_strings), dtype=numpy.uint32)
        for key, val in self.wv.vocab.items():
            vocab_strings[val.index] = key
            vocab_counts[val.index] = val.count
        tree["wv"] = {
            "vocab": {"strings": merge_strings(vocab_strings), "counts": vocab_counts},
            "vectors": self.wv.vectors,
            "min_n": self.wv.min_n,
            "max_n": self.wv.max_n,
            "bucket": self.wv.bucket,
            "vectors_ngrams": self.wv.vectors_ngrams,
        }
        return tree

    def _load_tree(self, tree: dict) -> None:
        self.__dict__.update(tree)
        self.tokens = set(split_strings(self.tokens))
        self.frequencies = {
            w: self.frequencies["vals"][i]
            for i, w in enumerate(split_strings(self.frequencies["keys"]))}
        self.checker = SymSpell(max_dictionary_edit_distance=self.config["max_distance"])
        self.checker.__dict__.update(tree["checker"])
        deletes = {}
        words = split_strings(self.checker._deletes["strings"])
        lengths = self.checker._deletes["lengths"]
        data = self.checker._deletes["data"]
        offset = 0
        for i, delindex in enumerate(self.checker._deletes["indexes"]):
            length = lengths[i]
            deletes[delindex] = [words[j] for j in data[offset:offset + length]]
            offset += length
        self.checker._deletes = deletes
        self.checker._words = {w: self.checker._words[i] for i, w in enumerate(words)}
        wv = FastTextKeyedVectors(self.wv["vectors"].shape[1], self.wv["min_n"], self.wv["max_n"],
                                  self.wv["bucket"], True)
        wv.vectors = numpy.array(self.wv["vectors"])
        vocab = split_strings(self.wv["vocab"]["strings"])
        wv.vocab = {
            s: Vocab(index=i, count=self.wv["vocab"]["counts"][i])
            for i, s in enumerate(vocab)}
        wv.bucket = self.wv["bucket"]
        wv.index2word = wv.index2entity = vocab
        wv.vectors_ngrams = numpy.array(self.wv["vectors_ngrams"])
        self.wv = wv


def get_candidates_features(candidates: pandas.DataFrame) -> numpy.ndarray:
    """
    Take the feature vectors belonging to the typo correction candidates from the table.
    """
    return numpy.vstack(candidates[Columns.Features].values)


def get_candidates_metadata(candidates: pandas.DataFrame) -> pandas.DataFrame:
    """
    Take the information about the typo correction candidates from the table.
    """
    return candidates[[Columns.Id, Columns.Token, Columns.Candidate]]
