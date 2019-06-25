import logging
import os
import pathlib
from pprint import pformat
import sys
import tempfile
from typing import Any, Dict, Mapping, Optional, Tuple
import urllib.request

import pandas
from sklearn.model_selection import train_test_split
from smart_open import smart_open
import spacy
from tqdm import tqdm

from lookout.style.common import merge_dicts
from lookout.style.typos.config import DEFAULT_CORRECTOR_CONFIG
from lookout.style.typos.corrector import TyposCorrector
from lookout.style.typos.corruption import corrupt_tokens_in_df
from lookout.style.typos.symspell import SymSpell
from lookout.style.typos.utils import Columns, flatten_df_by_column, print_frequencies,\
    read_frequencies


class _DownloadProgressBar(tqdm):
    def update_to(self, b: int = 1, bsize: int = 1, tsize: Optional[int] = None) -> None:
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def _download_url(url: str, output_path: str) -> None:
    with _DownloadProgressBar(unit="MB", unit_scale=True,
                              miniters=1, desc=url.split("/")[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def generate_vocabulary(frequencies_path: str, config: Mapping[str, Any]) -> Dict[str, int]:
    """
    Compose vocabulary from a set of tokens with known frequencies.

    Filtering of the input tokens depends on their frequencies and edit distances between them.
    All found English words and tokens that the algorithm considers word-like are added \
    regardless of their frequencies.
    :param frequencies_path: Path to the .csv file with space-separated word-frequency pairs \
                             one-per-line.
    :param config: Configuration for the vocabulary creation:
                   stable: How many tokens, which don't have more frequent \
                           edit-distance-neighbors, to take into the vocabulary.
                   suspicious: How many tokens, whose more frequent edit-distance-neighbor is
                               an English word, to take into the vocabulary.
                   non_suspicious: How many tokens, whose more frequent edit-distance-neighbor \
                                   is not an English word, to take into the vocabulary.
    :return: Dictionary with the vocabulary tokens as keys and their corresponding \
             frequencies as values.
    """
    checker = SymSpell(max_dictionary_edit_distance=2, prefix_length=100)
    checker.load_dictionary(frequencies_path)
    frequencies = read_frequencies(frequencies_path)
    sorted_frequencies = sorted(frequencies.items(), key=lambda x: -x[1])

    # For every token, find a token on edit distance 1, which has higher frequency, if there is one
    def _correct_token(token_freq):
        token, freq = token_freq
        suggestions = checker.lookup(token, 2, 1)
        if len(suggestions) > 1:
            correction = suggestions[1].term
            return correction, frequencies[correction]
        return token, freq
    corrections = list(tqdm(map(_correct_token, sorted_frequencies),
                            total=len(sorted_frequencies)))

    all_tokens = pandas.DataFrame(columns=["token", "token_freq", "correction", "correction_freq"])
    all_tokens["token"] = [token for token, _ in sorted_frequencies]
    all_tokens["token_freq"] = [freq for _, freq in sorted_frequencies]
    all_tokens["correction"] = [token_freq[0] if token_freq[1] > sorted_frequencies[i][1]
                                else sorted_frequencies[i][0]
                                for i, token_freq in enumerate(corrections)]
    all_tokens["correction_freq"] = [token_freq[1] if token_freq[1] > sorted_frequencies[i][1]
                                     else sorted_frequencies[i][1]
                                     for i, token_freq in enumerate(corrections)]
    all_tokens["rel"] = all_tokens["correction_freq"] / all_tokens["token_freq"]

    # Find all English words among all the tokens
    eng_voc = set()
    with smart_open(str(pathlib.Path(__file__).parent / "words_alpha.txt.xz"), "r") as f:
        for line in f:
            eng_voc.add(line.strip())

    # Leave only non-english tokens for analysis
    stable = all_tokens[(all_tokens.rel == 1.0) & ~all_tokens.token.isin(eng_voc)]
    unstable = all_tokens[(all_tokens.rel > 1) & ~all_tokens.token.isin(eng_voc)]

    # Get tokens and their corrections lemmas
    spacy.cli.download("en")
    nlp = spacy.load("en", disable=["parser", "ner"])

    def _lemmatize(token):
        lemm = nlp(token)
        if len(lemm) > 1 or lemm[0].lemma_ == "-PRON-" or (token[-2:] == "ss" and
                                                           lemm[0].lemma_ == token[:-1]):
            return token
        return lemm[0].lemma_
    token_lemma = list(tqdm(map(_lemmatize, list(unstable.token)), total=len(unstable)))
    correction_lemma = list(tqdm(map(_lemmatize, list(unstable.correction)), total=len(unstable)))
    unstable["token_lemma"] = token_lemma
    unstable["cor_lemma"] = correction_lemma

    # Equal lemmas -> different forms of a morphologically changing token -> token is a "word"
    # Use some heuristics to remove noise
    eq_lemmas = unstable[
        (unstable["token_lemma"] == unstable["cor_lemma"]) |
        (unstable["token_lemma"] == unstable["correction"]) &
        (~unstable["correction"].isin(eng_voc) |
         (unstable["correction"].apply(lambda x: x[-3:]) == "ing"))]
    dif_lemmas = unstable[(unstable["token_lemma"] != unstable["cor_lemma"]) &
                          (unstable["token_lemma"] != unstable["correction"])]

    # Stemming heuristics
    def _norm(word: str) -> str:
        if word[-2:] == "ed" or word[-2:] == "er" or word[-1] == "s" and word[-2] != "s":
            return word[:-1]
        return word
    norm_eq = dif_lemmas[(dif_lemmas.token.apply(_norm) == dif_lemmas.correction)]

    # Gather all results
    good = all_tokens[all_tokens.token.isin(set(
        list(eq_lemmas[:].token) + list(eq_lemmas[:].correction) +
        list(norm_eq.token) + list(norm_eq.correction)))]
    unstable = unstable[~unstable.token.isin(good.token)]
    stable = stable[~stable.token.isin(good.token)]

    # Suspicious - have high probability to be typo-ed English words
    suspicious = unstable[unstable.correction.isin(eng_voc)]
    non_suspicious = unstable[~unstable.correction.isin(eng_voc)]
    vocabulary = all_tokens[all_tokens.token.isin(set(
        list(stable[:config["stable"]].token) +
        list(suspicious[:config["suspicious"]].token) +
        list(non_suspicious[:config["non_suspicious"]].token) +
        list(eng_voc) +
        list(good.token)))]
    return {token: freq for token, freq in vocabulary[["token", "token_freq"]].values}


def prepare_data(config: Optional[Mapping[str, Any]] = None) -> pandas.DataFrame:
    """
    Generate all the necessary data from the raw dataset of split identifiers.

    Brief algorithm description:
    1. Derive vocabulary for typos correction which is a set of tokens, which is considered
       correctly spelled. All typos corrections will belong to the vocabulary.
       It is a set of most frequent tokens (based on given statistics).
    2. Save vocabulary and statistics for a given amount of most frequent tokens for future use.
    3. Filter raw data, leaving identifiers, containing only tokens from the vocabulary.
       The result is a dataset of tokens which will be considered correct. It will be used
       for creating artificial misspelling cases for training and testing the corrector model.
    4. Save prepared dataset, if needed.
    :param config: Dictionary with parameters for data preparation. Used fields are:
                   data_dir: Directory to put all derived data to.
                   drive_dataset_id: ID of google drive document, where a raw dataset is stored.
                   input_path: Path to a .csv dump of input dataframe. Should contain \
                               column Columns.Split. If None or file doesn't exist,
                               the dataset will be loaded from Google drive.
                   frequency_column: Name of the column with identifiers frequencies. If not \
                                     specified, every split is considered to have frequency 1.
                   vocabulary_size: Number of most frequent tokens to take as a vocabulary.
                   frequencies_size: Number of most frequent tokens to save frequencies info for. \
                                     This information will be used by corrector as features for \
                                     these tokens when they will be checked. If not specified, \
                                     frequencies for all present tokens will be saved.
                   raw_data_filename: Name of the .csv file in data_dir to put raw dataset \
                                      in case of loading from drive.
                   vocabulary_filename: Name of the .csv file in data_dir to save vocabulary to.
                   frequencies_filename: Name of the .csv file in data_dir to save frequencies to.
                   prepared_filename: Name of the .csv file in data_dir to save prepared \
                                      dataset to.
    :return: Dataset baked for training the typos correction.
    """
    log = logging.getLogger("prepare_data")
    if config is None:
        config = {}
    config = merge_dicts(DEFAULT_CORRECTOR_CONFIG["preparation"], config)

    os.makedirs(config["data_dir"], exist_ok=True)
    raw_data_path = config["input_path"]
    if raw_data_path is None or not os.path.exists(raw_data_path):
        raw_data_path = os.path.join(config["data_dir"],
                                     config["raw_data_filename"])
        log.warning("raw dataset was not found, downloading from %s to %s",
                    config["dataset_url"], raw_data_path)
        _download_url(config["dataset_url"], raw_data_path)

    data = pandas.read_csv(raw_data_path, index_col=0, keep_default_na=False)
    log.debug("raw dataset shape: %s", data.shape)
    if config["frequency_column"] not in data.columns:
        log.info("frequency column is not found. Set all frequencies to 1")
        data[Columns.Frequency] = 1
    else:
        log.info("frequency column `%s` is found", config["frequency_column"])
        data = data.rename(columns={config["frequency_column"]: Columns.Frequency})

    # Expand dataframe by splits (repeat rows for every token in splits)
    data[Columns.Split] = data[Columns.Split].astype(str)
    log.debug("expand data by splits")
    flat_data = flatten_df_by_column(data, Columns.Split, Columns.Token,
                                     apply_function=lambda x: x.split())
    log.debug("expanded data shape %s", flat_data.shape)

    log.info("collect statistics for tokens")
    stats = flat_data[[Columns.Frequency, Columns.Token]].groupby([Columns.Token]).sum()
    stats = stats.sort_values(by=Columns.Frequency, ascending=False)[Columns.Frequency]

    log.info("save all frequencies")
    frequencies = stats.to_dict()
    log.info("tokens with frequencies data size: %d", len(frequencies))
    frequencies_filepath = os.path.join(config["data_dir"], config["frequencies_filename"])
    print_frequencies(frequencies, frequencies_filepath)
    log.info("tokens with frequencies data are saved to %s", frequencies_filepath)

    vocabulary = generate_vocabulary(frequencies_filepath, config["vocabulary"])
    log.info("vocabulary size: %d", len(vocabulary))
    vocabulary_filepath = os.path.join(config["data_dir"], config["vocabulary_filename"])
    print_frequencies(vocabulary, vocabulary_filepath)
    log.info("vocabulary saved to %s", vocabulary_filepath)

    # Leave only splits that contain tokens from vocabulary
    flat_data.reset_index(drop=True, inplace=True)
    log.info("final dataset shape: %s", flat_data.shape)
    if config["prepared_filename"] is not None:
        prepared_data_filepath = os.path.join(config["data_dir"], config["prepared_filename"])
        flat_data.to_csv(prepared_data_filepath)
        log.info("final dataset is saved to %s", prepared_data_filepath)
    return flat_data


def train_fasttext(data: pandas.DataFrame, config: Optional[Mapping[str, Any]] = None) -> None:
    """
    Train fasttext model on the given dataset of code identifiers.

    :param data: Dataframe with columns Columns.Split and Columns.Frequency.
    :param config: Parameters for training the model, options:
                   size: Number of identifiers to pick from the given data to train fasttext on.
                   corrupt: Value indicating whether to make random artificial typos in \
                            the training data. Identifiers are corrupted with `typo_probability`.
                   typo_probability: Token corruption probability if `corrupt == True`.
                   add_typo_probability: Probability of second corruption in a corrupted token. \
                                         used if `corrupt == True`.
                   path: Path where to store the trained fasttext model.
                   dim: Number of dimensions for embeddings in the new model.
                   bucket: Number of hash buckets to keep in the fasttext model: \
                           the less there are, the more compact the model gets.
                   adjust_frequencies: Whether to divide frequencies by the number of tokens in \
                                       the identifiers. Needs to be done when the result of the \
                                       `prepare` function is used as data to have a true \
                                       identifiers distribution.
    """
    try:
        import fastText
    except ImportError:
        sys.exit("Please install fastText."
                 "Run `pip3 install git+https://github.com/facebookresearch/fastText"
                 "@51e6738d734286251b6ad02e4fdbbcfe5b679382`")
    log = logging.getLogger("train_fasttext")
    if config is None:
        config = {}
    config = merge_dicts(DEFAULT_CORRECTOR_CONFIG["fasttext"], config)
    tokens_number = data[Columns.Split].apply(lambda x: len(str(x).split()))
    if config["adjust_frequencies"]:
        weights = data[Columns.Frequency] / tokens_number
    else:
        weights = data[Columns.Frequency]
    train_data = data[tokens_number > 1].sample(config["size"], weights=weights, replace=True)
    if config["corrupt"]:
        train_data = corrupt_tokens_in_df(train_data, config["typo_probability"],
                                          config["add_typo_probability"])
    with tempfile.NamedTemporaryFile() as ids_file:
        with open(ids_file.name, "w") as f:
            for token_split in train_data[Columns.Split]:
                f.write(token_split + "\n")
        log.info("Training fasttext model...")
        model = fastText.train_unsupervised(ids_file.name, minCount=1, epoch=10,
                                            dim=config["dim"],
                                            bucket=config["bucket"])
    model.save_model(config["path"])
    log.info("fasttext model is saved to %s", config["path"])


def get_datasets(prepared_data: pandas.DataFrame,
                 config: Optional[Mapping[str, Any]] = None,
                 ) -> Tuple[pandas.DataFrame, pandas.DataFrame]:
    """
    Create the train and the test datasets of typos.

    1. Take the specified number of lines from the input dataset.
    2. Make artificial typos in picked identifiers and split them into train and test.
    3. Return results.
    :param prepared_data: Dataframe of correct splitted identifiers. Must contain columns \
                          Columns.Split, Columns.Frequency and Columns.Token.
    :param config: Parameters for creating train and test datasets, options:
                   train_size: Train dataset size.
                   test_size: Test dataset size.
                   typo_probability: Probability of token corruption.
                   add_typo_probability: Probability of second corruption for a corrupted token.
                   train_path: Path to the .csv file where to save the train dataset.
                   test_path: Path to the .csv file where to save the test dataset.
                   processes_number: Number of processes for multiprocessing.
    :return: Train and test datasets.
    """
    log = logging.getLogger("get_datasets")
    if config is None:
        config = {}
    config = merge_dicts(DEFAULT_CORRECTOR_CONFIG["datasets"], config)

    # Use only the given number of the most frequent identifiers (less likely to contain typos)
    prepared_data = prepared_data[:config["portion"]]

    # With replace=True we get the real examples distribution, but there's a small
    # probability of having the same examples of misspellings in train and test datasets
    # (it IS small because a big number of random typos can be made in a single word)
    data = prepared_data[[len(x) > 1 for x in prepared_data[Columns.Token]]].sample(
        config["train_size"] + config["test_size"], weights=Columns.Frequency,
        replace=True)
    train, test = train_test_split(data[[Columns.Token, Columns.Split]],
                                   test_size=config["test_size"])
    train.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)
    log.info("train dataset shape: %s", train.shape)
    log.info("test dataset shape: %s", test.shape)
    train = corrupt_tokens_in_df(train, config["typo_probability"], config["add_typo_probability"],
                                 config["processes_number"])
    test = corrupt_tokens_in_df(test, config["typo_probability"], config["add_typo_probability"],
                                config["processes_number"])
    if config["test_path"] is not None:
        test.to_csv(config["test_path"])
        log.info("test dataset is saved to %s", config["test_path"])
    if config["train_path"] is not None:
        train.to_csv(config["train_path"])
        log.info("train dataset is saved to %s", config["train_path"])
    return train, test


def train_and_evaluate(train_data: pandas.DataFrame, test_data: pandas.DataFrame,
                       vocabulary_path: str, frequencies_path: str, fasttext_path: str,
                       generation_config: Optional[Mapping[str, Any]] = None,
                       ranking_config: Optional[Mapping[str, Any]] = None,
                       processes_number: int = DEFAULT_CORRECTOR_CONFIG["processes_number"],
                       ) -> TyposCorrector:
    """
    Create and train TyposCorrector model on the given data.

    :param train_data: Dataframe which contains columns Columns.Token, Columns.Split and \
                       Columns.CorrectToken.
    :param test_data: Dataframe which contains columns Columns.Token, Columns.Split and \
                      Columns.CorrectToken.
    :param vocabulary_path: Path to a file with vocabulary.
    :param frequencies_path: Path to a file with tokens' frequencies.
    :param fasttext_path: Path to a FastText model dump.
    :param generation_config: Candidates generation configuration.
    :param ranking_config: Ranking configuration.
    :param processes_number: Number of processes for multiprocessing.
    :return: Trained model.
    """
    model = TyposCorrector(ranking_config)
    model.initialize_generator(vocabulary_file=vocabulary_path,
                               frequencies_file=frequencies_path,
                               embeddings_file=fasttext_path, config=generation_config)
    model.processes_number = processes_number
    model.train(train_data)
    _, report = model.evaluate(test_data)
    print(report)
    return model


def train_from_scratch(config: Optional[Mapping[str, Any]] = None) -> TyposCorrector:
    """
    Train TyposCorrector on raw data.

    1. Prepare data, for more info check :func:`prepare_data`.
    2. Construct train and test datasets, for more info check :func:`get_train_test`.
    3. Train and evaluate TyposCorrector model, for more info check :func:`train_and_evaluate`.
    4. Return result.
    :param config: Parameters for data preparation and corrector training.
    :return: Trained TyposCorrector model.
    """
    log = logging.getLogger("train_from_scratch")
    if config is None:
        config = {}
    config = merge_dicts(DEFAULT_CORRECTOR_CONFIG, config)
    log.info("effective config:\n%s", pformat(config, width=120, compact=True))
    prepared_data = prepare_data(config["preparation"])
    if config["fasttext"]["path"] is None or not os.path.exists(config["fasttext"]["path"]):
        log.info("fasttext model is not found and will be trained")
        train_fasttext(prepared_data, config["fasttext"])
    train_data, test_data = get_datasets(prepared_data, config["datasets"])
    model = train_and_evaluate(train_data, test_data,
                               os.path.join(config["preparation"]["data_dir"],
                                            config["preparation"]["vocabulary_filename"]),
                               os.path.join(config["preparation"]["data_dir"],
                                            config["preparation"]["frequencies_filename"]),
                               config["fasttext"]["path"], config["generation"],
                               config["ranking"], config["processes_number"])
    if config["corrector_path"] is not None:
        model.save(config["corrector_path"], series=0.0)
        log.info("corrector model is saved to %s", config["corrector_path"])
    return model
