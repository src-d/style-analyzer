import pandas

from lookout.style.typos.corrector import TyposCorrector
from lookout.style.typos.utils import COLUMNS, flatten_df_by_column
from lookout.style.typos.preprocessing import (filter_splitted_identifiers, pick_subset_of_df,
                                               corrupt_tokens_in_df, print_frequencies, train_test_split)
from lookout.style.typos.preprocessing.metrics import print_scores


INPUT_FILE = "100k_repos2ids.csv"
VOCABULARY_FILE = "vocabulary_frequencies.csv"
FREQUENCIES_FILE = "frequencies.csv"
EMBEDDINGS_FILE = "lookout/style/typos/tests/id_vecs_10.bin"

FREQUENCY_COLUMN = "num_occ"
VOCABULARY_SIZE = 10000
FREQUENCIES_DATA_SIZE = 200000
TEST_SIZE = 10000
TRAIN_SIZE = 50000
THREADS_NUMBER = 8
SAVE_MODEL_PATH = "typos_corrector.asdf"


def train_corrector():
    data = pandas.read_csv(INPUT_FILE)

    # Splitting identifiers on tokens and expanding dataframe by
    flat_data = flatten_df_by_column(data, COLUMNS["SPLIT"], COLUMNS["TOKEN"],
                                     apply_function=lambda x: str(x).split())

    stats = flat_data[[FREQUENCY_COLUMN, COLUMNS["TOKEN"]]].groupby([COLUMNS["TOKEN"]]).sum()
    stats = stats.sort_values(by=[FREQUENCY_COLUMN], ascending=False)

    # Selecting vocabulary tokens - most frequent tokens in data
    vocabulary_tokens = set(stats.loc[stats.index[:VOCABULARY_SIZE]])
    filtered = filter_splitted_identifiers(flat_data, vocabulary_tokens)

    # Saving frequencies info for future use in the corrector model
    print_frequencies(vocabulary_tokens, stats, VOCABULARY_FILE)
    print_frequencies(set(stats.loc[stats.index[:FREQUENCIES_DATA_SIZE]]), stats, FREQUENCIES_FILE)

    # Leaving only necessary amount of data
    total = TRAIN_SIZE + TEST_SIZE
    active_data = pick_subset_of_df(filtered, size=total, weight_column=FREQUENCY_COLUMN)
    cor_active_data = corrupt_tokens_in_df(active_data, 0.5, 0.01)

    train, test = train_test_split(cor_active_data, TEST_SIZE / total)

    # Training TyposCorrector model on obtained data
    model = TyposCorrector()
    model.initialize_ranker()
    model.initialize_generator(vocabulary_file=VOCABULARY_FILE,
                               frequencies_file=FREQUENCIES_FILE,
                               embeddings_file=EMBEDDINGS_FILE)
    model.threads_number = THREADS_NUMBER
    model.train(train)

    model.save(SAVE_MODEL_PATH)

    # Printing result on test part
    if TEST_SIZE > 0:
        suggestions_test = model.suggest(test)
        print_scores(test, suggestions_test)

    return model
