import multiprocessing
import pathlib


DEFAULT_DATA_DIR = pathlib.Path(__file__).parent / "data"
DEFAULT_CORRECTOR_CONFIG = {
    "preparation": {
        "data_dir": str(DEFAULT_DATA_DIR),
        "input_path": str(DEFAULT_DATA_DIR / "raw_data.csv"),
        "dataset_url": "https://docs.google.com/uc?export=download&"
                       "id=1tedGTGacNYVZ1hzMIN-Xo1thtT5oDR5M",
        "frequency_column": "num_occ",
        "vocabulary": {
            "stable": 6000,
            "suspicious": 2500,
            "non_suspicious": 3000,
        },
        "raw_data_filename": "raw_data.csv",
        "vocabulary_filename": "vocabulary.csv",
        "frequencies_filename": "frequencies.csv",
        "prepared_filename": "prepared.csv",
    },
    "fasttext": {
        "size": 500000000,  # Number of identifiers to pick to train fasttext on
        "corrupt": False,  # Whether to corrupt some of the identifiers with artificial typos
        "typo_probability": 0.2,  # Which portion of picked identifiers contain a typoed token
        "add_typo_probability": 0.005,  # Which portion of corrupted tokens contain >1 mistake
        "path": str(DEFAULT_DATA_DIR / "fasttext.bin"),  # Where to store trained fasttext model
        "dim": 10,  # Number of dimensions of embeddings
        "bucket": 200000,  # Number of hash buckets in the model
        "adjust_frequencies": True,  # Whether to divide identifiers frequencies by tokens number.
    },
    "datasets": {
        "portion": 400000,
        "train_size": 100000,
        "test_size": 10000,
        "typo_probability": 0.5,
        "add_typo_probability": 0.01,
        "train_path": str(DEFAULT_DATA_DIR / "train.csv"),
        "test_path": str(DEFAULT_DATA_DIR / "test.csv"),
        "processes_number": multiprocessing.cpu_count(),
    },
    "generation": {
        "radius": 3,
        "max_distance": 2,
        "neighbors_number": 0,
        "edit_dist_number": 4,
        "max_corrected_length": 30,
        "start_pool_size": 256,
        "chunksize": 256,
        "set_min_freq": False,
    },
    "ranking": {
        "train_rounds": 1000,
        "early_stopping": 100,
        "verbose_eval": False,
        "boost_param": {
            "max_depth": 5,
            "eta": 0.03,
            "min_child_weight": 2,
            "silent": 1,
            "objective": "binary:logistic",
            "subsample": 0.5,
            "colsample_bytree": 0.5,
            "alpha": 1,
            "eval_metric": ["error"],
            "nthread": multiprocessing.cpu_count(),
        },
    },
    "processes_number": multiprocessing.cpu_count(),
    "corrector_path": str(DEFAULT_DATA_DIR / "corrector.asdf"),
}
