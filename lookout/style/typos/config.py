import multiprocessing
import pathlib


DEFAULT_DATA_DIR = pathlib.Path(__file__).parent / "data"
DEFAULT_CORRECTOR_CONFIG = {
    "preparation": {
        "data_dir": str(DEFAULT_DATA_DIR),
        "input_path": str(DEFAULT_DATA_DIR / "raw_data.csv"),
        "dataset_url": "https://docs.google.com/uc?export=download&"
                       "id=1muNVWPe68XK8SFvqIv3V728NmkT46aTx",
        "frequency_column": "num_occ",
        "vocabulary_size": 10000,
        "frequencies_size": None,
        "raw_data_filename": "raw_data.csv",
        "vocabulary_filename": "vocabulary.csv",
        "frequencies_filename": "frequencies.csv",
        "prepared_filename": "prepared.csv",
    },
    "fasttext": {
        "size": 100000000,  # Number of identifiers to pick to train fasttext on
        "corrupt": True,  # Whether to corrupt some of the identifiers with artificial typos
        "typo_probability": 0.2,  # Which portion of picked identifiers contain a typoed token
        "add_typo_probability": 0.005,  # Which portion of corrupted tokens contain >1 mistake
        "path": str(DEFAULT_DATA_DIR / "fasttext.bin"),  # Where to store trained fasttext model
        "dim": 8,  # Number of dimensions of embeddings
        "bucket": 200000,  # Number of hash buckets in the model
    },
    "datasets": {
        "train_size": 50000,
        "test_size": 10000,
        "typo_probability": 0.5,
        "add_typo_probability": 0.01,
        "train_path": str(DEFAULT_DATA_DIR / "train.csv"),
        "test_path": str(DEFAULT_DATA_DIR / "test.csv"),
    },
    "generation": {
        "radius": 3,
        "max_distance": 2,
        "neighbors_number": 0,
        "edit_dist_number": 20,
        "max_corrected_length": 12,
        "start_pool_size": 64,
        "chunksize": 256,
    },
    "ranking": {
        "train_rounds": 4000,
        "early_stopping": 200,
        "boost_param": {
            "max_depth": 6,
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
