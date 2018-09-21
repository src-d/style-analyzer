from typing import Iterable, Tuple
from gensim.models.fasttext import FastText
import sys
try:
    import keras
except ImportError:
    sys.exit("TF is required for this analyzer")
from keras.layers import Dense
from keras.models import Sequential
import numpy
import pandas

from lookout.style.typos.utils import collect_embeddings, CORRECT_TOKEN_COLUMN, TYPO_COLUMN


def get_features(fasttext: FastText, typos: Iterable[str]) -> numpy.ndarray:
    return numpy.concatenate((collect_embeddings(fasttext, typos),
                             numpy.ones((len(typos), 1))), axis=1)


def get_target(fasttext: FastText, identifiers: Iterable[str]) -> numpy.ndarray:
    return collect_embeddings(fasttext, identifiers)


def generator(features: numpy.ndarray, target: numpy.ndarray, batch_size: numpy.ndarray
              ) -> Tuple[numpy.ndarray, numpy.ndarray]:
    while True:
        indices = numpy.random.randint(features.shape[0], size=batch_size)
        batch_features = features[indices]
        batch_target = target[indices]
        yield batch_features, batch_target


def create_model(input_len: int, output_len: int) -> keras.models.Sequential:
    model = Sequential()
    model.add(Dense(units=256, activation="relu", input_dim=input_len))
    model.add(Dense(units=256, activation="relu"))
    model.add(Dense(units=output_len))
    return model


def train_model(model: keras.models.Sequential, features: numpy.ndarray, target: numpy.ndarray,
                save_model_file: str = None, batch_size: int = 64, lr: float = 0.1,
                decay: float = 1e-7, num_epochs: int = 100) -> keras.models.Sequential:
    model.compile(optimizer=keras.optimizers.SGD(lr=lr, decay=decay, momentum=0.9, nesterov=True),
                  loss="cosine_proximity")
    model.fit_generator(generator(features, target, batch_size=batch_size),
                        steps_per_epoch=len(features) // batch_size, epochs=num_epochs)

    if save_model_file is not None:
        model.save(save_model_file)
    return model


def create_and_train_nn_prediction(fasttext: FastText, data: pandas.DataFrame,
                                   save_model_file: str = None,
                                   batch_size: int = 64, lr: float = 0.1, decay: float = 1e-7,
                                   num_epochs: int = 100) -> keras.models.Sequential:
    """
    Train nn model for correction embedding prediction.
    :param fasttext: gensim.models.Fasttext model.
    :param data: DataFrame containing columns [CORRECT_TOKEN_COLUMN, TYPO_COLUMN].
    :param save_model_file: Path to file to dump trained nn model.
    :param batch_size: Batch size for training.
    :param lr: Learning rate.
    :param decay: Decay.
    :param num_epochs: Number of epochs.
    :return: Trained model.
    """
    typo_vecs = get_features(fasttext, data[TYPO_COLUMN])
    correction_vecs = get_target(fasttext, data[CORRECT_TOKEN_COLUMN])

    model = create_model(typo_vecs.shape[1], correction_vecs.shape[1])

    train_model(model, typo_vecs, correction_vecs, save_model_file, batch_size, lr, decay,
                num_epochs)
    if save_model_file is not None:
        model.save(save_model_file)
    return model


def get_predictions(fasttext: FastText, model: keras.models.Sequential, typos: Iterable[str]
                    ) -> numpy.ndarray:
    """
    Get predicted correction embeddings for tokens from typos
    :param fasttext: gensim.models.Fasttext model.
    :param model: Trained nn model
    :param typos: Iterable with tokens to check
    :return: Array of predicted correction embeddings
    """
    return model.predict(get_features(fasttext, typos))


def create_and_train_nn_prediction_from_file(fasttext_file: str, data_file: str,
                                             save_model_file: str = None, batch_size: int = 64,
                                             lr: float = 0.1, decay: float = 1e-7,
                                             num_epochs: int = 100) -> keras.models.Sequential:
    """
    Train nn model for correction embedding prediction from files.
    :param fasttext_file: Path to binary dump of fasttext model.
    :param data_file: Path to csv dump of pandas.DataFrame containing columns
                      [CORRECT_TOKEN_COLUMN, TYPO_COLUMN].
    :param save_model_file: Path to file to dump trained nn model.
    :param batch_size: Batch size for training.
    :param lr: Learning rate.
    :param decay: Decay.
    :param num_epochs: Number of epochs.
    :return: Trained model.
    """
    fasttext = FastText.load_fasttext_format(fasttext_file)
    data = pandas.read_csv(data_file, index_col=0).infer_objects()
    return create_and_train_nn_prediction(fasttext, data, save_model_file, batch_size, lr, decay,
                                          num_epochs)
