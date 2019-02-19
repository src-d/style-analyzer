import sys
from typing import Iterable, Sequence, Tuple

from gensim.models.fasttext import FastText

try:
    import keras
except ImportError:
    sys.exit("TensorFlow is required for this analyzer")
from keras.layers import Dense, BatchNormalization
from keras.models import Sequential
import numpy
import pandas

from lookout.style.typos.utils import Columns


def extract_embeddings_from_fasttext(fasttext: FastText, tokens: Iterable[str]) -> numpy.ndarray:
    """
    Convert the embeddings from FastText to a dense matrix.

    :param fasttext: trained embeddings.
    :param tokens: list of tokens - axis Y of the returned matrix.
    :return: matrix with extracted embeddings.
    """
    return numpy.array([fasttext.wv[token] for token in tokens])


def get_features(fasttext: FastText, typos: Sequence[str]) -> numpy.ndarray:
    return numpy.concatenate((extract_embeddings_from_fasttext(fasttext, typos),
                             numpy.ones((len(typos), 1))), axis=1)


def get_target(fasttext: FastText, identifiers: Iterable[str]) -> numpy.ndarray:
    return extract_embeddings_from_fasttext(fasttext, identifiers)


def generator(features: numpy.ndarray, target: numpy.ndarray, batch_size: numpy.ndarray
              ) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """
    Pumps the data for keras.Model.fit_generator()

    :param features: Inputs.
    :param target: Labels.
    :param batch_size: Batch size.
    :return: Another batch for fit_generator().
    """

    while True:
        indices = numpy.random.randint(features.shape[0], size=batch_size)
        batch_features = features[indices]
        batch_target = target[indices]
        yield batch_features, batch_target


def create_model(num_neurons: int, input_len: int, output_len: int) -> keras.models.Sequential:
    """
    Builds the fully-connected NN.

    :param num_neurons: Number of neurons in each hidden layer.
    :param input_len: Input size.
    :param output_len: Output size.
    :return: Built model.
    """
    model = Sequential()
    model.add(Dense(units=num_neurons, input_dim=input_len, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dense(units=num_neurons, activation="relu"))
    model.add(BatchNormalization())
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


DEFAULT_NUM_NEURONS = 256
DEFAULT_BATCH_SIZE = 64
DEFAULT_LR = 0.1
DEFAULT_DECAY = 0.9
DEFAULT_NUM_EPOCHS = 10


def create_and_train_nn_prediction(
        fasttext: FastText, data: pandas.DataFrame, saved_model_file: str,
        num_neurons: int = DEFAULT_NUM_NEURONS, batch_size: int = DEFAULT_BATCH_SIZE,
        lr: float = DEFAULT_LR, decay: float = DEFAULT_DECAY, num_epochs: int = DEFAULT_NUM_EPOCHS
        ) -> keras.models.Sequential:
    """
    Train NN model for correction embedding prediction.

    :param fasttext: gensim.models.Fasttext model.
    :param data: DataFrame containing columns [Columns.CorrectToken, Columns.Token].
    :param saved_model_file: Path to file to dump trained NN model.
    :param num_neurons: Number of neurons in each hidden layer.
    :param batch_size: Batch size for training.
    :param lr: Learning rate.
    :param decay: Learning rate exponential decay per epoch.
    :param num_epochs: Number of passes over the train dataset.
    :return: Trained Keras model.
    """
    typo_vecs = get_features(fasttext, data[Columns.Token])
    correction_vecs = get_target(fasttext, data[Columns.CorrectToken])
    model = create_model(num_neurons, typo_vecs.shape[1], correction_vecs.shape[1])
    train_model(
        model, typo_vecs, correction_vecs, saved_model_file, batch_size, lr, decay, num_epochs)
    if saved_model_file is not None:
        model.save(saved_model_file)
    return model


def get_predictions(fasttext: FastText, model: keras.models.Sequential, typos: Iterable[str]
                    ) -> numpy.ndarray:
    """
    Get predicted correction embeddings for tokens from typos.

    :param fasttext: gensim.models.FastText model.
    :param model: Trained NN model.
    :param typos: Iterable with tokens to check.
    :return: Array of predicted correction embeddings.
    """
    return model.predict(get_features(fasttext, typos))


def create_and_train_nn_prediction_from_file(
        fasttext: str, data: str, dump: str = None, num_neurons: int = DEFAULT_NUM_NEURONS,
        batch_size: int = DEFAULT_BATCH_SIZE, lr: float = DEFAULT_LR, decay: float = DEFAULT_DECAY,
        num_epochs: int = DEFAULT_NUM_EPOCHS) -> keras.models.Sequential:
    """
    Train NN model for correction embedding prediction from files.

    :param fasttext: Path to the binary dump of a FastText model.
    :param data: Path to a CSV dump of pandas.DataFrame containing columns \
                 [Columns.CorrectToken, Columns.Token].
    :param dump: Path to the file where to dump the trained NN model.
    :param num_neurons: Number of neurons in each hidden layer.
    :param batch_size: Batch size for training.
    :param lr: Learning rate.
    :param decay: Learning rate exponential decay per epoch.
    :param num_epochs: Number of training passes over the dataset.
    :return: Trained Keras model.
    """
    fasttext_model = FastText.load_fasttext_format(fasttext)
    df = pandas.read_csv(data, index_col=0).infer_objects()
    return create_and_train_nn_prediction(
        fasttext_model, df, dump, num_neurons, batch_size, lr, decay, num_epochs)
