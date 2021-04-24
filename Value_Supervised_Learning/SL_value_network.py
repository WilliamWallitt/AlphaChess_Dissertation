import tensorflow.keras as keras
from tensorflow.keras import layers
import os
from Value_Supervised_Learning.pgn_extractor import PGN_extractor
from Value_Supervised_Learning.network_generator import NetworkGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)


class ValueNetwork:

    def __init__(self, path_prefix, filters=256, batch_size=64):
        self.path_prefix = path_prefix
        self.path = self.path_prefix + "Data/Regression_Learning/"
        self.training_path = self.path_prefix + "Data/Regression_Learning/Training/"
        self.validation_path = self.path_prefix + "Data/Regression_Learning/Validation/"

        self.model_path = self.path + "Regression_Learning_Models/"
        self.model_checkpoints = self.path + "Regression_Learning_Checkpoints/"
        # yes i could iterate
        if not os.path.isdir(self.path):
            os.mkdir(self.path)
        if not os.path.isdir(self.training_path):
            os.mkdir(self.training_path)
        if not os.path.isdir(self.validation_path):
            os.mkdir(self.validation_path)
        if not os.path.isdir(self.model_path):
            os.mkdir(self.model_path)
        if not os.path.isdir(self.model_checkpoints):
            os.mkdir(self.model_checkpoints)

        self.filters = filters
        self.batch_size = batch_size

    def extract_dataset(self, path=None, games=300000):
        if path is None:
            path = self.path_prefix + "Data/human_chess_moves.pgn"
        else:
            path = self.path_prefix + path
        PGN_extractor(file_path=path, num_games=games).extract_labels_and_board_state()

    @staticmethod
    def generator(path, training=True, batch_size=64, raw_input=False):
        if training:
            return NetworkGenerator(training=training, training_path=path, validation_path=None,
                                    training_size=batch_size, raw_input=raw_input)
        else:
            return NetworkGenerator(training=training, validation_path=path, training_path=None,
                                    training_size=batch_size, raw_input=raw_input)

    def train(self, epochs, raw_input=False):

        if raw_input:
            features = 12
        else:
            features = 18

        model = keras.models.Sequential()
        # add stuff
        model.add(
            layers.Conv2D(self.filters, (5, 5), (1, 1), padding="SAME", activation=keras.activations.relu,
                          data_format="channels_last",
                          input_shape=(8, 8, features)))
        # layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding="SAME")
        # model.add(layers.Conv2D(128, (5, 5), (1, 1), padding="SAME", activation=keras.activations.relu))
        model.add(layers.Conv2D(self.filters, (3, 3), (1, 1), padding="SAME", activation=keras.activations.relu))
        model.add(layers.Conv2D(self.filters, (3, 3), (1, 1), padding="SAME", activation=keras.activations.relu))
        model.add(layers.Conv2D(self.filters, (1, 1), (1, 1), padding="SAME", activation=keras.activations.relu))
        model.add(layers.Flatten())
        # model.add(layers.Dense(units=256, activation=keras.activations.relu))
        # model.add(layers.Dense(units=128, activation=keras.activations.relu))
        # model.add(layers.Dense(units=64, activation=keras.activations.relu))
        model.add(layers.Dense(units=1, activation=keras.activations.tanh))
        model.compile(loss=keras.losses.mean_squared_error, optimizer=keras.optimizers.SGD(), metrics=['accuracy'])
        model.save(self.model_path)

        training_generator = self.generator(training=True, path=self.training_path + 'training.txt',
                                            batch_size=self.batch_size, raw_input=raw_input)
        validation_generator = self.generator(training=False, path=self.validation_path + 'validation.txt',
                                              batch_size=self.batch_size, raw_input=raw_input)
        training_steps, validation_steps = training_generator.__len__(), validation_generator.__len__()

        filepath = self.model_checkpoints + "weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor="val_accuracy", verbose=1, save_best_only=True, mode="max")
        callbacks_list = [checkpoint]

        history = model.fit(training_generator, epochs=epochs, verbose=1, callbacks=callbacks_list, steps_per_epoch=training_steps, shuffle=True,
                            validation_data=validation_generator, validation_steps=validation_steps)

        self.plot_metrics(history, "Accuracy", "Epoch", ["accuracy", "val_accuracy"], "Model Accuracy",
                      ["Training", "Testing"])
        self.plot_metrics(history, "Loss", "Epoch", ["loss", "val_loss"], "Model Loss",
                      ["Training", "Testing"])

    @staticmethod
    def plot_metrics(history, x_lab, y_lab, metric_x, title, legend):
        plt.plot(history.history[metric_x[0]])
        plt.plot(history.history[metric_x[1]])
        plt.title(title)
        plt.xlabel(x_lab)
        plt.ylabel(y_lab)
        plt.legend(legend, loc="upper left")
        plt.show()
