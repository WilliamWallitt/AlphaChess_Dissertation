import numpy as np
import h5py
import pickle
from tensorflow.python.keras.utils import data_utils
from Preprocessing.board_to_input import Board
import tensorflow as tf


class NetworkGenerator(tf.keras.utils.Sequence):

    def __init__(self, training_path, validation_path, training=True, training_size=64, raw_input=False):
        self.raw_input = raw_input
        self.mini_batch_size = training_size

        self.feature_planes = 12 if raw_input else 18

        if training:
            with open(training_path, 'r') as f:
                self.states = f.readlines()

            self.size = sum(1 for _ in self.states)
        else:

            with open(validation_path, 'r') as f:
                self.states = f.readlines()

            self.size = sum(1 for _ in self.states)

    def __getitem__(self, idx):

        states = [a.rstrip("\n") for a in self.states[idx * self.mini_batch_size: (idx + 1) * self.mini_batch_size]]
        X, y = [], []

        for game_state in states:
            if self.raw_input:
                state, label = Board(fen=game_state).get_input_image(image_size=8, feature_planes=self.feature_planes, raw_input=self.raw_input)
            else:
                state, label = Board(fen=game_state).get_input_image(image_size=8, feature_planes=self.feature_planes)
            X.append(state)
            y.append(int(label))

        X = np.reshape(X, (-1, 8, 8, self.feature_planes))
        y = np.reshape(y, (-1))

        return X, y

    def __len__(self):

        return int(np.ceil(self.size / self.mini_batch_size))