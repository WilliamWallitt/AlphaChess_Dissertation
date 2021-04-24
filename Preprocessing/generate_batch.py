import numpy as np
import tensorflow.keras
import pickle
from Preprocessing.board_to_input import Board
import linecache

# tensorflow uses a custom sequence class for batch updates
class Generator(tensorflow.keras.utils.Sequence):

    def __init__(self, *, batch_size, validation=False, path_prefix="../", raw_input=False):
        # path to dataset
        self.path = path_prefix + "Data/Supervised_Learning"
        # if samples generated either are encoded as raw input or full feature image stack
        self.raw_input = raw_input
        # default image size and feature planes
        self.image_size = 8
        self.feature_planes = 18
        # if raw input is required, change number of feature planes
        if raw_input:
            self.feature_planes = 12
        # open unique labels file store as variable
        with open(self.path + "/Labels/labels.txt", 'r') as f:
            labels = f.read()
            self.labels = labels.split(" ")
        # open hyper_params file (containing length of unique lables)
        with open(self.path + "/Labels/hyper_params.pk", 'rb') as fi:
            self.label_size = pickle.load(fi)
        # if validation dataset required
        if validation:
            # set fen_path to the validation dataset
            self.fen_path = self.path + '/Validation/validation_states.txt'
            # get size of dataset
            self.size = sum(1 for _ in open(self.path + '/Validation/validation_states.txt'))
        # otherwise training dataset required
        else:
            # set fen_path to the training dataset
            self.fen_path = self.path + '/Training/training_states.txt'
            # get size of dataset
            self.size = sum(1 for _ in open(self.path + '/Training/training_states.txt'))
        # store minibatch size
        self.mini_batch = batch_size

    def __len__(self):
        # so keras knows what the number of steps to train on is
        return int(np.ceil(self.size / self.mini_batch)) - 2

    def __getitem__(self, idx):
        # this is used to return the mini-batch of samples
        # get current index range of dataset to load in
        indices = [i for i in range(idx * self.mini_batch, (idx + 1) * self.mini_batch)]
        # to store fen string from indices
        fen_states = []
        # get the fen string from the file and store in arr
        for index in indices:
            fen_state = linecache.getline(self.fen_path, index + 1)
            fen_states.append(fen_state)
        # remove all new lines from lines
        fen_states = [a.rstrip("\n") for a in fen_states]
        # to store state, action(label) of batch
        states, actions = [], []
        # go through fen strings
        for fen in fen_states:
            # convert to netwrok input and label
            state, action = Board(fen=fen).get_input_image(image_size=self.image_size, feature_planes=self.feature_planes, raw_input=self.raw_input)
            # one hot encoding for label
            index = self.labels.index(action)
            one_hot = np.zeros(len(self.labels))
            one_hot[index] = 1
            actions.append(one_hot)
            states.append(state)
        # reshape so that keras accepts the input image
        X = np.reshape(states, (-1, 8, 8, self.feature_planes))
        y = np.reshape(actions, (-1, self.label_size))
        # return actions and labels
        return X, y




