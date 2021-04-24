from Preprocessing.board_to_input import Board
import numpy as np
import h5py
import random


class HDF5Store(object):

    def __init__(self, datapath, dataset, shape, dtype=np.float32, compression="gzip", chunk_len=1):

        self.datapath = datapath
        self.dataset = dataset
        self.shape = shape
        self.i = 0

        with h5py.File(self.datapath, mode='w') as h5f:

            self.dset = h5f.create_dataset(
                dataset,
                shape=(0,) + shape,
                maxshape=(None,) + shape,
                dtype=dtype,
                compression=compression,
                chunks=(chunk_len,) + shape)

    def append(self, values):

        with h5py.File(self.datapath, mode='a') as h5f:
            dset = h5f[self.dataset]
            dset.resize((self.i + 1,) + self.shape)
            dset[self.i] = [values]
            self.i += 1
            h5f.flush()


class Image_Label_Extractor:

    def __init__(self, *, state_paths, action_paths, batch_size, split=(0.9, 0.1, 0.1)):

        self.state_paths = state_paths
        self.action_paths = action_paths
        self.states = []
        self.actions = []
        self.shape = (8, 8, 18)

        for s_p, a_p in zip(state_paths, action_paths):
            self.states.append(HDF5Store(s_p, 'X', shape=(8, 8, 18), chunk_len=batch_size))
            self.actions.append(open(a_p, 'a'))

        self.batch_size = batch_size
        self.dataset = open("Data/game_states.txt", "r").readlines()
        random.shuffle(self.dataset)
        self.file_length = sum(1 for _ in self.dataset)
        self.splits = split

    def extract(self, raw_input=False):

        training_split = int(self.file_length * self.splits[0])
        validation_split = int(self.file_length * self.splits[1]) + training_split

        for index, fen in enumerate(self.dataset):
            image, label = Board(fen=fen).get_input_image(image_size=8, feature_planes=18, raw_input=raw_input)
            if index % 100000 == 0 and index != 0:
                print("batch", index)
            if index < training_split:
                self.states[0].append(image)
                self.actions[0].write(label + "\n")
            elif training_split < index < validation_split:
                self.states[1].append(image)
                self.actions[1].write(label + "\n")
            else:
                self.states[2].append(image)
                self.actions[2].write(label + "\n")





