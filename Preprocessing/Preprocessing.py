from Preprocessing.png_extractor import PGN_extractor
from Preprocessing.generate_batch import Generator
from Preprocessing.training_validation_split import split_dataset
import pickle
import os


# helper class to make all preprocessing easily accessible
class Preprocessing:

    def __init__(self, *, dataset_path, local_path=None):
        # creating required directories for storage
        self.local_path = local_path if local_path is not None else ""
        path = "Data/Supervised_Learning/" if local_path is None else local_path + "Data/Supervised_Learning/"
        self.path = path
        sub_paths = ["Training", "Validation", "Labels"]
        self.data_paths = [path + sub_path for sub_path in sub_paths]
        self.dataset_path = dataset_path

        try:
            if not os.path.isdir(path):
                os.mkdir(path)
            for sub_path in sub_paths:
                if not os.path.isdir(path + sub_path):
                    os.mkdir(path + sub_path)
                else:
                    print(path + sub_path + " directory already exists")
            else:
                print(path + " directory already exits")
        except OSError:
            print("Creating of data directory failed")
        else:
            print("Creating of data directory successful")

    # convert PGN to FEN helper function
    def pgn_to_fen(self, *, number_of_games):
        extractor = PGN_extractor(file_path=self.dataset_path,
                                  num_games=number_of_games,
                                  labels_path=self.data_paths[-1], dataset_path=self.path)
        extractor.extract_labels_and_board_state()

    # split dataset into training and validation datasets
    def fen_to_training_validation_datasets(self):
        split_dataset(path_prefix=self.local_path)

    # get current length of unique labels (for one hot encoding)
    @staticmethod
    def get_unique_labels_length(local_path=""):
        with open(local_path + "Data/Supervised_Learning/Labels/hyper_params.pk", 'rb') as f:
            label_size = pickle.load(f)
        return label_size

    # get total number of training and validation samples
    def get_total_number_of_training_and_validation_games(self):
        training_len = sum(1 for _ in open(self.local_path + "Data/Supervised_Learning/Training/training_states.txt"))
        validation_len = sum(1 for _ in open(self.local_path + "Data/Supervised_Learning/Validation/validation_states.txt"))
        return training_len, validation_len


