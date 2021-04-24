import numpy as np


def split_dataset(path_prefix=""):
    # splitting fen moves dataset into training and testing datasets
    path = path_prefix + 'Data/Supervised_Learning'

    fen_moves_len = sum(1 for _ in open(path + '/game_states.txt'))
    training_len = int(np.ceil(fen_moves_len * 0.7))

    training = open(path + "/Training/training_states.txt", 'w')
    validation = open(path + "/Validation/validation_states.txt", 'w')
    with open(path + "/game_states.txt", 'r') as f:
        for i, fen in enumerate(f):
            if i <= training_len:
                training.write(fen)
            else:
                validation.write(fen)

    training.close()
    validation.close()
