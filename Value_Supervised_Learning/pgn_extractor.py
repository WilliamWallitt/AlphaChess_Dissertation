import chess.pgn
import pickle
import numpy as np


class PGN_extractor:

    def __init__(self, *, file_path, num_games, path_prefix=""):
        self.path_prefix = path_prefix + "Data/Regression_Learning/"
        self.board_states = []
        self.num_games = num_games
        self.dataset_paths = [self.path_prefix + "Training/training.txt", self.path_prefix + "Validation/validation.txt"]
        try:
            pgn_file = open(file_path)
            self.pgn_file = pgn_file
        except FileNotFoundError:
            raise Exception("File not found")

    def __extract_labels(self):
        for i in range(self.num_games):
            try:
                game = chess.pgn.read_game(self.pgn_file)
                game_pgn_result = game.headers["Result"]
                result = self.game_result(game_pgn_result)

            except UnicodeDecodeError:
                print("UnicodeDecodeError - skipping")
                continue
            if game is None:
                break
            node = game.root()
            if i % 500 == 0:
                print("Batch: " + str(i) + " is done")

            while not node.is_end():
                next_node = node.variation(0)
                self.board_states.append(node.board().fen() + ":" + str(result) + "\n")
                node = next_node

            if i > (self.num_games * 0.7):
                self.__extract_board_state(1)
                self.board_states = []
            else:
                self.__extract_board_state(0)
                self.board_states = []

        self.pgn_file.close()

    @staticmethod
    def game_result(result):
        if result == "0-1":
            return -1
        elif result == "1-0":
            return +1
        else:
            return 0

    def __extract_board_state(self, index=0):

        replace_numbers = {
            "2": "1" * 2,
            "3": "1" * 3,
            "4": "1" * 4,
            "5": "1" * 5,
            "6": "1" * 6,
            "7": "1" * 7,
            "8": "1" * 8
        }

        board_fen = []

        for state in self.board_states:
            str_state = state.split(" ")
            str_board = str_state[0]
            for k, v in replace_numbers.items():
                str_board = str_board.replace(k, v)
            for i, field in enumerate(str_state):
                if i > 0:
                    str_board += " " + field
            board_fen.append(str_board)

        # state_length = len(board_fen)
        # training_split = [0, int(np.ceil(state_length * 0.7))]
        # validation_split = [training_split[1], int(np.ceil(state_length * 0.9))]
        # testing_split = [validation_split[1], int(np.ceil(state_length))]

        split = self.dataset_paths[index]

        with open(split, "a") as f:
            f.write(''.join([str(label) for label in board_fen]))

        # for path, split in zip(self.dataset_paths, [training_split, validation_split, testing_split]):
        #     with open(path, "w") as f:
        #         f.write(''.join([str(label) for label in board_fen[split[0]:split[1]]]))

    def extract_labels_and_board_state(self):
        self.__extract_labels()
        # self.__extract_board_state()

