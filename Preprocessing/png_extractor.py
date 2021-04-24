import chess.pgn
import pickle


class PGN_extractor:

    def __init__(self, *, file_path, num_games, labels_path, dataset_path):
        # where to store the unique labels found
        self.labels_path = labels_path
        # where to load in the PGN file
        self.dataset_path = dataset_path
        # to store all extracted board states
        self.board_states = []
        # number of games to extract state, actions from
        self.num_games = num_games
        try:
            # try to open PGN file and load it
            pgn_file = open(file_path)
            self.pgn_file = pgn_file
        except FileNotFoundError:
            raise Exception("File not found")

    def __extract_labels(self):
        # to store all unique labels found
        unique_labels = []
        # for game number in range number of games required
        for i in range(self.num_games):
            try:
                # read whole game from PGN file
                game = chess.pgn.read_game(self.pgn_file)
            except UnicodeDecodeError:
                # checking if PGN representation is correct
                print("UnicodeDecodeError - skipping")
                continue
            # check if we are at the end of the PGN file
            if game is None:
                break
            # get start of game
            node = game.root()
            # just to see where we are in extracting the games
            if i % 500 == 0:
                print("Batch: " + str(i) + " is done")
            # until we reach the end of the current game
            while not node.is_end():
                # get child variations
                next_node = node.variation(0)
                # get the current move's label in algebraic notation
                label = node.board().san(next_node.move)
                # add the FEN notation of the board state to self.board_states
                self.board_states.append(node.board().fen() + ":" + label + "\n")
                # if label is unique, add to unique labels
                if label not in unique_labels:
                    unique_labels.append(label)
                # move to next move in game
                node = next_node
        # close PGN file
        self.pgn_file.close()
        # store number of unique labels
        with open(self.labels_path + "/hyper_params.pk", 'wb') as fi:
            pickle.dump(len(unique_labels), fi)
        labels_text_file = open(self.labels_path + "/labels.txt", "w")
        labels_text_file.write(' '.join([str(label) for label in unique_labels]))
        labels_text_file.close()
        print("Saved to " + self.labels_path + "/labels.txt")

    def __extract_board_state(self):
        # we want all empty squares to have a 1
        replace_numbers = {
            "2": "1"*2,
            "3": "1"*3,
            "4": "1"*4,
            "5": "1"*5,
            "6": "1"*6,
            "7": "1"*7,
            "8": "1"*8
        }
        # to store preprocessed fens
        board_fen = []
        # for each fen in the extracted fen states
        for state in self.board_states:
            # split string
            str_state = state.split(" ")
            # get board encoding in the fen string
            str_board = str_state[0]
            # change encoding for empty squares
            for k, v in replace_numbers.items():
                str_board = str_board.replace(k, v)
            # add new encoding back into the fen string
            for i, field in enumerate(str_state):
                if i > 0:
                    str_board += " " + field
            board_fen.append(str_board)

        # save converted fen strings
        game_state_text_file = open(self.dataset_path + "game_states.txt", "w")
        game_state_text_file.write(''.join([str(label) for label in board_fen]))
        game_state_text_file.close()

    # to extract both fen strings and actions
    def extract_labels_and_board_state(self):
        self.__extract_labels()
        self.__extract_board_state()





