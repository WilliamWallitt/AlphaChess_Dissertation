import numpy as np


class Board:

    def __init__(self, *, fen):
        # takes in a converted fen from preprocessed dataset txt file
        self.fen = fen
        # remove new line from fen (as it from txt file)
        self.fen = self.fen.strip("\n")

    # used to convert the fen into either a 8x8x12 or 8x8x19 feature input image depending on user requirement
    def get_input_image(self, *, image_size, feature_planes, raw_input=False):

        # each piece is encoded as an 8x8 image (where each piece has an index in the 8x8xF feature input
        piece_index = {
            "p": 0,
            "n": 1,
            "b": 2,
            "r": 3,
            "q": 4,
            "k": 5,
            "P": 6,
            "N": 7,
            "B": 8,
            "R": 9,
            "Q": 10,
            "K": 11,
        }
        # split fen and convert to list
        fen = list(self.fen.split(" "))
        # initialise input feature image
        chess_board, row, col = [np.zeros((image_size, image_size)) for _ in range(feature_planes)], 0, 0
        # go through each square's value in the fen
        for square in self.fen:
            # if the square contains a piece
            if square in piece_index.keys():
                try:
                    # add the pieces position to it's input image as a 1
                    chess_board[piece_index[square]][row][col] = 1
                except IndexError as ex:
                    print(ex)
            # reached end of board fen representation
            if square == "/":
                row += 1
                col = 0
                continue
            # now we can get our label from the input fen string
            if square == " ":
                label = self.fen.partition(":")[2]
                break
            col += 1

        # if all features need to be encoded
        if not raw_input:
            # encode current playerd
            current_player = fen[1]
            current_player = np.full((image_size, image_size), int(current_player == 'w'), dtype=np.float)

            # encode castiling rights
            wK_castle = np.full((image_size, image_size), int('K' in fen[2]), dtype=np.float)
            wQ_castle = np.full((image_size, image_size), int('Q' in fen[2]), dtype=np.float)
            bK_castle = np.full((image_size, image_size), int('k' in fen[2]), dtype=np.float)
            bQ_castle = np.full((image_size, image_size), int('q' in fen[2]), dtype=np.float)
            # encode full move number
            move_number = fen[4]
            move_number = np.full((image_size, image_size), int(move_number), dtype=np.float)
            # create 8x8xF image stack
            planes = np.vstack((np.copy(chess_board[0]),
                                np.copy(chess_board[1]),
                                np.copy(chess_board[2]),
                                np.copy(chess_board[3]),
                                np.copy(chess_board[4]),
                                np.copy(chess_board[5]),
                                np.copy(chess_board[6]),
                                np.copy(chess_board[7]),
                                np.copy(chess_board[8]),
                                np.copy(chess_board[9]),
                                np.copy(chess_board[10]),
                                np.copy(chess_board[11]),
                                np.copy(current_player),
                                np.copy(move_number),
                                np.copy(wK_castle),
                                np.copy(wQ_castle),
                                np.copy(bK_castle),
                                np.copy(bQ_castle)
                                ))
        # otherwise we just encode the pieces (12x8x8 input image)
        else:
            planes = np.vstack((np.copy(chess_board[0]),
                                np.copy(chess_board[1]),
                                np.copy(chess_board[2]),
                                np.copy(chess_board[3]),
                                np.copy(chess_board[4]),
                                np.copy(chess_board[5]),
                                np.copy(chess_board[6]),
                                np.copy(chess_board[7]),
                                np.copy(chess_board[8]),
                                np.copy(chess_board[9]),
                                np.copy(chess_board[10]),
                                np.copy(chess_board[11])
                                ))
            feature_planes = 12

        # convert into neural network acceptable input
        planes = np.reshape(planes, (image_size, image_size, feature_planes))
        # return input and label
        return planes, label
