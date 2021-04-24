from Minimax_Search.Minimax_Search import Minimax
import chess


class AlphaChess_Minimax:

    def __init__(self, board=None, depth=1, is_max=True, path_prefix="../", value_network=None):
        self.board = chess.Board() if board is None else board
        self.depth = depth
        self.is_max = is_max
        self.player_turn = True
        self.path_prefix = path_prefix
        self.value_network = value_network

    def start_game(self):
        while not self.board.is_game_over():

            if self.player_turn:
                self.__get_user_uci_move()
            else:
                _ = self.black_game_step()

    def black_game_step(self, not_human=True):

        # print("ALPHA_CHESS_MINIMAX is thinking.....")
        # print(self.board)
        move = Minimax(self.depth, self.board, self.is_max, value_network=self.value_network).get_move()

        x = [chess.Board(fen=self.board.fen()).san(mmove) for mmove in self.board.legal_moves]
        y = [chess.Board(fen=self.board.fen()).uci(mmove) for mmove in self.board.legal_moves]
        index = y.index(move.uci())
        m = x[index]
        if not not_human:
            self.board.push(chess.Move.from_uci(str(move)))
        self.player_turn = not self.player_turn
        # print(self.board)
        # print("ALPHA_CHESS_MINIMAX makes move: ", move)

        return m

    def __get_user_uci_move(self):

        legal_moves = [self.board.san(m) for m in self.board.legal_moves]
        print(legal_moves)
        index_move = input("Enter index of move")
        try:
            index_move = int(index_move)
            move = legal_moves[index_move]
            self.board.push_san(move)
            self.player_turn = not self.player_turn

        except ValueError as val_err:
            print("Error ->", val_err)




