import chess


class GameState(object):

    def __init__(self):
        self.board = chess.Board()
        # chess colour, True is white False if black
        self.current_player = 1 if self.board.turn else -1
        self.is_end_of_game = self.board.is_game_over()
        self.legal_moves_cache = None

    def do_move(self, action):

        colour = self.current_player
        self.current_player = colour

        if self.is_legal(action):
            self.board.push_uci(action)
            self.legal_moves_cache = None
            self.current_player = -colour

        else:
            # return True
            self.current_player = colour
            # return self.board.is_game_over()

        # check for end of game
        self.is_end_of_game = self.board.is_game_over()

        return self.is_end_of_game

    def is_legal(self, action):
        legal_actions = [m.uci() for m in self.board.legal_moves]
        if action in legal_actions:
            return True
        else:
            return False

    def get_legal_moves(self):

        if self.legal_moves_cache is not None:

            return self.legal_moves_cache

        self.legal_moves_cache = []

        moves = [m.uci() for m in self.board.legal_moves if self.is_legal(m.uci())]
        self.legal_moves_cache = moves

        return self.get_legal_moves()

    def get_winner(self):
        result = self.board.result()
        if result == "1-0":
            return 1
        elif result == "0-1":
            return -1
        else:
            return 0

    def get_winner_color(self):
        return self.board.turn

    def get_current_player(self):
        return 1 if self.current_player else -1

    def copy(self):

        c = GameState()
        c.board = self.board
        c.current_player = 1 if self.board.turn else -1
        c.is_end_of_game = self.board.is_game_over()

        return c
