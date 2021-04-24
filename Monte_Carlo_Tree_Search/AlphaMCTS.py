from collections import namedtuple
from random import choice
from Monte_Carlo_Tree_Search.MCTS import MCTS
from Monte_Carlo_Tree_Search.Node import Node
from collections import namedtuple
from random import choice

import chess


_TTTB = namedtuple("ChessBoard", "fen turn winner terminal")


class ChessBoard(_TTTB, Node):

    def find_children(self):
        if self.terminal:
            return set()

        moves = [chess.Board(self.fen).san(m) for m in chess.Board(self.fen).legal_moves]

        return {
            self.make_move(i) for i in range(len(moves))
        }

    def find_random_child(self, policy_selection=False):
        if self.terminal:
            return None  # If the game is finished then no moves can be made
        legal_moves = []

        for move in chess.Board(self.fen).legal_moves:
            legal_moves.append(chess.Board(self.fen).san(move))

        # if policy selection is true - sample from policy probability mapping to legal moves
        # this is used for MCTS rollout, dont return a random move

        return self.make_move(legal_moves.index(choice(legal_moves)))

    def reward(self):
        # returns an int
        if not self.terminal:
            raise RuntimeError(f"reward called on non-terminal board {self}")
        if self.winner is self.turn:
            # It's your turn and you've already won. Should be impossible.
            raise RuntimeError(f"reward called on unreachable board {self}")
        if self.turn is (not self.winner):
            return 0    # Your opponent has just won. Bad. 0
        if self.winner is None:
            return 0.5  # Board is a tie 0.5
        # The winner is neither True, False, nor None
        raise RuntimeError(f"board has unknown winner type {self.winner}")

    def is_terminal(self):
        # returns boolean
        return self.terminal

    def make_move(self, index):

        moves = [chess.Board(self.fen).san(m) for m in chess.Board(self.fen).legal_moves]
        tup = chess.Board(self.fen)
        tup.push_san(moves[index])

        is_terminal = False
        winner = None

        if tup.is_game_over() and tup.result() == "1-0":
            # print("White wins")
            winner = True
            is_terminal = True
        elif tup.is_game_over() and tup.result() == "0-1":
            # print("Black wins")
            winner = False
            is_terminal = True
        elif tup.is_game_over() and tup.result() == "1/2-1/2":
            # print("Draw")
            is_terminal = True
            winner = None

        turn = not self.turn

        return ChessBoard(tup.fen(), turn, winner, is_terminal)


class AlphaChess_MCTS:

    def __init__(self, board=None, rollouts=5, turn=True, winner=None, terminal=False):
        self.board = ChessBoard(chess.Board().fen() if not board else board.fen(),
                                turn=turn,
                                winner=winner,
                                terminal=terminal)

        self.rollouts = rollouts
        self.search_tree = MCTS()
        # self.board.print_board()

    def run_game(self):
        while not self.board.terminal:
            # print(self.board)
            self.white_game_step()
            _ = self.black_game_step()

    def black_game_step(self):

        # print(self.board, "ai")

        x = chess.Board(self.board.fen)

        legal_moves = [x.san(m) for m in x.legal_moves]

        possible_states = [chess.Board(x.fen()) for _ in range(len(legal_moves))]
        _ = [x.push_san(legal_moves[index]) for index, x in enumerate(possible_states)]

        for _ in range(self.rollouts):
            self.search_tree.do_rollout(self.board)
        self.board = self.search_tree.choose(self.board)

        move = chess.Board(self.board.fen)
        for index, state in enumerate(possible_states):
            if state == move:
                move_made = legal_moves[index]

        return move_made

    def white_game_step(self):

        # print(self.board, "me")
        index = AlphaChess_MCTS.get_user_index_move()
        board = self.board.make_move(index=index)
        self.board = board
        # self.board.print_board()

    @staticmethod
    def get_user_index_move():
        index_move = input("Enter index of move")
        try:
            index_move = int(index_move)
            return index_move
        except ValueError as val_err:
            print("Error ->", val_err)


