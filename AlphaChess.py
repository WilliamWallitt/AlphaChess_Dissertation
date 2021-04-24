import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from Monte_Carlo_Tree_Search.AlphaMCTS import AlphaChess_MCTS
from Minimax_Search.AlphaChess import AlphaChess_Minimax
from CPuct_MCTS.AlphaChessMCTS import actual_value_fn
from CPuct_MCTS.AlphaChessMCTS import actual_rollout_policy
from CPuct_MCTS.AlphaChessMCTS import actual_dummy_policy
from CPuct_MCTS.AlphaChessMCTS import analyse_move
from CPuct_MCTS.AlphaChessMCTS import policy_fn
from CPuct_MCTS.AlphaChessMCTS import SL_policy_fn
from CPuct_MCTS.GameState import GameState
from CPuct_MCTS.MCTS import MCTS
import chess
from stockfish import Stockfish
import matplotlib.pyplot as plt
import numpy as np

dummy_policy, dummy_rollout = policy_fn, policy_fn
rollout = SL_policy_fn


class AlphaChess_Search_Methods:

    def __init__(self):

        self.board = chess.Board()
        self.white_eval, self.black_eval = [], []
        self.stockfish = Stockfish("Networks/stockfish_13_win_x64_bmi2/stockfish_13_win_x64_bmi2.exe")

        self.MCTS = None
        self.gs = GameState()

        self.opp_MCTS = None
        self.opp_gs = GameState()

    @staticmethod
    def get_user_index_move():
        index_move = input("Enter index of move, starting from 0: ")
        try:
            index_move = int(index_move)
            return index_move
        except ValueError as val_err:
            print("Error ->", val_err)

    def user_opponent(self):
        legal_moves = [x.uci() for x in self.board.legal_moves]
        print("Moves -> ", legal_moves)
        index = self.get_user_index_move()
        self.board.push_uci(legal_moves[index])
        return legal_moves[index]

    def minimax_search(self, *, search_depth=2, is_max=True):
        engine = AlphaChess_Minimax(path_prefix="", value_network=actual_value_fn, is_max=is_max,
                                    board=self.board,
                                    depth=search_depth)
        move = engine.black_game_step(not_human=True)
        m = chess.Board(fen=self.board.fen()).parse_san(move)
        m = m.uci()
        self.board.push_san(move)
        return m

    def init_cpuct_mcts(self, rollouts=100, playout_depth=20, rollout_lim=500, policy=actual_dummy_policy,
                        rollout=actual_rollout_policy, value=actual_value_fn, lmbda=0.5):

        self.MCTS = MCTS(value, policy, rollout,
                         n_playout=rollouts,
                         lmbda=lmbda,
                         rollout_limit=rollout_lim,
                         playout_depth=playout_depth)

    def c_puct_mcts(self):
        move = self.MCTS.get_move(self.gs)
        self.board.push_uci(move)
        self.gs.board = chess.Board(self.board.fen())
        self.gs.is_end_of_game = self.board.is_game_over()
        self.gs.current_player = 1 if self.board.turn else -1
        self.MCTS.update_with_move(move)
        return move

    def update_cpuct_search_tree(self, uci):
        self.gs.board = chess.Board(self.board.fen())
        self.gs.is_end_of_game = self.board.is_game_over()
        self.gs.current_player = 1 if self.board.turn else -1
        self.MCTS.update_with_move(uci)

    def base_mcts(self, *, rollouts=10, white=False):
        engine = AlphaChess_MCTS(board=chess.Board(fen=chess.Board(fen=self.board.fen()).fen()),
                                 rollouts=rollouts,
                                 turn=white, winner=None, terminal=False)
        move = engine.black_game_step()

        m = chess.Board(fen=self.board.fen()).parse_san(move)
        m = m.uci()
        self.board.push_san(move)
        return m

    def minimax_vs_base_mcts(self):
        while True:
            print(self.board.fen())
            if self.board.is_game_over():
                break
            _ = self.minimax_search()
            self.white_eval.append(self.get_state_evaluation(white=True))

            if self.board.is_game_over():
                break

            _ = self.base_mcts()
            self.black_eval.append(self.get_state_evaluation(white=False))

    def neural_network_mcts_vs_base_mcts(self, *, sl_policy_network=False, no_value_network=False, no_policy_network=False):

        if no_policy_network:
            p = dummy_policy
            r = dummy_rollout
        elif sl_policy_network:
            p = SL_policy_fn
            r = rollout
        else:
            p = actual_dummy_policy
            r = actual_rollout_policy

        if not no_value_network:
            self.init_cpuct_mcts(lmbda=1, policy=p, rollout=r)
        else:
            self.init_cpuct_mcts(lmbda=0.5, policy=p, rollout=r)

        while True:
            print(self.board.fen())
            if self.board.is_game_over():
                break
            _ = self.c_puct_mcts()
            self.white_eval.append(self.get_state_evaluation(white=True))

            if self.board.is_game_over():
                break

            move = self.base_mcts()
            self.black_eval.append(self.get_state_evaluation(white=False))

            self.update_cpuct_search_tree(move)

    def human_vs_neural_network_mcts(self):

        self.init_cpuct_mcts(lmbda=0.5, policy=actual_dummy_policy, rollout=actual_rollout_policy)

        while True:

            if self.board.is_game_over():
                break
            _ = self.c_puct_mcts()
            self.white_eval.append(self.get_state_evaluation(white=True))
            print(self.board)

            if self.board.is_game_over():
                break

            move = self.user_opponent()
            print(self.board)

            self.black_eval.append(self.get_state_evaluation(white=False))

            self.update_cpuct_search_tree(move)

    def get_state_evaluation(self, white=True, expectation=True):
        return analyse_move(self.board, white, expectation=expectation)

    def stockfish_player(self, skill_level=None, depth=None, elo=None):
        self.stockfish.set_fen_position(self.board.fen())
        if skill_level:
            self.stockfish.set_skill_level(skill_level)
        if depth:
            self.stockfish.set_depth(depth)
        if elo:
            self.stockfish.set_elo_rating(elo)
        move = self.stockfish.get_best_move()
        self.board.push_uci(move)

    def plot_game_results(self):
        x_white = [i for i in range(len(self.white_eval))]
        x_black = [i for i in range(len(self.black_eval))]

        plt.plot(x_white, self.white_eval)
        plt.plot(x_black, self.black_eval)
        plt.title("Game prediction scores for each player's move")
        plt.xlabel("Move number")
        plt.ylabel("Scores")
        plt.show()

    def add_game_info_to_file(self, file_path):
        with open(file_path, 'a') as file:
            game_length = len(self.white_eval) + len(self.black_eval)
            file.write(self.board.result() + "," + str(np.mean(self.white_eval)) +
                       "," + str(np.mean(self.black_eval)) + "," + str(game_length) + "\n")

