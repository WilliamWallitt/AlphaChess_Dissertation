import chess
from Preprocessing.board_to_input import Board
import numpy as np
import tensorflow as tf
import tensorflow_probability.python.distributions as tfp
from Policy_Reinforcement_Learning.Stockfish import evaluate_position
import asyncio
import chess.engine
# from Policy_Reinforcement_Learning.Stockfish import play_move


class Game:

    def __init__(self, policy_curr, policy_prev, expectation=True, discounted=True, path_prefix=""):
        self.path_prefix = path_prefix + "Data/Reinforcement_Learning/"
        self.board = chess.Board()
        self.curr_policy, self.prev_policy = policy_curr, policy_prev
        self.player_move = True
        self.fen_moves, self.game_results = [], None
        self.same_reward_format = []

        with open(self.path_prefix + "Labels/labels.txt", 'r') as f:
            labels = f.read()
            self.labels = labels.split(" ")

        self.black_scores = []

        self.state_memory, self.action_memory = [], []
        self.probability_distributions = []
        self.action_indexes = []
        self.scores = []

        self.expectation, self.discounted = expectation, discounted

    def play_turn(self):

        fen = self.board.fen()
        fen = self.extract_board_state(fen)
        self.fen_moves.append(fen)
        state, _ = Board(fen=fen).get_input_image(image_size=8, feature_planes=18, raw_input=False)
        temp = state
        state = tf.convert_to_tensor([state], dtype=tf.float32)

        if self.player_move:
            predicted_actions = self.curr_policy.predict(state).flatten()
            # .flatten()
            self.probability_distributions.append(predicted_actions)

            self.state_memory.append(temp)
            # prob
            predicted_actions = predicted_actions / np.sum(predicted_actions)

        else:
            predicted_actions = self.prev_policy.predict(state).flatten()
            # self.probability_distributions.append(predicted_actions)
            predicted_actions = predicted_actions / np.sum(predicted_actions)

        current_legal_actions = self.get_legal_moves()
        legal_labels = [1 if label in current_legal_actions else 0 for label in self.labels]
        made_move = self.check_if_valid_and_make_move(
            predicted_actions=predicted_actions,
            legal_labels=legal_labels, legal_moves=current_legal_actions)

        if not made_move:
            return False
        else:
            return True

    def check_if_valid_and_make_move(self, *, predicted_actions, legal_labels, legal_moves):

        legal_distribution = predicted_actions * legal_labels
        action_probs = tfp.Categorical(probs=legal_distribution)

        while True:

            try:
                # get index
                action_index = action_probs.sample().numpy()
                action = self.labels[action_index]
                if action in legal_moves:

                    self.board.push_san(action)

                    if self.player_move:

                        asyncio.set_event_loop_policy(chess.engine.EventLoopPolicy())
                        expectation, score = asyncio.run(evaluate_position(self.board, path_prefix=self.path_prefix))
                        if self.expectation:
                            self.scores.append(expectation)
                        else:
                            self.scores.append(score)
                        # all tests will use raw score from stockfish, for reward plotting
                        self.same_reward_format.append(score)
                        self.action_indexes.append(action_index)

                        # print(self.board)
                        # print(self.board.fen())
                        # print(score)
                        # print(self.board.result())
                        # print("__________")

                    self.player_move = not self.player_move

                    return True

            except (ValueError, IndexError) as e:
                print("Error ->", e)
                return False

    def get_fen_position(self):
        return self.board.fen()

    def get_legal_moves(self):
        moves = []
        for move in self.board.legal_moves:
            moves.append(self.board.san(move))
        return moves

    def get_result(self):
        return self.board.result()

    @staticmethod
    def discounted_rewards(rewards):
        G = np.zeros_like(rewards)
        for t in range(len(rewards)):
            G_sum = 0
            discount = 1
            for k in range(t, len(rewards)):
                G_sum += rewards[k] * discount
                discount *= 0.7
            G[t] = G_sum

        return G

    @staticmethod
    def extract_board_state(state):

        replace_numbers = {
            "2": "1"*2,
            "3": "1"*3,
            "4": "1"*4,
            "5": "1"*5,
            "6": "1"*6,
            "7": "1"*7,
            "8": "1"*8
        }

        str_state = state.split(" ")
        str_board = str_state[0]
        for k, v in replace_numbers.items():
            str_board = str_board.replace(k, v)
        for i, field in enumerate(str_state):
            if i > 0:
                str_board += " " + field

        return str_board

    def run(self):
        is_legal = True
        while not self.board.is_game_over(claim_draw=False):
            is_legal = self.play_turn()
            if not is_legal:
                break

        result = self.get_result()
        zt_white_move = 0

        self.game_results = [0 for i in range(len(self.fen_moves))]

        if result == "1-0":
            # print("white wins")
            zt_white_move = 1
            self.game_results = [1 if i % 2 == 0 else 0 for i in range(len(self.fen_moves))]
        elif result == "0-1":
            # print("black wins")
            zt_white_move = -1
            self.game_results = [0 if i % 2 == 0 else 1 for i in range(len(self.fen_moves))]

        if not is_legal or result == "*":
            zt_white_move = -2
            del self.state_memory[-1]
            del self.probability_distributions[-1]

        if self.discounted:
            self.scores = self.discounted_rewards(np.array(self.scores))

        return [self.state_memory, self.scores, self.probability_distributions, zt_white_move, self.action_indexes, self.same_reward_format, [self.fen_moves, self.game_results]]


