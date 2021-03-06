import chess
from CPuct_MCTS.GameState import GameState
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
from Preprocessing.board_to_input import Board
from CPuct_MCTS.MCTS import MCTS
from chess.engine import Mate
import chess.engine


def analyse_move(fen, side, stockfish_path="Networks/stockfish_13_win_x64_bmi2/stockfish_13_win_x64_bmi2.exe",
                 expectation=True):
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    info = engine.analyse(fen, chess.engine.Limit(depth=17))
    if side:
        if not expectation:
            if info["score"].is_mate():
                y = info["score"].white().mate()
                y = Mate(y).score(mate_score=10000)
                x = y
            else:
                x = info["score"].white().score()
        else:
            x = (2 * info["score"].white().wdl().expectation()) - 1
    else:
        if not expectation:
            if info["score"].is_mate():
                y = info["score"].black().mate()
                y = Mate(y).score(mate_score=10000)
                x = y
            else:
                x = info["score"].black().score()
        else:
            x = (2 * info["score"].black().wdl().expectation()) - 1
    engine.quit()
    return x


def value_fn(state):
    # value = 2 * random.random() - 1
    return 0.0


value_network = load_model("Networks/Value_Network/Regression_Learning_Models/")
value_network.load_weights("Networks/Value_Network/Regression_Learning_Checkpoints/weights-improvement-15-0.62.hdf5")


def actual_value_fn(state, minimax=False):
    if not minimax:
        fen_state = extract_board_state(state.board.fen())
    else:
        fen_state = extract_board_state(state.fen())
    s, _ = Board(fen=fen_state).get_input_image(image_size=8, feature_planes=18, raw_input=False)
    s = tf.convert_to_tensor([s], dtype=tf.float32)

    v = value_network.predict(s).flatten()
    return v[0]


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



SL_policy_network = load_model("Networks/Policy_Network/Supervised_Learning_Model/Model_2")
SL_policy_network.load_weights("Networks/Policy_Network/"
                               "Supervised_Learning_Model/Weights/weights-improvement-29-0.40.hdf5")

policy_network = tf.keras.models.load_model("Networks/Policy_Network/my_model")

with open("Networks/Policy_Network/labels.txt", 'r') as f:
    labels = f.read()
    labels = labels.split(" ")


def actual_poliy_fn(state):

    fen_state = extract_board_state(state.board.fen())

    s, _ = Board(fen=fen_state).get_input_image(image_size=8, feature_planes=18, raw_input=False)

    s = tf.convert_to_tensor([s], dtype=tf.float32)

    probabilities = policy_network.predict(s).flatten()
    probabilities = probabilities / np.sum(probabilities)

    current_legal_actions = [state.board.san(m) for m in state.board.legal_moves]

    prob_dist_test = []
    # for each legal action
    for san in current_legal_actions:
        try:
            # get index of action
            i = labels.index(san)
            # get probability of action
            prob_dist_test.append(probabilities[i])
        except ValueError as ex:
            prob_dist_test.append(0.5)

    tup = [(x, y) for x, y in zip(state.get_legal_moves(), prob_dist_test)]

    if state.board.is_game_over():
        return []

    return tup


def SL_policy_fn(state):

    fen_state = extract_board_state(state.board.fen())

    s, _ = Board(fen=fen_state).get_input_image(image_size=8, feature_planes=18, raw_input=False)

    s = tf.convert_to_tensor([s], dtype=tf.float32)

    probabilities = SL_policy_network.predict(s).flatten()
    probabilities = probabilities / np.sum(probabilities)

    current_legal_actions = [state.board.san(m) for m in state.board.legal_moves]

    prob_dist_test = []
    # for each legal action
    for san in current_legal_actions:
        try:
            # get index of action
            i = labels.index(san)
            # get probability of action
            prob_dist_test.append(probabilities[i])
        except ValueError as ex:
            prob_dist_test.append(0.5)

    tup = [(x, y) for x, y in zip(state.get_legal_moves(), prob_dist_test)]

    if state.board.is_game_over():
        return []

    return tup


def policy_fn(state):

    legal_moves = state.get_legal_moves()

    dist = np.random.dirichlet(np.ones(len(legal_moves)), size=1).flatten()
    dist = list(dist)
    # dist = [0.5 for _ in range(len(legal_moves))]

    tup = [(x, y) for x, y in zip(legal_moves, dist)]
    return tup


rollout_policy = policy_fn
dummy_policy = policy_fn

actual_rollout_policy = actual_poliy_fn
actual_dummy_policy = actual_poliy_fn
