import tensorflow as tf
from Policy_Supervised_Learning.SL_policy_network import SLPolicyNetwork
from Policy_Reinforcement_Learning.self_play_game_generator import Self_Play
from Value_Supervised_Learning.SL_value_network import ValueNetwork
from AlphaChess import AlphaChess_Search_Methods
from Policy_Reinforcement_Learning.agent import run_tests


class AlphaChess:

    def __init__(self, policy_network=None, value_network=None):
        self.policy_network, self.value_network = policy_network, value_network

    def policy_network_supervised_learning(self, *, dataset_path="Data/human_chess_moves.pgn", preprocessing=False,
                                           batch_size=64, filters=256,
                                           feature_planes=18, epochs=15, raw_input=False,
                                           optimiser="sgd", path_prefix=""):

        self.__gpu_memory_allocation()

        SLPolicyNetwork(dataset_path=dataset_path, preprocessing=preprocessing, batch_size=batch_size,
                        filters=filters, feature_planes=feature_planes, epochs=epochs,
                        raw_input=raw_input, optimiser=optimiser, path_prefix=path_prefix).run_all_models()

    @staticmethod
    def policy_network_reinforcement_learning(generate_self_play_games=False, num_self_play_games=100000):
        # reinforcement learning stage uses multi-processing, we can't train the network using GPU
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        if not generate_self_play_games:
            # time to train the RL policy network
            run_tests()
            # reinforce()
        else:
            Self_Play(total_games=num_self_play_games, path_prefix="").play()

    def value_network_supervised_learning(self, dataset_path="Data/stockfish_games.pgn", num_games=294999,
                                          preprocessing=False, path_prefix="", epochs=15, shuffle=False,
                                          batch_size=64, filters=256, raw_input=False):

        SLValueNetwork = ValueNetwork(path_prefix=path_prefix, batch_size=batch_size, filters=filters)
        if preprocessing:
            SLValueNetwork.extract_dataset(path=dataset_path, games=num_games)
        if shuffle and not preprocessing:
            print("shuffling training and validation datasets")
            import random
            lines = open("Data/Regression_Learning/Training/training.txt").readlines()
            random.shuffle(lines)
            open("Data/Regression_Learning/Training/training.txt", "w").writelines(lines)
            lines = open("Data/Regression_Learning/Validation/validation.txt").readlines()
            random.shuffle(lines)
            open("Data/Regression_Learning/Validation/validation.txt", "w").writelines(lines)

        SLValueNetwork.train(epochs=epochs, raw_input=raw_input)

    def neural_network_mcts_vs_base_mcts(self, *, sl_policy_network=False, no_value_network=False, no_policy_network=False):
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        episode = AlphaChess_Search_Methods()
        episode.neural_network_mcts_vs_base_mcts(sl_policy_network=sl_policy_network, no_value_network=no_value_network, no_policy_network=no_policy_network)
        episode.add_game_info_to_file(file_path="Data/alpha_chess_results.txt")

    def minimax_vs_base_mcts(self):
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        episode = AlphaChess_Search_Methods()
        episode.minimax_vs_base_mcts()
        episode.add_game_info_to_file(file_path="Data/alpha_chess_results.txt")

    def neural_network_mcts_vs_human(self):
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        episode = AlphaChess_Search_Methods()
        episode.human_vs_neural_network_mcts()
        episode.add_game_info_to_file(file_path="Data/alpha_chess_results.txt")

    @staticmethod
    def __gpu_memory_allocation():
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                print(e)

# training of the SL policy network
AlphaChess().policy_network_supervised_learning(dataset_path="Data/human_chess_moves.pgn", preprocessing=False,
                                                batch_size=64, filters=256, epochs=15)

# training of the value network
AlphaChess().value_network_supervised_learning(preprocessing=False, batch_size=64, filters=256, epochs=15,
                                               raw_input=False)
# guard for the multiprocessing RL training
if __name__ == "__main__":

    # training of the RL policy network
    AlphaChess().policy_network_reinforcement_learning(generate_self_play_games=True, num_self_play_games=16)
    # self play games of the RL policy network
    AlphaChess().policy_network_reinforcement_learning()

    # Neural network MCTS vs standard MCTS
    AlphaChess().neural_network_mcts_vs_base_mcts(sl_policy_network=False, no_policy_network=False, no_value_network=False)
    # Neural network MCTS vs human player
    AlphaChess().neural_network_mcts_vs_human()
    # Neural network MCTS vs minimax search with value network
    AlphaChess().minimax_vs_base_mcts()


