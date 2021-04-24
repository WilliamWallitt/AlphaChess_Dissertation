from Policy_Reinforcement_Learning.agent import Agent
import os


class Self_Play:

    def __init__(self, total_games=64, path_prefix="../"):

        self.total_number_of_games = total_games
        self.batch_size = 16
        self.batch_size = total_games if total_games < self.batch_size else self.batch_size
        self.dataset_path = path_prefix + "Data/Reinforcement_Learning/Self_Play_Games/"
        if not os.path.isdir(self.dataset_path):
            os.mkdir(self.dataset_path)
        self.ds_path = "RL_self_play_games"
        self.self_play_store = None
        self.path_prefix = path_prefix

        try:
            self.self_play_store = open(self.dataset_path + "self_play_games.txt", 'a')

        except OSError as ex:
            print("Error -> ", ex)

    def __run_episode(self):
        return Agent(data=["adagrad", True, 2, False, True, ""]).play_game()

    def __generate_batch_and_store(self) -> None:

        # with open(self.rewards_path, 'a') as f:
        for episode in range(self.batch_size):
            res = self.__run_episode()
            # for each fen move in episode
            for fen, result in zip(res[-1][0], res[-1][1]):
                state_action_pair = fen + ":" + str(result) + "\n"
                self.self_play_store.write(state_action_pair)

    def play(self):
        total_batches = int(self.total_number_of_games / self.batch_size)
        for i in range(total_batches):
            self.__generate_batch_and_store()

