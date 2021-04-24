import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import random
from tensorflow.keras.models import load_model
from Policy_Reinforcement_Learning.game import Game
import numpy as np
from concurrent.futures.process import ProcessPoolExecutor
from concurrent.futures._base import as_completed
from multiprocessing import Process
import tensorflow_probability.python.distributions as dist

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


global_optimizor, global_standardised, global_batch_size = 'sgd', False, 32
global_discounted = False
global_expectation = False
global_text_file_name = ""
global_learning_rate = 0.000001


class Agent(object):

    # 0.000001 = lr
    def __init__(self, data=None, learning_rate=0.000001, gamma=0.9, path_prefix=""):

        self.path_prefix = path_prefix + "Data/Reinforcement_Learning/"
        # variables passed through by process
        self.optimizor = data[0]
        self.standardised = data[1]
        self.discounted_rewards = data[3]
        self.expectation = data[4]
        self.file_name = data[5]
        # _____________________________

        self.supervised_model_path = self.path_prefix + "Network/Model_2"
        self.supervised_weights_path = self.path_prefix + "Network_Weights/"
        self.reinforcement_weights_path = self.path_prefix + "Reinforcement_Network/" + self.file_name + "/"
        if not os.path.isdir(self.reinforcement_weights_path):
            os.mkdir(self.reinforcement_weights_path)

        self.learning_rate = learning_rate
        self.gamma = gamma
        self.curr_policy = self.load_current_policy()
        self.previous_policy = self.load_previous_policy()
        # TEST STUFF
        with open(self.path_prefix + "Labels/labels.txt", 'r') as f:
            labels = f.read()
            self.label_size = labels.split(" ")
            self.label_size = sum([1 for _ in self.label_size])

    def load_current_policy(self):
        try:
            reinforcement_weights = os.listdir(self.reinforcement_weights_path)
            for item in reinforcement_weights:
                if not item.startswith('.'):
                    # print("Loading reinforcement learning weights -> ", self.reinforcement_weights_path + item)
                    return tf.keras.models.load_model(self.reinforcement_weights_path + "my_model")
            supervised_weights = os.listdir(self.supervised_weights_path)[-1]
            # print("Loading supervised learning weights -> ", supervised_weights)
            return self.load_model(self.supervised_model_path, self.supervised_weights_path + supervised_weights)

        except OSError as e:
            print("Error loading model and weights -> ", e)

    def load_previous_policy(self):
        try:
            supervised_weights = os.listdir(self.supervised_weights_path)
            del supervised_weights[-1]
            random_previous_weight = random.sample(supervised_weights, 1)[0]
            opponent_weights = self.supervised_weights_path + random_previous_weight
            # print("Loading previous policy weights -> ", random_previous_weight)
            return self.load_model(self.supervised_model_path, opponent_weights)

        except (OSError, ValueError) as e:
            print("Error loading model and weights -> ", e)

    def load_model(self, model_path, weights_path):
        model = load_model(model_path)
        model.load_weights(weights_path)
        # need loss for non gradient tape updates
        # loss='categorical_crossentropy',
        # learning_rate = self.learning_rate
        if self.optimizor == "adagrad":
            model.compile(optimizer=tf.optimizers.Adadelta())
            # model.compile(optimizer=tf.optimizers.SGD(learning_rate=self.learning_rate, momentum=0.9, nesterov=True))
        else:
            model.compile(optimizer=tf.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def play_game(self):
        # self.previous_policy = self.load_previous_policy()
        return Game(policy_curr=self.curr_policy, policy_prev=self.previous_policy, expectation=self.expectation, discounted=self.discounted_rewards).run()

    def train_gradient_tape(self, results):

        states, probabilities, rewards, actions = results[0], results[1], results[2], results[3]
        # actual_rewards = rewards
        rewards = np.array(rewards)
        if self.standardised:
            rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-7)

        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        print("Wins/Loss/Draw -> ", results[4])
        print("______________________")
        print("Average Rewards", np.mean(results[5]))
        print("______________________")
        print("Episode length", len(rewards))

        with tf.GradientTape() as tape:
            loss = 0
            for idx, (g, state) in enumerate(zip(rewards, states)):
                # create input of current state
                state = tf.convert_to_tensor([state], dtype=tf.float32)
                # get the policy probability distribution for that state
                probs = self.curr_policy(state)
                # use tensorflow_probability to log the probabilities
                action_probs = dist.Categorical(probs=probs)
                # get the action taken log probability
                log_prob = action_probs.log_prob(actions[idx])
                # loss is the discounted reward * that log probability
                loss += -g * tf.squeeze(log_prob)
        print("Total loss ->", loss.numpy())
        with open(self.path_prefix + "Results/"+self.file_name+".txt", 'a') as f:
            results[4] = [str(x) for x in results[4]]
            f.write(str(np.mean(results[5])) + "," + ",".join(results[4]) + "," + str(len(rewards)) + "," + str(loss.numpy()) + "\n")
        # get the current networks gradient with respect to the loss
        gradient = tape.gradient(loss, self.curr_policy.trainable_variables)
        # adjust the network in the direction of the gradient
        self.curr_policy.optimizer.apply_gradients(zip(gradient, self.curr_policy.trainable_variables))
        self.curr_policy.save(self.reinforcement_weights_path + "my_model")


def return_func(f, arr):
    return f(arr)


def run_episode(arr):
    return Agent(data=arr).play_game()


def run_cpu_tasks_in_parallel(arr, workers=2):
    episodes = []
    batch_size = arr[2]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_episodes = {executor.submit(return_func, run_episode, arr): _ for _ in range(batch_size)}
        for episode in as_completed(future_episodes):
            try:
                data = episode.result()
                episodes.append(data)
            except Exception as ex:
                print(ex)

        states, gradients, scores, probs, results = [], [], [], [], []
        # gradient tape
        actions = []
        actual_scores = []
        wins = draws = loss = 0
        for game in episodes:

            states.extend(game[0])
            scores.extend(game[1])
            probs.extend(game[2])
            actions.extend(game[4])
            actual_scores.extend(game[5])
            if game[3] == 1:
                wins += 1
            elif game[3] == -1:
                loss += 1
            else:
                draws += 1

        Agent(data=arr).train_gradient_tape([states, probs, scores, actions, [wins, loss, draws], actual_scores])


def reinforce(*, num_episodes=1):
    for i in range(num_episodes):
        print("Batch -> " + str(i) + " started")
        data = [global_optimizor, global_standardised, global_batch_size, global_discounted, global_expectation, global_text_file_name]
        p = Process(target=return_func, args=(run_cpu_tasks_in_parallel, data))
        p.start()
        p.join()


# optimizor, standardised, batch_size, discouted_rewards, expectation, file name

tests = [
    ["adagrad", True, 2, False, True,
     "32_batch_adagrad_expectation_standardised"]
]


def run_tests():
    for test in tests:
        # update global variables for each test -> pass through to Agent()
        global global_optimizor
        global global_standardised
        global global_batch_size
        global global_discounted
        global global_expectation
        global global_text_file_name
        global_optimizor = test[0]
        global_standardised = test[1]
        global_batch_size = test[2]
        global_discounted = test[3]
        global_expectation = test[4]
        global_text_file_name = test[5]

        reinforce()

#
# if __name__ == "__main__":
#     run_tests()


import matplotlib.pyplot as plt

# d = tests[0: 10]
# d = []
# d.append(tests[1])
# d.append(tests[6])

# paths = ["../Policy_Reinforcement_Learning/Results/" + t[5] + ".txt" for t in d]
#
# x, scores = [], []
# # legend = ["all", "discounted", "standardised", "none", "score all", "adam all", "adam discounted", "adam standardised", "none"]
# for path in paths:
#     with open(path, 'r') as f:
#         x_axis = []
#         y_scores = []
#         for index, line in enumerate(f.readlines()):
#             info = line.split(",")
#             score = float(info[0])
#             x_axis.append(index)
#             y_scores.append(score)
#
#         plt.plot(x_axis, y_scores)
#
# # plt.legend(legend, loc="best")
# plt.show()
#





import matplotlib.pyplot as plt

# x, y = [], []
#
# win, loss, draw = [], [], []
# game_length = []
# actual_loss = []
#
# with open("Results/v2.txt") as f:
#     for i, line in enumerate(f.readlines()):
#         line = line.split(",")
#         x.append(float(line[0]))
#         y.append(i)
#         win.append(float(line[1]))
#         loss.append(float(line[2]))
#         draw.append(float(line[3]))
#         game_length.append(float(line[4]))
#         actual_loss.append(float(line[5]))
#
# def best_fit(X, Y):
#     xbar = sum(X)/len(X)
#     ybar = sum(Y)/len(Y)
#     n = len(X) # or len(Y)
#
#     numer = sum([xi*yi for xi, yi in zip(X, Y)]) - n * xbar * ybar
#     denum = sum([xi**2 for xi in X]) - n * xbar**2
#
#     b = numer / denum
#     a = ybar - b * xbar
#
#     print('best fit line:\ny = {:.2f} + {:.2f}x'.format(a, b))
#
#     return a, b
# a, b = best_fit(y, x)
# plt.scatter(y, x, s=10)
# yfit = [a + b * xi for xi in y]
# plt.plot(y, yfit)
# plt.plot(y, x)
# plt.show()

