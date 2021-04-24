from collections import defaultdict
import math


class MCTS:

    def __init__(self, exploration_weight=0.5):
        self.Q = defaultdict(int)  # total reward of each node
        self.N = defaultdict(int)  # total visit count for each node
        self.P = defaultdict(int)  # prior probabilities for each node from the policy network
        self.children = dict()  # children of each node
        self.exploration_weight = exploration_weight

    def choose(self, node):
        if node.is_terminal():
            raise RuntimeError(f"choose called on terminal node {node}")

        if node not in self.children:
            return node.find_random_child()

        def score(n):
            if self.N[n] == 0:
                return float("-inf")  # avoid unseen moves
            return self.Q[n] / self.N[n]  # average reward

        return max(self.children[node], key=score)

    def do_rollout(self, node):
        path = self._select(node)
        leaf = path[-1]
        self._expand(leaf)
        reward = self._simulate(leaf)
        self._backpropagate(path, reward)

    def _select(self, node):
        path = []
        while True:
            path.append(node)
            if node not in self.children or not self.children[node]:
                # node is either unexplored or terminal
                return path
            unexplored = self.children[node] - self.children.keys()
            if unexplored:
                n = unexplored.pop()
                path.append(n)
                return path

            node = self._uct_select(node)  # descend a layer deeper

    def _expand(self, node):
        if node in self.children:
            return  # already expanded
        self.children[node] = node.find_children()

    @staticmethod
    def _simulate(node):
        invert_reward = True
        while True:
            if node.is_terminal():
                reward = node.reward()
                return 1 - reward if invert_reward else reward
            node = node.find_random_child()
            invert_reward = not invert_reward

    def _backpropagate(self, path, reward):
        # I think we need to get the parent node, use RL network to get policy
        # then find the node probability for that policy and store it in self.P[node] = prob
        for node in reversed(path):
            # print(node, reward)
            self.N[node] += 1
            # add V(SL) here
            self.Q[node] += reward
            # add self.P[node] = prob_node
            reward = 1 - reward  # 1 for me is 0 for my enemy, and vice versa

    def _uct_select(self, node):

        # All children of node should already be expanded:
        assert all(n in self.children for n in self.children[node])

        log_N_vertex = math.log(self.N[node])

        def uct(n):
            "Upper confidence bound for trees"
            return self.Q[n] / self.N[n] + self.exploration_weight * math.sqrt(
                log_N_vertex / self.N[n]
            )

        return max(self.children[node], key=uct)



