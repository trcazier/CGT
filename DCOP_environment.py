import itertools
import random
import numpy as np

from Graph import generate_random_graph

CONSTANT = 10


class DCOPGame:
    """
    DCOP Game environment.
    """

    def __init__(self, num_agents, num_actions, deterministic=True, only_binary=True):
        self.num_agents = num_agents
        self.num_actions = num_actions
        self.only_binary = only_binary
        if deterministic:
            self.initialize_weights_and_rewards_deterministic()
        else:
            self.initialize_weights_and_rewards_random()

    def initialize_weights_and_rewards_deterministic(self):
        """
        This function is used to initialize the weights and rewards for each
        action of both agents connected by an edge in the graph.
        The variant implemented in this function is the graph in fig. 5 of the paper.
        another function can be created and used in __init__ instead if another
        scenario is being tested.

        The rewards are implemented like this:
        Each edge of the graph represents a binary constraint and has 2 nodes connected
        to it. Therefore, for every edge, there is a (n_actions * n_actions) matrix
        containing the rewards for that specific edge. The global reward is then
        obtained by adding all the rewards multiplied by their weight together.
        """
        self.graph = generate_random_graph(self.num_agents, self.num_agents - 1, False)  # undirected complete graph
        self.weights = {}  # weights of all the edges of the graph

        # edges that have a weight value of 0.9 (edges in dark grey fig.5)
        edges_09 = [(0, 1), (1, 2), (0, 2), (4, 5)]
        # the rest have a weight of 0.1
        for (a, b) in list(self.graph.edges):
            if (a, b) in edges_09:
                self.weights[(a, b)] = 0.9
            elif (b, a) in edges_09:
                self.weights[(b, a)] = 0.9
            else:
                self.weights[(a, b)] = 0.1

        self.rewards = {}  # rewards of all the edges of the graph
        for c in self.weights.keys():  # for each constraint
            # N(0, sigma*w_i) with sigma = 10 * # agents
            # n_actions per agent -> each edge has a n_actions * n_actions payoff matrix
            self.rewards[c] = np.random.normal(0, self.weights[c] * (self.num_agents * CONSTANT),
                                               size=(self.num_actions, self.num_actions))

    def initialize_weights_and_rewards_random(self):
        """
        This function is used to initialize the weights and rewards for each
        action of both agents connected by an edge in the graph.

        The rewards are implemented like this:
        Each edge of the graph represents a binary constraint and has 2 nodes connected
        to it. Therefore, for every edge, there is a (n_actions * n_actions) matrix
        containing the rewards for that specific edge. The global reward is then
        obtained by adding all the rewards multiplied by their weight together.
        """
        self.graph = generate_random_graph(self.num_agents, self.num_agents - 1, False)  # undirected complete graph
        self.weights = {}  # weights of all the edges of the graph

        # random weights € [0, 1]
        for (a, b) in list(self.graph.edges):
            self.weights[(a, b)] = round(random.random(), 2)

        self.rewards = {}  # rewards of all the edges of the graph
        for c in self.weights.keys():  # for each constraint
            # N(0, sigma*w_i) with sigma = 10 * # agents
            # n_actions per agent -> each edge has a n_actions * n_actions payoff matrix
            self.rewards[c] = np.random.normal(0, self.weights[c] * (self.num_agents * CONSTANT),
                                               size=(self.num_actions, self.num_actions))

    def act(self, actions: list[int]):
        """
        Method to perform an action in the DCOP Game and obtain the total global reward.
        :param action: The joint action.
        :return: The reward.
        """
        total = 0
        # sum for i=1 to m
        for (n1, n2) in list(self.graph.edges):
            a1, a2 = actions[n1], actions[n2]
            # c_i * (v(x_a), ... , v(x_k))
            total += self.rewards[(n1, n2)][a1][a2]

        return total


if __name__ == "__main__":
    # test
    n_agents = 7
    n_actions = 4
    game = DCOPGame(n_agents, n_actions)

    # # n_agents number of loops
    # for i in range(n_actions):
    #     for j in range(n_actions):
    #         for k in range(n_actions):
    #             for l in range(n_actions):
    #                 for m in range(n_actions):
    #                     # also checks the array indexing does not fail
    #                     assert game.rewards[i][j][k][l][m] == game.act([i, j, k, l, m])
