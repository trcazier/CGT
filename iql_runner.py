from typing import Tuple, List

import numpy as np
from numpy import ndarray

from Graph import generate_graph
from iql_agent import IQLAgent
import matplotlib.pyplot as plt
import pandas as pd

from Environment import NArmedBanditGame


def run_episode(env: NArmedBanditGame, agents, training: bool) -> float:
    actions = [agent.act(training=training) for agent in agents]
    rew = env.act(actions)
    for i in range(0, len(agents)):
        other_actions = {}
        for (idx, action) in enumerate(actions):
            if idx != i:
                other_actions[idx] = action
        agents[i].learn(actions[i], other_actions, rew)
    return rew


def train_iql(env: NArmedBanditGame, graph, t_max: int) -> Tuple[List[IQLAgent], ndarray]:
    """
    Training loop.

    :param temperature:
    :param env: The gym environment.
    :param t_max: The number of timesteps.
    :param evaluate_every: Evaluation frequency.
    :param num_evaluation_episodes: Number of episodes for evaluation.
    :return: Tuple containing the list of agents, the returns of all training episodes, the averaged evaluation
    return of each evaluation, and the list of the greedy joint action of each evaluation.
    """
    agents = [IQLAgent(4, graph.neighbors(i)) for i in range(0, 5)]
    returns = np.zeros(t_max)
    counter = 0
    m = 0
    while counter < t_max:
        rew = run_episode(env, agents, True)
        returns[counter] = rew
        m = max(rew, m)
        counter += 1

    return agents, returns, m


if __name__ == '__main__':
    totals = 0
    for i in range(0, 10000):
        if i % 1000 == 0:
            print(i)
        env = NArmedBanditGame(5, 4)
        graph = generate_graph(5, 4)
        number_of_iterations = 100
        agents, returns, m = train_iql(env, graph, 200)
        totals += returns[-1]
    print(totals/10000)
