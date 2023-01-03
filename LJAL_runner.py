from typing import Tuple, List

import numpy as np
from numpy import ndarray

from Graph import generate_random_graph
from LJAL_agent import LJALAgent
import matplotlib.pyplot as plt
import pandas as pd

from LJAL_environment import NArmedBanditGame


def run_episode(env: NArmedBanditGame, agents, training: bool) -> float:
    actions = [agent.act(training=training) for agent in agents]
    rew = env.act(actions)
    for i in range(len(agents)):
        other_actions = {}
        for (idx, action) in enumerate(actions):
            if idx != i:
                other_actions[idx] = action
        agents[i].learn(actions[i], other_actions, rew)
    return rew


def train_LJAL(env: NArmedBanditGame, graph, t_max: int) -> Tuple[List[LJALAgent], ndarray]:
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
    agents = [LJALAgent(4, list(graph.neighbors(i))) for i in range(5)]
    returns = np.zeros(t_max)
    counter = 0
    while counter < t_max:
        rew = run_episode(env, agents, True)
        returns[counter] = rew
        counter += 1
    return agents, returns


if __name__ == '__main__':
    totals = np.array([np.zeros(200) for i in range(4)])
    ctr = 0
    labels = ["IQL", "LJAL-2", "LJAL-3", "JAL"]
    for edges in [0, 2, 3, 4]:
        for i in range(1000):
            if i % 1000 == 0:
                print(i)
            env = NArmedBanditGame(5, 4)
            graph = generate_random_graph(5, edges)
            agents, returns = train_LJAL(env, graph, 200)
            totals[ctr] += returns
        totals[ctr] = totals[ctr]/1000
        ctr += 1

    for i in range(len(totals)):
        plt.plot(totals[i], label=labels[i])

    plt.legend()
    plt.show()
