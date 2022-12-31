import time
from typing import Tuple, List

import numpy as np
from numpy import ndarray

from Graph import DCOP_generate_IL, DCOP_generate_JAL, DCOP_generate_JLAL_1, DCOP_generate_JLAL_2, DCOP_generate_JLAL_3
from LJAL_agent import LJALAgent
import matplotlib.pyplot as plt
import pandas as pd

from DCOP_environment import DCOPGame


def run_episode(env: DCOPGame, agents, training: bool) -> float:
    actions = [agent.act(training=training) for agent in agents]
    rew = env.act(actions)
    for i in range(len(agents)):
        other_actions = {}
        for (idx, action) in enumerate(actions):
            if idx != i:
                other_actions[idx] = action
        agents[i].learn(actions[i], other_actions, rew)
    return rew


def train_DCOP(env: DCOPGame, graph, t_max: int) -> Tuple[List[LJALAgent], ndarray]:
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
    agents = [LJALAgent(env.num_actions, list(graph.neighbors(i))) for i in range(env.num_agents)]
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
    ctr = 0
    
    num_plays = 200
    num_agents = 7
    num_actions = 4

    labels = ["IL", "JAL", "LJAL-1", "LJAL-2", "LJAL-3"]
    graphs = [
        DCOP_generate_IL(),
        DCOP_generate_JAL(),
        DCOP_generate_JLAL_1(),
        DCOP_generate_JLAL_2(),
        DCOP_generate_JLAL_3()
    ]
    
    totals = np.array([np.zeros(num_plays) for _ in range(len(graphs))])

    for graph in graphs:
        t1 = time.time()
        for i in range(1000):
            if i % 1000 == 0:
                print(i)
            env = DCOPGame(num_agents, num_actions)
            agents, returns, m = train_DCOP(env, graph, num_plays)
            totals[ctr] += returns
        totals[ctr] = totals[ctr]/1000
        ctr += 1
        t2 = time.time()
        print("time: ", t2-t1)

    for i in range(len(totals)):
        plt.plot(totals[i], label=labels[i])

    plt.legend()
    plt.show()
