import time
from typing import Tuple, List

import numpy as np
from numpy import ndarray
from DCOP_environment import DCOPGame
from DCOP_runner import train_DCOP

from Graph import DCOP_generate_IL, DCOP_generate_JAL, DCOP_generate_JLAL_1, DCOP_generate_JLAL_2, DCOP_generate_JLAL_3, GC_generate_OPTJLAL
from LJAL_agent import LJALAgent
import matplotlib.pyplot as plt
import pandas as pd

from CG_environment import CGGame


def run_episode(env: CGGame, meta_agents, training: bool) -> float:

    actions = [meta_agent.act(training=training) for meta_agent in meta_agents]
    rew = env.act(actions)
    for i in range(len(meta_agents)):
        meta_agents[i].learn(actions[i], [], rew)
    return rew


def train_CG(env: CGGame, t_max: int) -> Tuple[List[LJALAgent], ndarray]:
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
    meta_agents = [LJALAgent(env.num_actions, []) for _ in range(env.num_agents)]
    returns = np.zeros(t_max)
    counter = 0
    while counter < t_max:
        rew = run_episode(env, meta_agents, True)
        returns[counter] = rew
        counter += 1
    return meta_agents, returns

def run_exp_3(deterministic):
    t_max = 1500

    num_agents = 7
    num_actions = 4
    num_runs = 100
    num_plays = 200
    runs = 1000

    # with meta loops
    ctr = 0
    labels = ["OptLJAL-1", "OptLJAL-2"]
    envs = [
        CGGame(num_agents, num_actions, num_runs, num_plays, 1, deterministic), # OptLJAL-1
        CGGame(num_agents, num_actions, num_runs, num_plays, 2, deterministic) # OptLJAL-2
    ]
    meta_totals = np.array([np.zeros(num_plays) for _ in range(len(envs))])
    for env in envs:
        t1 = time.time()
        meta_agents, returns = train_CG(env, t_max)
        meta_totals[ctr] += returns
        t2 = time.time()
        print(f"{labels[ctr]} time: ", t2-t1)
        ctr += 1
    meta_totals[ctr] = meta_totals[ctr]/runs

    # without meta loops
    ctr = 0
    labels = ["IL", "JAL", "LJAL"]
    graphs = [
        DCOP_generate_IL(),
        DCOP_generate_JAL(),
        DCOP_generate_JLAL_1()
    ]
    nonmeta_totals = np.array([np.zeros(num_plays) for _ in range(len(graphs))])
    for graph in graphs:
        t1 = time.time()
        for i in range(runs):
            env = DCOPGame(num_agents, num_actions, deterministic)
            agents, returns = train_DCOP(env, graph, num_plays)
            nonmeta_totals[ctr] += returns
        nonmeta_totals[ctr] = nonmeta_totals[ctr]/runs
        t2 = time.time()
        print(f"{labels[ctr]} time: ", t2-t1)
        ctr += 1

    # for i in range(len(totals)):
    #     plt.plot(totals[i], label=labels[i])

    # plt.legend()
    # plt.show()


if __name__ == '__main__':
    run_exp_3(True)
    run_exp_3(False)
