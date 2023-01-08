import time
from typing import Tuple, List

import numpy as np
from numpy import ndarray
from DCOP_environment import DCOPGame
from DCOP_runner import train_DCOP

from Graph import DCOP_generate_IL, DCOP_generate_JAL, DCOP_generate_LJAL_1
from LJAL_agent import LJALAgent
import matplotlib.pyplot as plt
import pandas as pd

from CG_environment import CGGame


def run_episode(env: CGGame, meta_agents) -> float:
    actions = [meta_agent.act() for meta_agent in meta_agents]
    rew = env.act(actions)
    for i in range(len(meta_agents)):
        meta_agents[i].learn(actions[i], {}, rew)
    return rew


def train_CG(env: CGGame, t_max: int) -> Tuple[List[LJALAgent], ndarray]:
    meta_agents = [LJALAgent(env.num_actions, [], 0.994) for _ in range(env.num_agents)]
    returns = np.zeros(t_max)
    counter = 0
    while counter < t_max:
        rew = run_episode(env, meta_agents)
        returns[counter] = rew
        counter += 1
    return meta_agents, returns


def run_exp_3(deterministic):
    if deterministic:
        print("deterministic")
    else:
        print("non-deterministic")

    t_max = 1500

    num_agents = 7
    num_actions = 4
    num_runs = 10
    num_plays = 200
    runs = 10

    # with meta loops
    ctr = 0
    meta_labels = ["OptLJAL-1"]
    envs = [
        CGGame(num_agents, num_actions, num_runs, num_plays, 1, deterministic),  # OptLJAL-1
    ]
    meta_totals = np.array([np.zeros(t_max) for _ in range(len(envs))])
    for env in envs:
        t1 = time.time()
        for i in range(runs):
            meta_agents, returns = train_CG(env, t_max)
            meta_totals[ctr] += returns
            tt2 = time.time()
            time_done = tt2 - t1
            time_left = (time_done / (i + 1)) * (runs - (i + 1))
            print(f"run {i + 1}/{runs} done, estimated time left: {time_left}")
        t2 = time.time()
        print(f"{meta_labels[ctr]} time: ", t2 - t1)
        for i in range(num_agents):
            print(list(env.graph.neighbors(i)))
        meta_totals[ctr] = meta_totals[ctr] / runs
        ctr += 1

    # without meta loops
    ctr = 0
    labels = ["IL", "JAL", "LJAL-1"]
    graphs = [
        DCOP_generate_IL(),
        DCOP_generate_JAL(),
        DCOP_generate_LJAL_1()
    ]
    nonmeta_solution_quality = np.array([np.zeros(runs) for _ in range(len(graphs))])
    for graph in graphs:
        t1 = time.time()
        for i in range(runs):
            print(i)
            env = DCOPGame(num_agents, num_actions, deterministic)
            agents, returns = train_DCOP(env, graph, num_plays)
            nonmeta_solution_quality[ctr] += returns[num_plays-1]
        nonmeta_solution_quality[ctr] = nonmeta_solution_quality[ctr] / runs
        t2 = time.time()
        print(f"{labels[ctr]} time: ", t2 - t1)
        ctr += 1

    nonmeta_solution_quality = list(map(lambda x: [x] * t_max, np.mean(nonmeta_solution_quality, axis=1)))

    labels = meta_labels + labels
    totals = list(meta_totals) + nonmeta_solution_quality

    fig, ax = plt.subplots()

    fig.patch.set_visible(False)
    ax.axis('off')

    solution_quality = list(map(lambda x: x[-1], totals))
    solution_quality = np.array(list(map(lambda x: round(x/solution_quality[len(labels)-1] * 100, 1), solution_quality)))

    sq = pd.DataFrame([solution_quality], columns=labels)
    sq.style.set_caption("Solution quality")

    table = ax.table(cellText=sq.values, colLabels=sq.columns, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)

    fig.tight_layout()
    plt.show()
    plt.clf()

    for i in range(len(totals)):
        plt.plot(totals[i], label=labels[i])

    plt.legend()
    plt.show()


if __name__ == '__main__':
    run_exp_3(True)
    #run_exp_3(False)
