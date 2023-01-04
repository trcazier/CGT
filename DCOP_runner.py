import time
from typing import Tuple, List

import numpy as np
from numpy import ndarray

from Graph import DCOP_generate_IL, DCOP_generate_JAL, DCOP_generate_LJAL_1, DCOP_generate_LJAL_2, DCOP_generate_LJAL_3
from LJAL_agent import LJALAgent
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
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
    while counter < t_max:
        rew = run_episode(env, agents, True)
        returns[counter] = rew
        counter += 1
    return agents, returns


if __name__ == '__main__':
    ctr = 0

    num_plays = 200
    num_agents = 7
    num_actions = 4
    runs = 3

    labels = ["IL", "LJAL-1", "LJAL-2", "LJAL-3", "JAL"]
    graphs = [
        DCOP_generate_IL(),
        DCOP_generate_LJAL_1(),
        DCOP_generate_LJAL_2(),
        DCOP_generate_LJAL_3(),
        DCOP_generate_JAL()
    ]

    totals = np.array([np.zeros(num_plays) for _ in range(len(graphs))])
    solution_quality = np.array([np.zeros(runs) for _ in range(len(graphs))])
    run_times = np.zeros(len(graphs))

    for graph in graphs:
        t1 = time.time()
        for i in range(runs):
            env = DCOPGame(num_agents, num_actions)
            agents, returns = train_DCOP(env, graph, num_plays)
            totals[ctr] += returns
            solution_quality[ctr][i] = returns[num_plays-1]
        totals[ctr] = totals[ctr] / runs
        t2 = time.time()
        run_times[ctr] = t2-t1
        print(f"{labels[ctr]} time: ", t2 - t1)
        ctr += 1

    for i in range(0, len(labels)):
        for j in range(i + 1, len(labels)):
            v1 = solution_quality[i]
            v2 = solution_quality[j]
            _stat, p = ttest_ind(v1, v2)
            if p >= 0.05:
                print("T-test: Solution qualities for {} and {} are NOT significantly different.".format(labels[i], labels[j]))
            t1 = run_times[i]
            t2 = run_times[j]
            _stat2, p2 = ttest_ind(t1, t2)
            if p2 >= 0.05:
                print("T-test: Run-times for {} and {} are NOT significantly different.".format(labels[i], labels[j]))

    fig, ax = plt.subplots()

    fig.patch.set_visible(False)
    ax.axis('off')

    solution_quality = np.mean(solution_quality, axis=1)
    solution_quality = np.array(list(map(lambda x: round(x/solution_quality[len(graphs)-1] * 100, 1), solution_quality)))
    run_times = np.array(list(map(lambda x: round(run_times[len(graphs)-1]/x, 1), run_times)))

    sq = pd.DataFrame([solution_quality], columns=labels)
    rt = pd.DataFrame([run_times], columns=labels)
    sq.style.set_caption("Solution quality")
    rt.style.set_caption("Speed")

    table = ax.table(cellText=sq.values, colLabels=sq.columns, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)

    fig.tight_layout()
    plt.show()
    plt.clf()

    fig, ax = plt.subplots()

    fig.patch.set_visible(False)
    ax.axis('off')

    table = ax.table(cellText=rt.values, colLabels=rt.columns, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)

    fig.tight_layout()
    plt.show()
    plt.clf()

    for i in range(len(totals)):
        plt.plot(totals[i], label=labels[i])

    plt.legend()
    plt.show()
