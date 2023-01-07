from itertools import permutations
import numpy as np
from DCOP_environment import DCOPGame
from DCOP_runner import train_DCOP

from Graph import generate_random_graph

CONSTANT = 10


class CGGame:
    """
    CG Game environment.
    """

    def __init__(self, num_agents, num_actions, num_runs, num_plays, complexity, deterministic=True):

        self.num_meta_agents = num_agents
        self.num_agents = num_agents
        self.num_actions = num_actions
        self.num_runs = num_runs
        self.num_plays = num_plays
        self.graph = generate_random_graph(self.num_meta_agents, 0)
        self.complexity = complexity
        self.deterministic = deterministic
        if complexity == 1:
            self.perms = None
            self.num_actions = self.num_meta_agents
        elif complexity == 2:
            self.perms = list(permutations(range(7), 2))
            self.num_actions = len(self.perms)

    def act(self, actions: list[int]):
        self.graph = generate_random_graph(self.num_meta_agents, 0)
        if self.complexity == 1:
            for i in range(len(actions)):
                neighbor = actions[i]
                agent = i
                if agent != neighbor:
                    self.graph.add_edge(agent, neighbor)
        elif self.complexity == 2:
            for i in range(len(actions)):
                n1, n2 = self.perms[i]
                if n1 != i:
                    self.graph.add_edge(i, n1)
                if n2 != i:
                    self.graph.add_edge(i, n2)

        rew = np.zeros(self.num_runs)
        for i in range(self.num_runs):
            DCOP_env = DCOPGame(self.num_agents, self.num_actions, self.deterministic)
            agents, returns = train_DCOP(DCOP_env, self.graph, self.num_plays)
            rew[i] = np.mean(returns)
        return np.mean(rew)


if __name__ == "__main__":
    n_agents = 7
    n_actions = 4
    game = CGGame(n_agents, n_actions)