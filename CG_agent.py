import numpy as np
import json


class CGAgent:
    """
    The agent class for part 5
    """

    def __init__(self,
                 num_actions: int,
                 neigbours: list = []):
        """
        :param num_actions: Number of actions.
        """
        self.num_actions = num_actions
        self.count = {}
        self.neighbours = neigbours
        self.neighbours_counts = self.init_n_counts()
        self.q_table = {}
        self.plays = 0
        self.temperature = 1000

    def init_n_counts(self):
        counts = {}
        for n in self.neighbours:
            for a in range(0, self.num_actions):
                counts[(n, a)] = 1

        return counts

    def calculate_frequencies_product(self, n_actions):
        frequency_product = 1
        for idx, action in enumerate(n_actions):
            frequency_product *= self.neighbours_counts[(self.neighbours[idx], action)] / \
                                 sum([self.neighbours_counts[(self.neighbours[idx], action2)]
                                      for action2 in range(0, self.num_actions)])
        return frequency_product

    def compute_evaluations(self) -> np.array:
        evaluations = np.zeros(self.num_actions)
        for action in range(0, self.num_actions):
            evaluations[action] = sum(self.q_table[(a, n_actions)]
                                      * self.calculate_frequencies_product(json.loads(n_actions)) for (a, n_actions)
                                      in self.q_table if a == action)
        return evaluations

    def act(self, training: bool = True) -> int:
        """
        Return the action.

        :param training: Boolean flag for training.
        :return: The action.
        """
        self.plays += 1
        self.temperature = max(1000 * pow(0.94, self.plays), 0.0001)
        evaluations = self.compute_evaluations()
        probabilities = np.zeros(self.num_actions)
        to_exp = [ev / self.temperature for ev in evaluations]
        diff = max(to_exp) - 700
        if diff > 0:
            to_exp = list(map(lambda x: x - diff, to_exp))
        total_sum = sum(np.exp(x) for x in to_exp)

        for action in range(0, self.num_actions):
            p = np.exp(to_exp[action]) / total_sum
            if np.isnan(p):
                probabilities[action] = 1
            else:
                probabilities[action] = p

        action = np.random.choice(range(0, self.num_actions), p=probabilities)
        return action

    def learn(self, act: int, other_actions: dict[int, int], rew: float) -> None:
        """
        Update the Q-Value.

        :param other_actions: the actions of the other agents.
        :param act: The action.
        :param rew: The reward.
        """
        neighbour_actions = []
        for (agent, action) in other_actions.items():
            if agent in self.neighbours:
                if (agent, action) in self.neighbours_counts:
                    self.neighbours_counts[(agent, action)] += 1
                else:
                    self.neighbours_counts[(agent, action)] = 1
                neighbour_actions += [action]

        neighbour_actions = json.dumps(list(map(lambda x: int(x), neighbour_actions)))

        if not (act, neighbour_actions) in self.q_table:
            self.q_table[(act, neighbour_actions)] = 0
            self.count[(act, neighbour_actions)] = 0

        current_estimate = self.q_table[(act, neighbour_actions)]

        self.count[(act, neighbour_actions)] += 1
        count = self.count[(act, neighbour_actions)]

        self.q_table[(act, neighbour_actions)] = current_estimate + (1 / count) * (rew - current_estimate)
