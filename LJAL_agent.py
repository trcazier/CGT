import numpy as np
import json


class LJALAgent:
    """
    The agent class for par 3.2.
    """

    def __init__(self,
                 num_actions: int,
                 neigbours: list = [],
                 temp_fact: float = 0.94):
        """
        :param num_actions: Number of actions.
        """
        self.num_actions = num_actions
        self.count = {}
        self.neighbours = neigbours
        self.neighbours_counts = self.init_n_counts()
        self.q_table = {}
        self.plays = 1
        self.temperature = 1000
        self.temp_fact = temp_fact

    def init_n_counts(self):
        counts = {}
        for n in self.neighbours:
            for a in range(0, self.num_actions):
                counts[(n, a)] = 0

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

    def act(self) -> int:
        """
        Return the action.

        :param training: Boolean flag for training.
        :return: The action.
        """
        self.temperature = 1000 * pow(self.temp_fact, self.plays)
        evaluations = self.compute_evaluations()
        exp_evaluations = np.array([ev / self.temperature for ev in evaluations])
        max_eval = exp_evaluations.max()
        exp_evaluations = np.exp(exp_evaluations - max_eval)
        probabilities = exp_evaluations/exp_evaluations.sum()
        action = np.random.choice(range(0, self.num_actions), p=probabilities)
        self.plays += 1
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
                neighbour_actions += [action]

        neighbour_actions = json.dumps(list(map(lambda x: int(x), neighbour_actions)))

        if not (act, neighbour_actions) in self.q_table:
            self.q_table[(act, neighbour_actions)] = 0
            self.count[(act, neighbour_actions)] = 0

        current_estimate = self.q_table[(act, neighbour_actions)]

        self.count[(act, neighbour_actions)] += 1
        count = self.count[(act, neighbour_actions)]

        self.q_table[(act, neighbour_actions)] = current_estimate + 0.5 * (rew - current_estimate)
