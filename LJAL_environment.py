import numpy as np

CONSTANT = 10


class NArmedBanditGame:
    def __init__(self, num_agents, num_actions):
        self.num_agents = num_agents
        self.num_actions = num_actions

        self.rewards = np.random.normal(0, num_agents*CONSTANT, size=(tuple([num_actions for _ in range(num_agents)])))

    def act(self, actions: list[int]):
        rewards = self.rewards
        for i in actions:
            rewards = rewards[i]
        return rewards


if __name__ == "__main__":
    n_agents = 5
    n_actions = 4
    game = NArmedBanditGame(n_agents, n_actions)