import numpy as np

CONSTANT = 10


class NArmedBanditGame:
    """
    NArmedBandit Game environment.
    """

    def __init__(self, num_agents, num_actions):
        self.num_agents = num_agents
        self.num_actions = num_actions

        self.rewards = np.random.normal(0, num_agents*CONSTANT, size=(tuple([num_actions for _ in range(num_agents)])))

    def act(self, actions: list[int]):
        """
        Method to perform an action in the NArmedBandit Game and obtain the associated reward.
        :param action: The joint action.
        :return: The reward.
        """
        rewards = self.rewards
        for i in actions:
            rewards = rewards[i]
        return rewards


if __name__ == "__main__":
    # test
    n_agents = 5
    n_actions = 4
    game = NArmedBanditGame(n_agents, n_actions)

    # n_agents number of loops
    for i in range(n_actions):
        for j in range(n_actions):
            for k in range(n_actions):
                for l in range(n_actions):
                    for m in range(n_actions):
                        # also checks the array indexing does not fail
                        assert game.rewards[i][j][k][l][m] == game.act([i, j, k, l, m])
    