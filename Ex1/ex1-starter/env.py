import numpy as np
from typing import Tuple
import random


class BanditEnv:
    """Multi-armed bandit environment"""

    def __init__(self, k: int) -> None:
        """__init__.

        Args:
            k (int): number of arms/bandits
        """
        self.k = k

    def reset(self) -> None:
        """Resets the mean payout/reward of each arm.
        This function should be called at least once after __init__()
        """
        # Initialize means of each arm distributed according to standard normal
        self.means = np.random.normal(size=self.k)

    def step(self, action: int) -> Tuple[float, int]:
        """Take one step in env (pull one arm) and observe reward

        Args:
            action (int): index of arm to pull
        """
        # calculate reward of arm given by action
        # set scale (standard deviation) parameter to 1.
        # since variance = (standard deviation)**2, scale remains 1 for a variance of 1.
        # if action >0 and action <=self.k else raise ValueError("Action must be witin the range of k-arms.")
        reward = np.random.normal(self.means[action], scale=1.0)

        return reward

class NonStationaryBanditEnv(BanditEnv):
    """Non Stationary Multi-armed bandit environment"""

    def __init__(self, k: int) -> None:
        super().__init__(k)

    def step(self, action: int) -> Tuple[float, int]:
        reward = super().step(action)

        # Update means by a small random walk
        self.means += np.random.normal(0, 0.01, self.k) # this provides a non stationary reward distribution from the Bandit Environment.

        return reward

class BanditEnvWithSwitchingCost(BanditEnv):
    """Multi-armed bandit environment with a cost for switching arms"""

    def __init__(self, k: int, switch_cost: float) -> None:
        super().__init__(k)
        self.switch_cost = switch_cost
        self.last_action = None

    def step(self, action: int) -> float:
        reward = super().step(action)
        if self.last_action is not None and self.last_action != action:
            reward -= self.switch_cost
        self.last_action = action
        return reward

