from enum import IntEnum
from typing import Tuple, Optional, List
import gym
from gym import spaces
from gym.utils import seeding

class Action(IntEnum):
    """Enumeration of possible actions in the Windy Grid environment."""
    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3
    NORTHEAST = 4
    NORTHWEST = 5
    SOUTHEAST = 6
    SOUTHWEST = 7
    # NOACTION = 8

def actions_to_dxdy(action: Action) -> Tuple[int, int]:
    """Map an action to a change in x and y coordinates.
    
    Args:
        action (Action): The action taken by the agent.

    Returns:
        A tuple representing the change in the x and y coordinates.
    """
    return {
        Action.LEFT: (-1, 0),
        Action.DOWN: (0, -1), 
        Action.RIGHT: (1, 0),
        Action.UP: (0, 1),
        Action.NORTHEAST: (1, 1),
        Action.NORTHWEST: (-1, 1),
        Action.SOUTHEAST: (1, -1),
        Action.SOUTHWEST: (-1, -1),
        # Action.NOACTION: (0, 0),
    }[action]

class KingWindyGridEnv(gym.Env):
    """A gym environment for the Windy Gridworld task."""
    metadata = {'render.modes': ['human']}

    def __init__(self, goal_pos=(7, 3)) -> None:
        super(KingWindyGridEnv, self).__init__()
        self.rows = 6
        self.cols = 9
        self.wind = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
        self.start_pos = (0, 3)
        self.goal_pos = goal_pos
        self.agent_pos = self.start_pos

        self.action_space = spaces.Discrete(len(Action))
        self.observation_space = spaces.Tuple(
            (spaces.Discrete(self.cols), spaces.Discrete(self.rows))
        )

    def reset(self) -> Tuple[int, int]:
        self.agent_pos = self.start_pos
        return self.agent_pos

    def step(self, action: Action) -> Tuple[Tuple[int, int], float, bool, dict]:
        if self.agent_pos == self.goal_pos:
            return self.agent_pos, 0.0, True, {}

        # Calculate next position without wind effect
        dx, dy = actions_to_dxdy(action)
        next_pos = (self.agent_pos[0] + dx, self.agent_pos[1] + dy)

        # Apply wind effect
        wind_strength = self.wind[next_pos[0]-1]
        next_pos = (next_pos[0], max(0, min(self.rows - 1, next_pos[1] - wind_strength)))

        # Ensure the next position is valid; if not, agent stays in the same position
        if 0 <= next_pos[0] < self.cols and 0 <= next_pos[1] < self.rows:
            self.agent_pos = next_pos
        reward = -1.0
        done = self.agent_pos == self.goal_pos
        return self.agent_pos, reward, done, {}