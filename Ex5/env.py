from enum import IntEnum
from typing import Tuple, Optional, List
from gym import Env, spaces
from gym.utils import seeding
from gym.envs.registration import register
import random
# from algorithms import on_policy_mc_control_epsilon_soft
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def register_env() -> None:
    """Register custom gym environment so that we can use `gym.make()`

    In your main file, call this function before using `gym.make()` to use the Four Rooms environment.
        register_env()
        env = gym.make('FourRooms-v0')

    Note: the max_episode_steps option controls the time limit of the environment.
    You can remove the argument to make FourRooms run without a timeout.
    """
    register(id="FourRooms-v0", entry_point="env:FourRoomsEnv", max_episode_steps=459)


class Action(IntEnum):
    """Action"""

    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3


def actions_to_dxdy(action: Action) -> Tuple[int, int]:
    """
    Helper function to map action to changes in x and y coordinates
    Args:
        action (Action): taken action
    Returns:
        dxdy (Tuple[int, int]): Change in x and y coordinates
    """
    mapping = {
        Action.LEFT: (-1, 0),
        Action.DOWN: (0, -1),
        Action.RIGHT: (1, 0),
        Action.UP: (0, 1),
    }
    return mapping[action]


class FourRoomsEnv(Env):
    """Four Rooms gym environment.

    This is a minimal example of how to create a custom gym environment. By conforming to the Gym API, you can use the same `generate_episode()` function for both Blackjack and Four Rooms envs.
    """

    def __init__(self, goal_pos=(10, 10)) -> None:
        self.rows = 11
        self.cols = 11

        # Coordinate system is (x, y) where x is the horizontal and y is the vertical direction
        self.walls = [
            (0, 5),
            (2, 5),
            (3, 5),
            (4, 5),
            (5, 0),
            (5, 2),
            (5, 3),
            (5, 4),
            (5, 5),
            (5, 6),
            (5, 7),
            (5, 9),
            (5, 10),
            (6, 4),
            (7, 4),
            (9, 4),
            (10, 4),
        ]

        self.start_pos = (0, 0)
        self.goal_pos = goal_pos
        self.agent_pos = None

        self.action_space = spaces.Discrete(len(Action))
        self.observation_space = spaces.Tuple(
            (spaces.Discrete(self.rows), spaces.Discrete(self.cols))
        )

    def seed(self, seed: Optional[int] = None) -> List[int]:
        """Fix seed of environment

        In order to make the environment completely reproducible, call this function and seed the action space as well.
            env = gym.make(...)
            env.seed(seed)
            env.action_space.seed(seed)

        This function does not need to be used for this assignment, it is given only for reference.
        """

        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def is_position_valid(self, agent_pos: Tuple[int, int]) -> bool:
        x, y = agent_pos
        # Check within bounds
        if x < 0 or x >= self.rows or y < 0 or y >= self.cols:
            return False
        # Check not a wall
        if agent_pos in self.walls:
            return False
        return True
    
    def reset(self) -> Tuple[int, int]:
        """Reset agent to the starting position.

        Returns:
            observation (Tuple[int,int]): returns the initial observation
        """
        self.agent_pos = self.start_pos

        return self.agent_pos

    def step(self, action: Action) -> Tuple[Tuple[int, int], float, bool, dict]:
        """Take one step in the environment.

        Takes in an action and returns the (next state, reward, done, info).
        See https://github.com/openai/gym/blob/master/gym/core.py#L42-L58 foand r more info.

        Args:
            action (Action): an action provided by the agent

        Returns:
            observation (object): agent's observation after taking one step in environment (this would be the next state s')
            reward (float) : reward for this transition
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning). Not used in this assignment.
        """

        # Check if goal was reached
        if self.agent_pos == self.goal_pos:
            done = True
            reward = 1.0
        else:
            done = False
            reward = 0.0

        if random.random() < 0.1:
            if action in [Action.LEFT, Action.RIGHT]:
                action_taken = random.choice([Action.UP, Action.DOWN])
            
            elif action in [Action.UP, Action.DOWN]:
                action_taken = random.choice([Action.LEFT, Action.RIGHT])
        else:
            action_taken = action

        next_pos = tuple(map(sum, zip(self.agent_pos, actions_to_dxdy(action_taken))))


        if self.is_position_valid(next_pos):
            self.agent_pos = next_pos
        else:
            pass

        return self.agent_pos, reward, done, {}

# def off_policy_mc_prediction(episodes, behavior_policy_probabilities, gamma=0.99):
#     A = [Action.LEFT, Action.DOWN, Action.RIGHT, Action.UP]  # Define action space
#     Q = defaultdict(lambda: np.zeros(len(A)))
#     C = defaultdict(lambda: np.zeros(len(A)))  # Cumulative sum of the weights
#     i = 0
#     for episode, behavior_probs in tqdm(zip(episodes, behavior_policy_probabilities)):
#         G = 0
#         W = 1  # Importance sampling ratio, initialized to 1 for the start of each episode
#         for t in reversed(range(len(episode))):
#             state, action, reward = episode[t]
#             # Get the probability of the action taken at time t from the behavior policy
#             # which is stored in behavior_policy_probabilities
#             action_prob = behavior_probs[t]
#             if action_prob > 0:
#                 W /= action_prob  # Update the importance sampling ratio
#             else:
#                 print("here",state)
#                 break  # If action_prob is 0, terminate early as weight would be infinite

#             G = gamma * G + reward  # Update the return
#             C[state][action] += W
#             Q[state][action] += (W / C[state][action]) * (G - Q[state][action])
#             # If the action taken is not the best according to the current Q, break the loop
#             if action != np.argmax(Q[state]):
#                 break
        
#     return Q