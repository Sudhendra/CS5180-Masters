import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Callable
from enum import IntEnum
import matplotlib.pyplot as plt
import numpy as np
import random
import heapq
import json
import os

# random.seed(2)

# Declaring global variables
# Walls are listed for you
# Coordinate system is (x, y) where x is the horizontal and y is the vertical direction
WALLS = [
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

# Define a global boundary
left_boundary = [(-1, x) for x in range(11)]
right_boundary = [(11, x) for x in range(11)]
upper_boundary = [(x, 11) for x in range(11)]
lower_boundary = [(x, -1) for x in range(11)]
BOUNDARY = left_boundary + right_boundary + upper_boundary + lower_boundary

# Initialize a global goal state with (None, None)
GOAL_STATE = (-1, -1)


class Action(IntEnum):
    """Action"""

    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3


def actions_to_dxdy(action: Action):
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


def reset():
    """Return agent to start state"""
    return (0, 0)

def goal_state_generator(func):
    """A function which generates a randomized goal state which is neither a wall nor outside the boundary

    Implements a decorator to the simulate function
    Args:
        func: simulate(state: Tuple[int, int], action: Action)

    Returns:
        goal_state: A fixed random goal state (Tuple[int, int])
    """
    goal_state = None

    def wrapper(state, action):
        nonlocal goal_state
        if goal_state is None:
            N = 10
            stateRange = 10
            sample_goal_states = [divmod(element, stateRange + 1) for element in random.sample(range((stateRange + 1) * (stateRange + 1)), N) if divmod(element, stateRange + 1) not in WALLS]
            goal_state = random.choice(sample_goal_states)
            assert goal_state not in WALLS
        return func(state, action, goal_state)

    return wrapper

# Q1
@goal_state_generator
def simulate(state: Tuple[int, int], action: Action, goal_state: Tuple[int, int]):
    """Simulate function for Four Rooms environment

    Implements the transition function p(next_state, reward | state, action).
    The general structure of this function is:
        1. If goal was reached, reset agent to start state
        2. Calculate the action taken from selected action (stochastic transition)
        3. Calculate the next state from the action taken (accounting for boundaries/walls)
        4. Calculate the reward

    Args:
        state (Tuple[int, int]): current agent position (e.g. (1, 3))
        action (Action): selected action from current agent position (must be of type Action defined above)

    Returns:
        next_state (Tuple[int, int]): next agent position
        reward (float): reward for taking action in state
    """
    # check if goal was reached                                      
    goal_reached_flag = state == goal_state

    if goal_reached_flag:
        state = reset()

    # modify action_taken so that 10% of the time, the action_taken is perpendicular to action (there are 2 perpendicular actions for each action)
    # Perpendicular actions for LEFT & RIGHT are UP & DOWN
    # Perpendicular actions for UP & DOWN are LEFT & RIGHT
    if random.random() < 0.1:
        if action in [Action.LEFT, Action.RIGHT]:
            action_taken = random.choice([Action.UP, Action.DOWN])
        
        elif action in [Action.UP, Action.DOWN]:
            action_taken = random.choice([Action.LEFT, Action.RIGHT])
    else:
        action_taken = action

    # TODO calculate the next state and reward given state and action_taken
    # You can use actions_to_dxdy() to calculate the next state
    # Check that the next state is within boundaries and is not a wall
    # One possible way to work with boundaries is to add a boundary wall around environment and
    # simply check whether the next state is a wall
    next_state = tuple(map(sum, zip(state, actions_to_dxdy(action_taken))))
    # check if agent bumps into a wall or out of bounds
    if next_state in WALLS or next_state in BOUNDARY:
        next_state = state

    reward = 1 if next_state == goal_state else 0

    return next_state, reward


# Q2
def agent(
    steps: int = 1000,
    trials: int = 1,
    policy=Callable[[Tuple[int, int], set], Action],
):
    """
    An agent that provides actions to the environment (actions are determined by policy), and receives
    next_state and reward from the environment

    The general structure of this function is:
        1. Loop over the number of trials
        2. Loop over total number of steps
        3. While t < steps
            - Get action from policy
            - Take a step in the environment using simulate()
            - Keep track of the reward
        4. Compute cumulative reward of trial

    Args:
        steps (int): steps
        trials (int): trials
        policy: a function that represents the current policy. Agent follows policy for interacting with environment.
            (e.g. policy=manual_policy, policy=random_policy)
    """
    trial_tracking_dict = {}
    reward_observed_flag = False
    visited_states = set()
    global GOAL_STATE
    
    for t in range(trials):
        state = reset()
        i = 0
        cummulative_reward = 0
        reward_tracking_list = []
        steps_tracking_list = []
        while i < steps:
            previous_state = state
            # Exploration: Explore the environment with random actions until a reward of 1 is discovered
            action = policy(state, visited_states) if reward_observed_flag else random_policy(state)

            # take step in environment using simulate()
            next_state, reward = simulate(state, action)

            state = next_state
            # record the reward
            cummulative_reward += reward

            # set the reward flag to true if a reward of 1 is observed for the first time.
            if reward == 1 and not reward_observed_flag:
                GOAL_STATE = state
                reward_observed_flag = True

            print(f"Current State: {previous_state} | Next State: {state} | Goal State: {GOAL_STATE} | Action: {action} | Reward: {cummulative_reward} | Iteration: {i+1}")

            reward_tracking_list.append(cummulative_reward)
            steps_tracking_list.append(i)
            i+=1
        trial_tracking_dict[f'trial {t+1}'] = [steps_tracking_list, reward_tracking_list]

    # save the tracking as json
    with open(f"learned_{policy.__name__}_results.json", "w") as outfile: 
        json.dump(trial_tracking_dict, outfile)

def random_policy(state: Tuple[int, int]):
    """A random policy that returns an action uniformly at random

    Args:
        state (Tuple[int, int]): current agent position (e.g. (1, 3))

    Returns:
        action (Action)

    """
    return random.choice([Action.LEFT, Action.RIGHT, Action.UP, Action.DOWN])

def heuristic(a, b):
    """Calculate the Manhattan distance between a and b"""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def a_star_search(start, goal):
    frontier = []
    heapq.heappush(frontier, (0, start))
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0

    while frontier:
        _, current = heapq.heappop(frontier)

        if current == goal:
            break

        for action in [Action.LEFT, Action.RIGHT, Action.UP, Action.DOWN]:
            next_state = tuple(map(sum, zip(current, actions_to_dxdy(action))))
            if next_state in WALLS or next_state in BOUNDARY:
                continue

            new_cost = cost_so_far[current] + 1  # Assuming cost = 1 per step
            if next_state not in cost_so_far or new_cost < cost_so_far[next_state]:
                cost_so_far[next_state] = new_cost
                priority = new_cost + heuristic(next_state, goal)
                heapq.heappush(frontier, (priority, next_state))
                came_from[next_state] = current

    return came_from, cost_so_far

def reconstruct_path(came_from, start, goal):
    current = goal
    path = []
    while current != start:
        path.append(current)
        current = came_from[current]
    path.reverse()
    return path

def learned_policy(state: Tuple[int, int], visited_states: set):
    """A policy that learns to find the goal state and maximize reward

    Args:
        state (Tuple[int, int]): current agent position (e.g. (1, 3))

    Returns:
        action (Action)
    """
    global GOAL_STATE

    # Check if the current state is the goal
    if state == GOAL_STATE:
        # Return a random action if the current state is the .
        return random.choice([Action.LEFT, Action.RIGHT, Action.UP, Action.DOWN])

    came_from, _ = a_star_search(state, GOAL_STATE)
    path = reconstruct_path(came_from, state, GOAL_STATE)

    # Check if the path is empty (which might happen if the goal state is reached)
    if not path:
        # Return a default action, e.g., stay still. Here I choose Action.LEFT as a placeholder.
        return Action.LEFT

    # Get the next step in the path
    next_step = path[0]

    # Find the action that leads to the next_step
    for action in Action:
        potential_next_state = tuple(map(sum, zip(state, actions_to_dxdy(action))))
        if potential_next_state == next_step:
            return action

    # If no action leads to next_step (fallback, should not normally be reached)
    return random.choice([Action.LEFT, Action.RIGHT, Action.UP, Action.DOWN])  # Or any default action

def main():
    agent(steps=10000, trials=10, policy=learned_policy)


if __name__ == "__main__":
    main()
