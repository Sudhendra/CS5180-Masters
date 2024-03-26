import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Callable
from enum import IntEnum
import matplotlib.pyplot as plt
import random
import math
import json
import os

# random.seed(32)

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

left_boundary = [(-1, x) for x in range(11)]
right_boundary = [(11, x) for x in range(11)]
upper_boundary = [(x, 11) for x in range(11)]
lower_boundary = [(x, -1) for x in range(11)]
BOUNDARY = left_boundary + right_boundary + upper_boundary + lower_boundary


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


# Q1
def simulate(state: Tuple[int, int], action: Action):
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
    # TODO check if goal was reached
    goal_state = (10, 10)
    goal_reached_flag = state == goal_state

    if goal_reached_flag:
        state = reset()

    # TODO modify action_taken so that 10% of the time, the action_taken is perpendicular to action (there are 2 perpendicular actions for each action)

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
def manual_policy(state: Tuple[int, int]):
    """A manual policy that queries user for action and returns that action

    Args:
        state (Tuple[int, int]): current agent position (e.g. (1, 3))

    Returns:
        action (Action)
    """
    # TODO
    next_action = input("""Pick an action:
        1. UP - W
        2. LEFT - A
        3. DOWN - S
        4. RIGHT - D\n""")
    mapping = {
        "A": Action.LEFT,
        "S": Action.DOWN,
        "D": Action.RIGHT,
        "W": Action.UP,
    }
    return mapping[next_action.upper()]

# Q2
def agent(
    steps: int = 1000,
    trials: int = 1,
    policy=Callable[[Tuple[int, int]], Action],
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
    # TODO you can use the following structure and add to it as needed
    trial_tracking_dict = {}

    for t in range(trials):
        state = reset()
        i = 0
        cummulative_reward = 0
        reward_tracking_list = []
        steps_tracking_list = []
        while i < steps:
            previous_state = state
            # TODO select action to take
            action = policy(state)
            # TODO take step in environment using simulate()
            next_state, reward = simulate(state, action)
            state = next_state
            # TODO record the reward
            cummulative_reward += reward
            print(f"Current State: {previous_state} | Next State: {state} | Action: {action} | Reward: {cummulative_reward} | Iteration: {i+1}")
            reward_tracking_list.append(cummulative_reward)
            steps_tracking_list.append(i)
            i+=1
        trial_tracking_dict[f'trial {t+1}'] = [steps_tracking_list, reward_tracking_list]

    # save the tracking as json
    with open(f"{policy.__name__}_results.json", "w") as outfile: 
        json.dump(trial_tracking_dict, outfile)

# Q3
def random_policy(state: Tuple[int, int]):
    """A random policy that returns an action uniformly at random

    Args:
        state (Tuple[int, int]): current agent position (e.g. (1, 3))

    Returns:
        action (Action)
    """
    # TODO
    return random.choice([Action.LEFT, Action.RIGHT, Action.UP, Action.DOWN])


# Q4
def worse_policy(state: Tuple[int, int]):
    """A policy that is worse than the random_policy

    Args:
        state (Tuple[int, int]): current agent position (e.g. (1, 3))

    Returns:
        action (Action)
    """
    # TODO
    # we implement a policy which actively moves away from the goal.
    goal_state = (10, 10)
    
    if state[0] == goal_state[0]:
        action  = Action.LEFT # move away in X-axis
    elif state[1] == goal_state[0]:
        action = Action.DOWN # move away in Y-axis
    else:
        # If the state is not at or close to goal state, pick a random action with higfher probability to go left and down ratehr than up and right
        if random.random() < 0.6:
            action = random.choice([Action.LEFT, Action.DOWN])
        else:
            action = random.choice([Action.RIGHT, Action.UP])
    return action


# Q4
def distance(p0, p1):
    return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)

def compute_legal_actions(state: Tuple[int, int]):
    """A check that investigates which actions lead to the agent not bumping into the walls or the boundaries from it's current 
    state.

    Args:
        state (Tuple[int, int]): current agent position (e.g. (1, 3))

    Returns:
        list of actions (Action)
    """
    # compute all possible states the agent can get to from this state
    actions = [Action.LEFT, Action.RIGHT, Action.UP, Action.DOWN]
    possible_next_states = [
            (action, tuple(map(sum, zip(state, actions_to_dxdy(action)))))
            for action in actions
        ]

    # Filter actions that do not lead to a wall or boundary
    legal_actions = [action for action, next_state in possible_next_states 
                    if next_state not in WALLS and next_state not in BOUNDARY]
    
    return legal_actions

def better_policy(state: Tuple[int, int]):
    """A policy that is better than the random_policy

    Args:
        state (Tuple[int, int]): current agent position (e.g. (1, 3))

    Returns:
        action (Action)
    """
    # TODO
    # We imnplement a policy which actively tries to reduce the distance between its current state and the goal state.
    # calculate the distance from current state to goal state & pick an action which reduces the distance further.
    # we use epsilon greedy algorithm for a trade-off between exploration and exploitation.
    # Here we use a fixed epsilon value of 0.2 for best results
    epsilon = 0.2
    goal_state = (10, 10)
    actions = compute_legal_actions(state)

    if random.random() < epsilon:
        # calculate distance form each of the states that result by taking each possible action actions from the current state.
        distances_resulting_from_actions = [
            (action, distance(tuple(map(sum, zip(state, actions_to_dxdy(action)))), goal_state))
            for action in actions
        ]
        best_action, min_distance = min(distances_resulting_from_actions, key=lambda x: x[1])
        return best_action
    else:
        if state[0] == goal_state[0]:
            action  = Action.RIGHT # move away in X-axis
        elif state[1] == goal_state[0]:
            action = Action.UP # move away in Y-axis
        else:
            # If the state is not at or close to goal state, pick a random action with higher probability to go right and up rather than left and down.
            if random.random() < 0.6:
                action = random.choice([Action.RIGHT, Action.UP])
            else:
                action = random.choice([Action.LEFT, Action.DOWN])
            return action
        return action
        # return random.choice([Action.LEFT, Action.RIGHT, Action.UP, Action.DOWN])

def plot_curves(file_name):
    with open(file_name, 'r') as file:
        data = json.load(file)

    plt.figure(figsize=(10, 6))
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Initialize an empty list to store reward values for each step
    avg_rewards = []

    for i, (trial, values) in enumerate(data.items()):
        steps, rewards = values

        # Add the rewards to the avg_rewards list
        if not avg_rewards:
            avg_rewards = rewards.copy()
        else:
            for j in range(len(rewards)):
                avg_rewards[j] += rewards[j]

        plt.plot(steps, rewards, linestyle='dotted', label=trial, color=color_cycle[i % len(color_cycle)])

    # Calculate the average rewards
    avg_rewards = [r / len(data) for r in avg_rewards]

    # Plot the average rewards as a solid black line
    plt.plot(steps, avg_rewards, linestyle='solid', color='black', label='Average Reward')

    plt.xlabel('Steps')
    plt.ylabel('Cumulative Reward')
    plt.title('Steps vs Cumulative Reward')
    plt.legend()

    plt.savefig(f"{file_name[:-5]}.png")
    plt.close()



def main():
    # TODO run code for Q2~Q4 and plot results
    # You may be able to reuse the agent() function for each question
    agent(steps=10000, trials=1, policy=manual_policy)
    agent(steps=10000, trials=10, policy=random_policy)
    agent(steps=10000, trials=10, policy=worse_policy)
    agent(steps=10000, trials=10, policy=better_policy)
    plot_curves("random_policy_results.json")
    plot_curves("better_policy_results.json")
    plot_curves("worse_policy_results.json")
    plot_curves("q5Results/learned_learned_policy_results.json")
    plot_curves("q5Results/learned_better_policy_results.json")
    plot_curves("q5Results/learned_random_policy_results.json")
    plot_curves("q5Results/learned_worse_policy_results.json")

if __name__ == "__main__":
    main()
