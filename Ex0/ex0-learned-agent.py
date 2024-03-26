import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Callable
from enum import IntEnum
import matplotlib.pyplot as plt
import numpy as np
import random
import math
import json
import os
import time

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
        # print(f"The Goal State is: {goal_state}")
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
    trial_tracking_dict = {}
    global GOAL_STATE
    
    for t in range(trials):
        i = 0
        cummulative_reward = 0
        reward_tracking_list = []
        steps_tracking_list = []
        GOAL_STATE = (10, 10)
        # reward_observed_flag = False
        initialize_q_table()
        state = reset()
        while i < steps:
            previous_state = state
            # Exploration: Explore the environment with random actions until a reward of 1 is discovered
            action = policy(state)

            # take step in environment using simulate()
            next_state, reward = simulate(state, action)
            

            # Update the Q-table
            update_Q_values(state, action, reward, next_state)

            state = next_state

            # record the reward
            cummulative_reward += reward

            # set the reward flag to true if a reward of 1 is observed for the first time.
            # if reward == 1 and not reward_observed_flag:
            #     GOAL_STATE = state
            #     reward_observed_flag = True

            print(f"Current State: {previous_state} | Next State: {state} | Goal State: {GOAL_STATE} | Action: {action} | Reward: {cummulative_reward} | Iteration: {i+1}")

            reward_tracking_list.append(cummulative_reward)
            steps_tracking_list.append(i)
            i+=1
            
        trial_tracking_dict[f'trial {t+1}'] = [steps_tracking_list, reward_tracking_list]

    # save the tracking as json
    with open(f"qqResults/learned_{policy.__name__}_results.json", "w") as outfile: 
        json.dump(trial_tracking_dict, outfile)

def random_policy(state: Tuple[int, int]):
    """A random policy that returns an action uniformly at random

    Args:
        state (Tuple[int, int]): current agent position (e.g. (1, 3))

    Returns:
        action (Action)

    """
    return random.choice([Action.LEFT, Action.RIGHT, Action.UP, Action.DOWN])

# Initialize Q-table
Q = {}

# Hyperparameters
learning_rate = 0.2
discount_factor = 0.9
exploration_factor = 0.6

# Initialize Q-table
def initialize_q_table():
    """
    Initializes a Q-table of 10x10 grid with zeroes. 
    No Args and no Return values.
    """
    global Q
    Q = {}
    for x in range(11):
        for y in range(11):
            if (x, y) not in WALLS and (x, y) not in BOUNDARY:
                for action in Action:
                    Q[((x, y), action)] = 0

# Q-learning update function
def update_Q_values(state, action, reward, next_state):
    """
    Updates the Q values in teh Q table.
    Args:
        state (Tuple[int, int]): current state of teh agent.
        action (Action): The action the agent takes to get to this state.
        reward (int): The reward the agent recieves for the current transition.
        next_state (Tuple[int, int]): The next state the agent gets to form current state.
    
    Returns:
        None.
    """
    # Calculate the max Q-value for the next state
    max_q_next_state = max([Q[(next_state, a)] for a in Action])

    # Update the current Q-value using the Q-learning formula
    Q[(state, action)] = Q[(state, action)] + learning_rate * (reward + discount_factor * max_q_next_state - Q[(state, action)])

def learned_policy(state: Tuple[int, int]):
    """A Q-Learning policy that learns to find the goal state and maximize reward

    Args:
        state (Tuple[int, int]): current agent position (e.g. (1, 3))

    Returns:
        action (Action)
    """
    if random.random() < exploration_factor:
        return random.choice(list(Action))
    # Choose the best known action based on Q-values (exploitation)
    else:
        actions = [Q[(state, a)] for a in Action]
        return Action(np.argmax(actions))


# Q4
def worse_policy(state: Tuple[int, int]):
    """A policy that is worse than the random_policy

    Args:
        state (Tuple[int, int]): current agent position (e.g. (1, 3))

    Returns:
        action (Action)
    """
    # we implement a policy which actively moves away from the goal.
    
    if state[0] == GOAL_STATE[0]:
        action  = Action.LEFT # move away in X-axis
    elif state[1] == GOAL_STATE[0]:
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
    # We imnplement a policy which actively tries to reduce the distance between its current state and the goal state.
    # calculate the distance from current state to goal state & pick an action which reduces the distance further.
    # we use epsilon greedy algorithm for a trade-off between exploration and exploitation.
    # Here we use a fixed epsilon value of 0.2 for best results
    global GOAL_STATE
    epsilon = 0.2

    actions = compute_legal_actions(state)

    if random.random() < epsilon:
        # calculate distance form each of the states that result by taking each possible action actions from the current state.
        distances_resulting_from_actions = [
            (action, distance(tuple(map(sum, zip(state, actions_to_dxdy(action)))), GOAL_STATE))
            for action in actions
        ]
        best_action, min_distance = min(distances_resulting_from_actions, key=lambda x: x[1])
        return best_action
    else:
        if state[0] == GOAL_STATE[0]:
            action  = Action.RIGHT # move on X-axis
        elif state[1] == GOAL_STATE[0]:
            action = Action.UP # move on Y-axis
        else:
            # If the state is not at or close to goal state, pick a random action with higher probability to go right and up rather than left and down.
            if random.random() < 0.6:
                action = random.choice([Action.RIGHT, Action.UP])
            else:
                action = random.choice([Action.LEFT, Action.DOWN])
            return action
        return action
        # return random.choice([Action.LEFT, Action.RIGHT, Action.UP, Action.DOWN])

import json
import matplotlib.pyplot as plt

def plot_curves(file_names):
    plt.figure(figsize=(10, 6))
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for file_index, file_name in enumerate(file_names):
        with open(file_name, 'r') as file:
            data = json.load(file)

        # Initialize an empty list to store average rewards for each file
        file_avg_rewards = []

        for trial, values in data.items():
            steps, rewards = values

            # Add the rewards to the file_avg_rewards list
            if not file_avg_rewards:
                file_avg_rewards = rewards.copy()
            else:
                for j in range(len(rewards)):
                    file_avg_rewards[j] += rewards[j]

            # Plot each trial with a dotted line
            plt.plot(steps, rewards, linestyle='dotted', color=color_cycle[file_index % len(color_cycle)])

        # Calculate the average rewards for this file
        file_avg_rewards = [r / len(data) for r in file_avg_rewards]

        # Plot the average rewards with a solid line and a distinct marker
        labels = ['Learned Policy', 'Better Policy', 'Random Policy', 'Worse Policy']
        plt.plot(steps, file_avg_rewards, linestyle='solid', marker='o', markersize=4, linewidth=2, color=color_cycle[file_index % len(color_cycle)], label=labels[file_index])

    plt.xlabel('Steps')
    plt.ylabel('Cumulative Reward')
    plt.title('Steps vs Cumulative Reward Across Files')
    plt.legend()

    plt.savefig("combined_plot.png")
    plt.close()

# Example usage

def main():
    agent(steps=10000, trials=10, policy=learned_policy)
    time.sleep(3)
    agent(steps=10000, trials=10, policy=random_policy)
    time.sleep(3)
    agent(steps=10000, trials=10, policy=better_policy)
    time.sleep(3)
    agent(steps=10000, trials=10, policy=worse_policy)
    plot_curves(['q5Results\learned_learned_policy_results.json', 'q5Results\learned_better_policy_results.json', 'q5Results\learned_random_policy_results.json', 'q5Results\learned_worse_policy_results.json'])

    



if __name__ == "__main__":
    main()
