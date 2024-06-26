import numpy as np
from collections import defaultdict
from typing import Callable, Tuple


def default_blackjack_policy(state: Tuple[int, int, bool]) -> int:
    """default_blackjack_policy.

    Returns sticking on 20 or 21 and hit otherwise

    Args:
        state: the current state
    """
    if state[0] in [20, 21]:
        return 0
    else:
        return 1


def create_blackjack_policy(Q: defaultdict) -> Callable:
    """Creates an initial blackjack policy from default_blackjack_policy but updates policy using Q values.

    A policy is represented as a function here because the policies are simple. More complex policies can be represented using classes.

    Args:
        Q (defaultdict): current Q-values
    Returns:
        get_action (Callable): Takes a state as input and outputs an action
    """

    def get_action(state: Tuple) -> int:
        # If state was never seen before, use initial blackjack policy
        if state not in Q.keys():
            return default_blackjack_policy(state)
        else:
            # Choose deterministic greedy action
            chosen_action = np.argmax(Q[state]).item()
            return chosen_action

    return get_action


# def create_epsilon_policy(Q, epsilon):
#     policy_data = {}

#     def update_policy_data():
#         for state, actions in Q.items():
#             best_action = np.argmax(actions)
#             action_probabilities = np.ones(len(actions)) * epsilon / len(actions)
#             action_probabilities[best_action] += 1.0 - epsilon
#             policy_data[state] = action_probabilities

#     def policy_function(state):
#         if state in policy_data:
#             action_probabilities = policy_data[state]
#             action = np.random.choice(np.arange(len(action_probabilities)), p=action_probabilities)
#             return action
#         else:
#             return np.random.randint(len(Q[state]))

#     update_policy_data()  # Update policy data based on current Q
#     return policy_function

def create_epsilon_policy(Q: defaultdict, epsilon: float) -> Callable:
    """Creates an epsilon soft policy from Q values.

    A policy is represented as a function here because the policies are simple. More complex policies can be represented using classes.

    Args:
        Q (defaultdict): current Q-values
        epsilon (float): softness parameter
    Returns:
        get_action (Callable): Takes a state as input and outputs an action
    """
    num_actions = len(Q[0])

    def get_action(state: Tuple) -> int:
        if np.random.random() < epsilon:
            return np.random.choice(num_actions)
        else:
            # Get the best action with a random tie-breaking strategy
            best_actions = np.flatnonzero(Q[state] == np.max(Q[state]))
            return np.random.choice(best_actions)

    return get_action