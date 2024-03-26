import numpy as np
from enum import IntEnum
from typing import Tuple
import numpy as np


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


class Gridworld5x5:
    """5x5 Gridworld"""

    def __init__(self) -> None:
        """
        State: (x, y) coordinates

        Actions: See class(Action).
        """
        self.rows = 5
        self.cols = 5
        self.state_space = [
            (x, y) for x in range(0, self.rows) for y in range(0, self.cols)
        ]
        self.action_space = len(Action)

        # TODO set the locations of A and B, the next locations, and their rewards
        self.A = (0,1)
        self.A_prime = (4,1)
        self.A_reward = 10
        self.B = (0,3)
        self.B_prime = (2,3)
        self.B_reward = 5

    def transitions(
        self, state: Tuple, action: Action
    ) -> Tuple[Tuple[int, int], float]:
        """Get transitions from given (state, action) pair.

        Note that this is the 4-argument transition version p(s',r|s,a).
        This particular environment has deterministic transitions

        Args:
            state (Tuple): state
            action (Action): action

        Returns:
            next_state: Tuple[int, int]
            reward: float
        """
        if state == self.A:
            return self.A_prime, self.A_reward
        elif state == self.B:
            return self.B_prime, self.B_reward
        else:
            dx, dy = actions_to_dxdy(action)
            next_state = (max(0, min(state[0] + dx, self.rows - 1)),
                          max(0, min(state[1] + dy, self.cols - 1)))

            # Moving out of bounds results in -1 reward
            reward = -1 if next_state == state else 0
            return next_state, reward

    def expected_return(
        self, V, state: Tuple[int, int], action: Action, gamma: float
    ) -> float:
        """Compute the expected_return for all transitions from the (s,a) pair, i.e. do a 1-step Bellman backup.

        Args:
            V (np.ndarray): list of state values (length = number of states)
            state (Tuple[int, int]): state
            action (Action): action
            gamma (float): discount factor

        Returns:
            ret (float): the expected return
        """

        next_state, reward = self.transitions(state, action)
        # Compute the expected return using the 1-step Bellman backup
        ret = reward + gamma * V[next_state[0], next_state[1]]

        return ret
    
# ============================================ Q5 (a) ==========================================================
def verify_q5_a():
    # Constants for iterative policy evaluation
    gamma = 0.9  # Discount factor
    theta = 0.0001  # Small threshold determining accuracy of estimation

    # Initialize the gridworld
    env = Gridworld5x5()

    # Initialize state-value function with zeros
    V = np.zeros((env.rows, env.cols))

    # Assume uniform policy: equal probability for each action
    policy_prob = 1 / env.action_space  # Equiprobable random policy

    is_value_changed = True
    iteration = 0

    while is_value_changed:
        delta = 0
        for state in env.state_space:
            v = V[state[0], state[1]]  # Store the old value
            V[state[0], state[1]] = sum([policy_prob * env.expected_return(V, state, action, gamma)
                                        for action in Action])  # Bellman equation
            delta = max(delta, abs(v - V[state[0], state[1]]))  # Max change in value function across all states
        is_value_changed = delta > theta
        iteration += 1

    # After convergence, print the final state-value function V and the number of iterations
    print(V,"\n")
    print("Iteration: ", iteration)

# ============================================ Q5 (b) ==========================================================
def value_iteration_multi_action(env, gamma=0.9, theta=0.0001):
    V = np.zeros((env.rows, env.cols))
    # Initialize a policy list to hold lists of best actions for each state
    policy = [[[] for _ in range(env.cols)] for _ in range(env.rows)]

    while True:
        delta = 0
        for state in env.state_space:
            v = V[state[0], state[1]]
            action_values = [env.expected_return(V, state, Action(action), gamma) for action in Action]
            max_value = np.max(action_values)
            V[state[0], state[1]] = max_value
            delta = max(delta, abs(v - V[state[0], state[1]]))

            # Find all actions that are equally good
            policy[state[0]][state[1]] = [action for action, value in enumerate(action_values) if value == max_value]

        if delta < theta:
            break

    return V, policy


def verify_q5_b():
    # Instantiate the environment
    env = Gridworld5x5()

    # Perform value iteration
    V_star, pi_star = value_iteration_multi_action(env)

    print(V_star,"\n")
    print(pi_star, "\n")

# ============================================ Q5 (c) ==========================================================
def policy_iteration_multi_action(env, gamma=0.9, theta=0.0001):
    # Initialize a policy list to hold lists of best actions for each state
    policy = [[[] for _ in range(env.cols)] for _ in range(env.rows)]
    V = np.zeros((env.rows, env.cols))

    policy_stable = False

    while not policy_stable:
        # Policy Evaluation
        while True:
            delta = 0
            for state in env.state_space:
                v = V[state[0], state[1]]
                # Use current policy for the state if it exists, else consider all actions
                if policy[state[0]][state[1]]:
                    action_returns = [env.expected_return(V, state, Action(action), gamma) for action in policy[state[0]][state[1]]]
                else:
                    action_returns = [env.expected_return(V, state, Action(action), gamma) for action in Action]
                V[state[0], state[1]] = sum(action_returns) / len(action_returns) if action_returns else 0
                delta = max(delta, abs(v - V[state[0], state[1]]))
            if delta < theta:
                break

        # Policy Improvement
        policy_stable = True
        for state in env.state_space:
            old_actions = policy[state[0]][state[1]].copy()
            action_values = [env.expected_return(V, state, Action(action), gamma) for action in Action]
            max_value = max(action_values)
            best_actions = [action for action, value in enumerate(action_values) if value == max_value]

            policy[state[0]][state[1]] = best_actions

            if old_actions != best_actions:
                policy_stable = False

    return V, policy

def verify_q5_c():
    # Instantiate the environment
    env = Gridworld5x5()

    # Perform policy iteration
    V_star_pi, pi_star_pi = policy_iteration_multi_action(env)

    print(V_star_pi)
    # for i, layer in enumerate(pi_star_pi):
    #     for row in layer:
    #         print(row)
    print(pi_star_pi)

if __name__ == "__main__":
    verify_q5_a() # verified
    verify_q5_b() # verified
    verify_q5_c() # verified
    pass
