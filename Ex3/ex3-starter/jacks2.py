from scipy.stats import poisson
import numpy as np
from enum import IntEnum
from typing import Tuple
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

class Action(IntEnum):
    """Action"""

    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3

class JacksCarRental:
    def __init__(self, modified=False):
        self.modified = modified
        self.max_cars = 20
        self.max_transfer = 5
        self.transfer_cost = 2
        self.additional_parking_cost = 4
        self.rent_reward = 10
        self.gamma = 0.9  # Discount factor
        self.rental_request = [3, 4]
        self.rental_return = [3, 2]
        self.rent = [poisson(mu) for mu in self.rental_request]
        self.return_ = [poisson(mu) for mu in self.rental_return]
        self.free_transfer = 1 if modified else 0
        self.action_space = list(range(-self.max_transfer, self.max_transfer + 1))
        self.state_space = [(x, y) for x in range(self.max_cars + 1) for y in range(self.max_cars + 1)]
        
        # Precompute transitions and rewards
        self.precompute_transitions()
    
    def open_to_close(self, location_idx):
        # Updated maximum number of cars that can start and end at a location
        self.max_cars_start = self.max_cars
        self.max_cars_end = self.max_cars

        prob_matrix = np.zeros((self.max_cars_start + 1, self.max_cars_end + 1))
        reward_vector = np.zeros(self.max_cars_start + 1)

        for start in range(self.max_cars_start + 1):
            for rental_request in range(start + 1):
                rental_prob = self.rent[location_idx].pmf(rental_request)
                rental_reward = self.rent_reward * min(start, rental_request)
                reward_vector[start] += rental_reward * rental_prob

                for return_num in range(self.max_cars_end - start + rental_request + 1):
                    return_prob = self.return_[location_idx].pmf(return_num)
                    end = min(start - rental_request + return_num, self.max_cars_end)
                    prob_matrix[start, end] += rental_prob * return_prob

        return prob_matrix, reward_vector

    def _calculate_cost(self, state, action):
        # Cost of moving cars is 2 per car
        cost = abs(action) * self.transfer_cost
        
        # If modified scenario, the first car moved from the first location is free
        if self.modified and action > 0:
            cost -= self.transfer_cost  # One car is moved for free

        # Additional parking cost is incurred if more than 10 cars are kept overnight at a location
        if self.modified:
            if state[0] - action > 10:
                cost += self.additional_parking_cost
            if state[1] + action > 10:
                cost += self.additional_parking_cost
        
        return cost

    def precompute_transitions(self):
        # Precompute transitions and rewards for each location
        self.transitions = np.zeros((self.max_cars + 1, self.max_cars + 1, len(self.action_space), self.max_cars + 1, self.max_cars + 1))
        self.rewards = np.zeros((self.max_cars + 1, self.max_cars + 1, len(self.action_space)))

        # Use open_to_close function for each location
        day_probs_A, day_rewards_A = self.open_to_close(0)
        day_probs_B, day_rewards_B = self.open_to_close(1)
        
        # Compute transitions and rewards for all state and action pairs
        for state in self.state_space:
            for action in self.action_space:
                if self._valid_action(state, action):
                    next_state = (max(0, min(self.max_cars, state[0] - action)), max(0, min(self.max_cars, state[1] + action)))
                    reward = day_rewards_A[next_state[0]] + day_rewards_B[next_state[1]] - self._calculate_cost(state, action)
                    self.rewards[state[0], state[1], self.action_space.index(action)] = reward
                    for next_state_A in range(self.max_cars + 1):
                        for next_state_B in range(self.max_cars + 1):
                            self.transitions[state[0], state[1], self.action_space.index(action), next_state_A, next_state_B] = day_probs_A[next_state[0], next_state_A] * day_probs_B[next_state[1], next_state_B]

    def _valid_action(self, state, action):
        # Check if the action is valid for the state considering the car transfer constraints
        cars_at_first_location = state[0] - action
        cars_at_second_location = state[1] + action
        return (0 <= cars_at_first_location <= self.max_cars and
                0 <= cars_at_second_location <= self.max_cars and
                -self.max_transfer <= action <= self.max_transfer)

    def expected_return(self, V, state, action):
        # Compute the expected return for a state-action pair
        action_index = self.action_space.index(action)
        expected_return = 0
        for next_state in self.state_space:
            prob = self.transitions[state[0], state[1], action_index, next_state[0], next_state[1]]
            reward = self.rewards[state[0], state[1], action_index]
            expected_return += prob * (reward + self.gamma * V[next_state])
        return expected_return

    
    def transitions(self, state: Tuple, action: Action) -> np.ndarray:
        """Get transition probabilities for given (state, action) pair.

        Note that this is the 3-argument transition version p(s'|s,a).
        This particular environment has stochastic transitions

        Args:
            state (Tuple): state
            action (Action): action

        Returns:
            probs (np.ndarray): return probabilities for next states. Since transition function is of shape (locA, locB, action, locA', locB'), probs should be of shape (locA', locB')
        """
        probs = self.t[state[0], state[1], action + 5, :, :]
        return probs

    def rewards(self, state, action) -> float:
        """Reward function r(s,a)

        Args:
            state (Tuple): state
            action (Action): action
        Returns:
            reward: float
        """
        reward = self.r[state[0], state[1], action + 5]
        return reward
    
def policy_evaluation(V, policy, env, gamma, theta):
    while True:
        delta = 0
        for state in env.state_space:
            old_value = V[state]  # Use the tuple state directly to index V
            action = policy[state]
            # Initialize new value as zero
            new_value = 0
            # Sum over possible next states
            for next_state in env.state_space:
                # Get probability and reward for transitioning to next state
                prob = env.transitions[state[0], state[1], action, next_state[0], next_state[1]]
                reward = env.rewards[state[0], state[1], action]
                new_value += prob * (reward + gamma * V[next_state])
            # Update the value for the state
            V[state] = new_value
            # Update delta for convergence check
            delta = max(delta, abs(old_value - new_value))
        if delta < theta:
            break
    return V

def policy_improvement(V, policy, env, gamma):
    policy_stable = True
    for state in env.state_space:
        old_action = policy[state]
        # Use a list comprehension to calculate expected return for all actions
        action_values = []
        for action in env.action_space:
            action_return = 0
            for next_state in env.state_space:
                # Get probability and reward for transitioning to next state
                prob = env.transitions[state[0], state[1], action, next_state[0], next_state[1]]
                reward = env.rewards[state[0], state[1], action]
                action_return += prob * (reward + gamma * V[next_state])
            action_values.append(action_return)
        # Select the action which gives the maximum expected return
        best_action = env.action_space[np.argmax(action_values)]
        policy[state] = best_action
        # Check if the policy has changed
        if old_action != best_action:
            policy_stable = False
    return policy, policy_stable

def policy_iteration_Jacks(env, gamma=0.9, theta=0.0001):
    policy = {state: 0 for state in env.state_space}
    V = {state: 0 for state in env.state_space}
    policy_stable = False
    iteration = 0
    while not policy_stable:
        V = policy_evaluation(V, policy, env, gamma, theta)
        policy, policy_stable = policy_improvement(V, policy, env, gamma)
        plot_policy(policy, iteration, env.max_cars)  # Plot after each policy improvement
        iteration += 1
    plot_value_function(V, env.max_cars)  # Plot the final value function
    return policy, V

# ============================================ Q6 (a) =========================================================

def plot_policy(policy, iteration, max_cars):
    policy_array = np.zeros((max_cars + 1, max_cars + 1))
    for state, action in policy.items():
        policy_array[state[1], state[0]] = action  # Correct indexing with state tuple

    plt.figure(figsize=(10, 10))
    plt.imshow(policy_array, cmap='RdGy', interpolation='none', extent=[0, max_cars, 0, max_cars])
    plt.colorbar()
    plt.title('Policy (0 = no transfer, +ve = move to second loc, -ve = move to first loc)')
    plt.xlabel('Number of cars at second location')
    plt.ylabel('Number of cars at first location')
    plt.draw()
    plt.savefig(f"policy_iteration_b_{iteration}.png")

def plot_value_function(V, max_cars):
    value_array = np.zeros((max_cars + 1, max_cars + 1))
    for state, value in V.items():
        value_array[state[0], state[1]] = value  # Correct indexing with state tuple

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    X = np.arange(0, max_cars + 1)
    Y = np.arange(0, max_cars + 1)
    X, Y = np.meshgrid(X, Y)
    Z = value_array
    surf = ax.plot_surface(X, Y, Z, cmap='RdGy')
    ax.set_xlabel('Number of cars at first location')
    ax.set_ylabel('Number of cars at second location')
    ax.set_zlabel('Value')
    ax.set_title('Value Function')
    plt.draw()
    plt.savefig("value_function_b.png")

def verify_q6_a():
    # Initialize the Jack's Car Rental environment
    jcr = JacksCarRental()

    # Precompute transitions and rewards
    jcr.precompute_transitions()

    # Run policy iteration to get the optimal policy and value function
    optimal_policy, optimal_value = policy_iteration_Jacks(jcr)


def verify_q6_b():
    # Initialize the Jack's Car Rental environment
    jcr = JacksCarRental(modified=True)

    # Precompute transitions and rewards
    jcr.precompute_transitions()

    # Run policy iteration to get the optimal policy and value function
    optimal_policy, optimal_value = policy_iteration_Jacks(jcr)

if __name__ == "__main__":
    # verify_q6_a()
    verify_q6_b()
    pass