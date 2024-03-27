import numpy as np
import gym
from gym.envs.registration import register
from fourRoomsEnv import *
from tqdm import trange

def feature_function(state, num_states):
    """Simple state aggregation: One-hot encoding of states."""
    feature_vector = np.zeros(num_states)
    state_index = state[0] * 11 + state[1]  # Assuming grid size of 11x11
    feature_vector[state_index] = 1
    return feature_vector

def approximate_q_value(feature_vector, weight_matrix, action):
    """Compute the approximate Q-value for a given action."""
    return np.dot(feature_vector, weight_matrix[action])

def semi_gradient_sarsa(env, num_episodes, alpha, gamma, epsilon):
    num_states = env.rows * env.cols
    num_actions = env.action_space.n
    weight_matrix = np.zeros((num_actions, num_states))

    for episode in trange(num_episodes):
        state = env.reset()
        feature_vector = feature_function(state, num_states)
        action = np.argmax([approximate_q_value(feature_vector, weight_matrix, a) for a in range(num_actions)])
        done = False

        while not done:
            next_state, reward, done, _ = env.step(action)
            next_feature_vector = feature_function(next_state, num_states)
            if np.random.rand() < epsilon:
                next_action = env.action_space.sample()
            else:
                next_action = np.argmax([approximate_q_value(next_feature_vector, weight_matrix, a) for a in range(num_actions)])
            
            td_target = reward + gamma * approximate_q_value(next_feature_vector, weight_matrix, next_action) * (not done)
            td_error = td_target - approximate_q_value(feature_vector, weight_matrix, action)
            
            weight_matrix[action] += alpha * td_error * feature_vector
            
            state = next_state
            action = next_action

    return weight_matrix

if __name__ == "__main__":
    register_env()
    fourRoomsEnv = FourRoomsEnv()

    # Hyperparameters
    num_episodes = 100
    alpha = 0.1
    gamma = 0.99

    weights = semi_gradient_sarsa(fourRoomsEnv, num_episodes, alpha, gamma, 0.8)

    # Additional code to plot learning curves, visualize results, etc., can be added here.
