import numpy as np
import gym
from gym.envs.registration import register
from tqdm import tqdm
import matplotlib.pyplot as plt
from q3 import feature_function, approximate_q_value
from fourRoomsEnv import register_env, FourRoomsEnv

# Ensure that the environment registration and the FourRoomsEnv definition are in place
register_env()

# Define the feature function, approximate_q_value, and semi_gradient_sarsa as given

def plot_learning_curve(rewards, title='Learning Curve'):
    episodes = np.arange(rewards.shape[1])  # Shape[1] is the number of episodes
    mean_rewards = np.mean(rewards, axis=0)  # Mean across trials
    std_rewards = np.std(rewards, axis=0)  # Standard deviation across trials
    
    plt.fill_between(episodes, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.1, color="g")
    plt.plot(episodes, mean_rewards, 'g-', label='mean rewards')
    plt.title(title)
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.show()

def run_trials(num_trials, env, num_episodes, alpha, gamma, epsilon):
    all_rewards = np.zeros((num_trials, num_episodes))  # Initialize an array to hold all rewards
    for trial in tqdm(range(num_trials)):
        _, rewards = semi_gradient_sarsa(env, num_episodes, alpha, gamma, epsilon, return_rewards=True)
        all_rewards[trial, :] = rewards  # Store rewards for this trial
    
    plot_learning_curve(all_rewards)


# Modify semi_gradient_sarsa to also return rewards for each episode
def semi_gradient_sarsa(env, num_episodes, alpha, gamma, epsilon, return_rewards=False):
    num_states = env.rows * env.cols
    num_actions = env.action_space.n
    weight_matrix = np.zeros((num_actions, num_states))
    episode_rewards = []  # To store total reward received in each episode

    for episode in range(num_episodes):
        state = env.reset()
        feature_vector = feature_function(state, num_states)
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax([approximate_q_value(feature_vector, weight_matrix, a) for a in range(num_actions)])
        done = False
        total_reward = 0  # Total reward for this episode

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
            total_reward += reward

        episode_rewards.append(total_reward)

    if return_rewards:
        return weight_matrix, episode_rewards
    return weight_matrix

# Hyperparameters and trials setup
num_trials = 10
num_episodes = 100
alpha = 0.1
gamma = 0.99
epsilon = 0.8

fourRoomsEnv = FourRoomsEnv()
run_trials(num_trials, fourRoomsEnv, num_episodes, alpha, gamma, epsilon)
