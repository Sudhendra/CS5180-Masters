import numpy as np
import gym
from tiles_encoding import IHT, tiles
from q4 import semi_gradient_sarsa
import matplotlib.pyplot as plt

def x_function(iht, num_tilings, state, action=None):
    max_size = iht.size
    if isinstance(state, tuple):
        x = state[0][0]
        xdot = state[0][1]
    else:
        x = state[0]
        xdot = state[1]
    if action is not None:
        features= tiles(iht, num_tilings, [8 * x / (1.2 + 0.5), 8 * xdot / (0.07 + 0.07)], [action])
    else:
        features= tiles(iht, num_tilings, [8 * x / (1.2 + 0.5), 8 * xdot / (0.07 + 0.07)], [])
    feature_vector = np.zeros(max_size)
    for f in features:
        feature_vector[f] = 1
    return feature_vector

def gradient_td_update(alpha, gamma, w, R, S, S_prime, iht, num_tilings):
    x_S = x_function(iht, num_tilings, S)
    x_S_prime = x_function(iht, num_tilings, S_prime)
    
    v_S = np.dot(w, x_S)
    v_S_prime = np.dot(w, x_S_prime)
    
    td_error = R + gamma * v_S_prime - v_S
    
    w += alpha * td_error * (x_S - gamma * x_S_prime)
    return w

# Plotting
def plot_comparison(steps_per_episode_semi, steps_per_episode):
    plt.figure(figsize=(12, 8))
    plt.plot(steps_per_episode_semi, label='Semi-Gradient', alpha=0.75)
    plt.plot(steps_per_episode, label='Full-Gradient', alpha=0.75)
    plt.xlabel('Episodes')
    plt.ylabel('Steps per Episode')
    plt.title('Semi-Gradient vs Full-Gradient SARSA: Steps per Episode')
    plt.legend()
    plt.grid(True)
    plt.show()

    
if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    iht = IHT(4096)  # Initialize Index Hash Table for tile coding
    num_tilings = 8
    w = np.zeros(iht.size)  # Initialize weights
    alpha = 0.5 / num_tilings  # Adjust learning rate by number of tilings
    gamma = 1.0  # Discount factor

    num_episodes = 500
    steps_per_episode = []


    print("Running semi-gradient SARSA...")
    weights, steps_per_episode_semi, weights_per_episode = semi_gradient_sarsa(env, num_episodes, alpha, gamma, 0.1, iht.size, num_tilings, iht.size)

    # Main RL loop
    print("Running full gradient SARSA...")
    for episode in range(num_episodes):
        total_reward = 0
        S = env.reset()
        done = False
        steps = 0

        while not done:
            action = env.action_space.sample()  # Replace with policy based on Q-values for actual control
            S_prime, R, done, info, _ = env.step(action)
            w = gradient_td_update(alpha, gamma, w, R, S, S_prime, iht, num_tilings)
            S = S_prime
            total_reward += R
            steps += 1

        steps_per_episode.append(steps)
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")

    env.close()
    plot_comparison(steps_per_episode_semi, steps_per_episode)
