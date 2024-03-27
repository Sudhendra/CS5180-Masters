import numpy as np
import gym
import matplotlib.pyplot as plt
from tiles_encoding import IHT, tiles


num_episodes = 200
alpha = 0.1
gamma = 1.0  
epsilon = 0.1
iht_size = 4096
num_tilings = 8
max_size = iht_size

# Environment setup
env = gym.make('MountainCar-v0')

def semi_gradient_sarsa(env, num_episodes, alpha, gamma, epsilon, iht_size, num_tilings, max_size):
    iht = IHT(iht_size)
    w = np.zeros(max_size)
    steps_per_episode = []  # For tracking steps per episode for plotting

    for episode in range(num_episodes):
        total_reward = 0
        state = env.reset()
        steps = 0  # Track steps per episode
        
        done = False
        while not done:
            if isinstance(state, tuple):
                x = state[0][0]
                xdot = state[0][1]
            else:
                x = state[0]
                xdot = state[1]
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                q_values = [sum(w[tiles(iht, num_tilings, [8 * x / (1.2 + 0.5), 8 * xdot / (0.07 + 0.07)], [action])]) for action in range(env.action_space.n)]
                action = np.argmax(q_values)
                
            next_state, reward, done, info, _ = env.step(action)
            total_reward += reward
            steps += 1
            
            if done:
                target = reward
            else:
                next_q_values = [sum(w[tiles(iht, num_tilings, [8 * next_state[0] / (1.2 + 0.5), 8 * next_state[1] / (0.07 + 0.07)], [next_action])]) for next_action in range(env.action_space.n)]
                target = reward + gamma * np.max(next_q_values)
            
            indices = tiles(iht, num_tilings, [8 * x / (1.2 + 0.5), 8 * xdot / (0.07 + 0.07)], [action])
            for index in indices:
                w[index] += alpha * (target - sum(w[indices])) / num_tilings
            
            state = next_state
        
        steps_per_episode.append(steps)
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")

    return w, steps_per_episode


def plot_steps_per_episode(steps_per_episode):
    plt.figure(figsize=(10, 6))
    plt.plot(steps_per_episode, label='Steps per episode')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.title('Episodes versus Steps')
    plt.legend()
    plt.show()

# Running the training
weights, steps_per_episode = semi_gradient_sarsa(env, num_episodes, alpha, gamma, epsilon, iht_size, num_tilings, max_size)

# Now plot
plot_steps_per_episode(steps_per_episode)