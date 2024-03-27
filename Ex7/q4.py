import numpy as np
import gym
import matplotlib.pyplot as plt
from tiles_encoding import IHT, tiles

def semi_gradient_sarsa(env, num_episodes, alpha, gamma, epsilon, iht_size, num_tilings, max_size):
    iht = IHT(iht_size)
    w = np.zeros(max_size)
    steps_per_episode = []
    weights_per_episode = []  # Store weights for selected episodes

    for episode in range(num_episodes):
        total_reward = 0
        state = env.reset()
        steps = 0
        done = False
        while not done:
            if isinstance(state, tuple):
                x = state[0][0]
                xdot = state[0][1]
            else:
                x = state[0]
                xdot = state[1]
            # Epsilon greedy policy
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
        if episode in [49, 99, 149, 199]:  # Capture weights after these episodes
            weights_per_episode.append((episode, np.copy(w)))
        print(f"Alpha: {alpha} | Episode {episode + 1}: Total Reward = {total_reward}")

    return w, steps_per_episode, weights_per_episode

# Modified plotting function to handle multiple alphas
def plot_steps_per_episode(all_steps_per_episode, alphas):
    plt.figure(figsize=(10, 6))
    for steps, alpha in zip(all_steps_per_episode, alphas):
        plt.plot(steps, label=f'Alpha {alpha * 8}')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.title('Episodes versus Steps for different alphas')
    plt.legend()
    plt.show()

def plot_cost_to_go(env, weights_per_episode, iht, num_tilings, position_scale, velocity_scale, alpha):
    num_plots = len(weights_per_episode)
    position_grid = np.linspace(env.observation_space.low[0], env.observation_space.high[0], 100)
    velocity_grid = np.linspace(env.observation_space.low[1], env.observation_space.high[1], 100)
    X, Y = np.meshgrid(position_grid, velocity_grid)
    
    fig, axes = plt.subplots(1, num_plots, figsize=(20, 4))  # Adjust subplot layout here
    for ax, (episode, weights) in zip(axes, weights_per_episode):
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                features = tiles(iht, num_tilings, [position_scale * X[i, j], velocity_scale * Y[i, j]])
                Z[i, j] = -min([np.sum(weights[feature]) for feature in features])

        cp = ax.contourf(X, Y, Z, levels=50, cmap='viridis')
        fig.colorbar(cp, ax=ax)
        ax.set_title(f'Episode {episode+1} | Alpha {alpha}')
        ax.set_xlabel('Position')
        ax.set_ylabel('Velocity')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Parameters
    num_episodes = 500
    alphas = [0.1/8, 0.2/8, 0.5/8] 
    gamma = 1.0 
    epsilon = 0.1 
    iht_size = 4096
    num_tilings = 8
    max_size = iht_size 

    # Environment setup
    env = gym.make('MountainCar-v0')
    
    # Running the training for each alpha and plotting steps per episode
    all_steps_per_episode = []
    iht = IHT(iht_size)
    for alpha in alphas:
        _, steps, weights_per_episode = semi_gradient_sarsa(env, num_episodes, alpha, gamma, epsilon, iht_size, num_tilings, max_size)
        all_steps_per_episode.append(steps)
        plot_cost_to_go(env, weights_per_episode, iht, num_tilings, 8 / (1.2 + 0.5), 8 / (0.07 + 0.07), alpha)

    plot_steps_per_episode(all_steps_per_episode, alphas)
    env.close()