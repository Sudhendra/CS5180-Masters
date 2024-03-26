import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from env import FourRoomsEnv 
from policy import create_epsilon_policy  
from tqdm import tqdm  

def on_policy_mc_control(env, gamma=0.99, epsilon=0.1, num_episodes=10000, every_visit=True):
    """Performs on-policy Monte Carlo control with progress bar."""
    A = env.action_space.n  # Assuming env.action_space.n gives the number of actions
    Q = defaultdict(lambda: np.zeros(A))
    N = defaultdict(lambda: np.zeros(A))
    Gs = []

    policy = create_epsilon_policy(Q, epsilon)

    for _ in tqdm(range(num_episodes), desc="Episodes"):
        s = env.reset()
        episodes = []
        done = False
        while not done:
            a = policy(s)
            next_s, r, done, _ = env.step(a)
            episodes.append((s, a, r))
            s = next_s

        G = 0
        for t, (s, a, r) in enumerate(reversed(episodes)):
            G = gamma * G + r
            if every_visit or (s, a) not in episodes[:-t-1]:
                N[s][a] += 1
                Q[s][a] += (G - Q[s][a]) / N[s][a]

        Gs.append(G)

    return Gs

def plot_avg_se(data, num_se=1.96, label='', linestyle='-', color=None):
    means = np.mean(data, axis=0)
    ses = np.std(data, axis=0, ddof=1) / np.sqrt(len(data))
    episodes = np.arange(1, len(means) + 1)
    plt.plot(episodes, means, linestyle, label=label, color=color)
    plt.fill_between(episodes, means - num_se * ses, means + num_se * ses, alpha=0.2, color=color)

def simulate_and_plot(env, num_trials=10, num_episodes=1000, epsilons=[0.1, 0.01, 0]):
    """Simulates the environment and plots the results with progress bar for trials."""
    plt.figure(figsize=(10, 6))

    for epsilon, color, linestyle in zip(epsilons, ['b', 'r', 'g'], ['-', '--', '-.']):
        rewards = []
        for _ in tqdm(range(num_trials), desc=f"Trials ε={epsilon}"):
            rewards.append(on_policy_mc_control(env, epsilon=epsilon, num_episodes=num_episodes))
        plot_avg_se(rewards, label=f'ε={epsilon}', linestyle=linestyle, color=color)

    plt.title("Average Cumulative Reward over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Average Cumulative Reward")
    plt.legend()
    plt.grid(True)
    plt.draw()
    plt.savefig("Q4bBBB.jpg")

if __name__ == "__main__":
    env = FourRoomsEnv(goal_pos=(10, 10))  # Ensure this constructor matches your environment setup
    simulate_and_plot(env, num_trials=5, num_episodes=10)
