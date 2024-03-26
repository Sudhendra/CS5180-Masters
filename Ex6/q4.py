import gym
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

from windy_env import WindyGridEnv
from sarsa_on_policy import SARSAAgent, Action
from tqdm import tqdm, trange


def generate_episode(env, agent, max_steps=10000):
    episode = []
    state = env.reset()
    action = agent.choose_action(state)
    steps = 0

    while True:
        next_state, reward, done, _ = env.step(Action(action))
        episode.append((state, Action(action), reward))
        if done or steps >= max_steps:
            break
        action = agent.choose_action(next_state)
        state = next_state
        steps += 1
    return episode


def td_prediction(gamma, episodes_data, n=1):
    V = defaultdict(float)

    for episode in tqdm(episodes_data, desc=f"TD{n} Prediction"):
        states, _, rewards = zip(*episode)
        T = len(states)
        for t in range(T):
            tau = t + n
            G = sum([gamma ** (i - t - 1) * rewards[i] for i in range(t + 1, min(tau, T))])
            if tau < T:
                G += gamma ** n * V[states[tau]]
            V[states[t]] += 0.1 * (G - V[states[t]])
    return V


def learning_targets(V, gamma, episodes, n=None):
    targets = []

    if n is None:  # Monte Carlo
        for episode in episodes:
            rewards = [x[2] for x in episode]
            G = sum([gamma ** i * rewards[i] for i in range(len(rewards))])
            targets.append(G)
    else:
        for episode in episodes:
            states, _, rewards = zip(*episode)
            for t in range(len(states)):
                tau = t + n
                G = sum([gamma ** (i - t - 1) * rewards[i] for i in range(t + 1, min(tau, len(states)))])
                if tau < len(states):
                    G += gamma ** n * V[states[tau]]
                targets.append(G)
    return np.array(targets)


def plot_histograms(N_values, methods, n_values, evaluation_episodes, env):
    fig, axs = plt.subplots(len(N_values), len(methods), figsize=(15, 15), tight_layout=True)
    axs = axs.flatten()

    for i, N in enumerate(N_values):
        env.reset()
        agent = SARSAAgent(env)
        agent.train(num_episodes=N)
        for j, method in enumerate(methods):
            n = n_values[method]
            if method != 'Monte Carlo':
                V = td_prediction(agent.gamma, evaluation_episodes, n=n)
                targets = learning_targets(V, agent.gamma, evaluation_episodes, n=n)
            else:
                targets = learning_targets(None, agent.gamma, evaluation_episodes, n=None)

            start_state_targets = [target for episode, target in zip(evaluation_episodes, targets) if
                                   episode[0][0] == env.start_pos]

            axs[i * len(methods) + j].hist(start_state_targets, bins=20, alpha=0.5)
            axs[i * len(methods) + j].set_title(f'{method} - N={N}')
            axs[i * len(methods) + j].set_xlabel('Targets')
            axs[i * len(methods) + j].set_ylabel('Frequency')

    plt.show()

def main():
    N_values = [1, 10, 50]
    methods = ['TD(0)', 'n-step TD', 'Monte Carlo']
    n_values = {'TD(0)': 1, 'n-step TD': 4, 'Monte Carlo': None}
    env = WindyGridEnv()
    evaluation_episodes = [generate_episode(env, SARSAAgent(env)) for _ in range(100)]
    plot_histograms(N_values, methods, n_values, evaluation_episodes, env)

if __name__ == "__main__":
    main()
