import numpy as np
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import gym
from gym import spaces
from racetracks import track0, track1, initialize_track_environment

# Assuming TrackEnvironment, initialize_track_environment, and track0/track1 are defined as provided previously

def generate_episode_from_limit_stochastic(env, Q, epsilon, nA):
    """ Generates an episode using an epsilon-greedy policy based on Q. """
    episode = []
    state = env.reset()
    while True:
        if state in Q and np.random.rand() > epsilon:
            action = np.argmax(Q[state])
        else:
            action = np.random.choice(np.arange(nA))
        next_state, reward, done, _ = env.step(action)
        episode.append((state, action, reward))
        if done:
            break
        state = next_state
    return episode

def off_policy_mc_control(env, num_episodes, discount_factor=1.0, epsilon=0.1):
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    C = defaultdict(lambda: np.zeros(env.action_space.n))
    target_policy = defaultdict(float)
    
    for i_episode in range(num_episodes): 
        episode = generate_episode_from_limit_stochastic(env, Q, epsilon, env.action_space.n)
        G = 0.0
        W = 1.0
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G = discount_factor * G + reward
            C[state][action] += W
            Q[state][action] += (W / C[state][action]) * (G - Q[state][action])
            if action != np.argmax(target_policy[state]):
                break
            W = W * 1./ (epsilon / env.action_space.n + (1 - epsilon) if action == np.argmax(Q[state]) else epsilon / env.action_space.n)
    
    for state in Q:
        target_policy[state] = np.argmax(Q[state])
    return Q, target_policy

def run_simulation(track, num_trials, num_episodes, discount_factor, epsilon):
    env = initialize_track_environment(track)
    rewards = np.zeros((num_trials, num_episodes))

    for i in tqdm(range(num_trials)):
        Q, policy = off_policy_mc_control(env, num_episodes, discount_factor, epsilon)
        for episode_num in range(num_episodes):
            episode = generate_episode_from_limit_stochastic(env, Q, epsilon, env.action_space.n)
            rewards[i, episode_num] = sum([reward * discount_factor**i for i, (_, _, reward) in enumerate(episode)])
    
    mean_rewards = rewards.mean(axis=0)
    std_rewards = rewards.std(axis=0)
    
    plt.fill_between(range(num_episodes), mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.1)
    plt.plot(range(num_episodes), mean_rewards, label='Mean Total Reward')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Off-Policy MC Control Learning Curve')
    plt.legend()
    plt.draw()
    plt.savefig("q5bt1.png")

if __name__ == "__main__":
    num_trials = 10
    num_episodes = 10000
    discount_factor = 0.99
    epsilon = 0.1
    track = track1()  # or track1() for the second track

    run_simulation(track, num_trials, num_episodes, discount_factor, epsilon)
