import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from collections import defaultdict
from windy_env import *
from tqdm import tqdm, trange

# Monte Carlo Control
class MonteCarloControl:
    def __init__(self, environment, epsilon=0.8, gamma=0.9):
        self.env = environment
        self.epsilon = epsilon
        self.gamma = gamma
        self.Q = defaultdict(lambda: np.zeros(len(Action)))
        self.returns = defaultdict(list)
        self.policy = defaultdict(lambda: np.random.choice([a for a in Action]))
        self.episode_rewards = []
        self.time_steps_per_episode = []

    def generate_episode(self):
        episode = []
        state = self.env.reset()
        done = False
        rewards = 0
        time_steps = 0
        while not done:
            probs = self.get_policy_probs(state)
            action = np.random.choice(np.arange(len(Action)), p=probs)
            next_state, reward, done, _ = self.env.step(Action(action))
            episode.append((state, action, reward))
            state = next_state
            rewards += reward
            time_steps += 1
        return episode, rewards, time_steps

    def get_policy_probs(self, state):
        probs = np.ones(len(Action)) * self.epsilon / len(Action)
        best_action = np.argmax(self.Q[state])
        probs[best_action] += (1.0 - self.epsilon)
        return probs

    def update_Q(self, episode):
        G = 0
        for state, action, reward in reversed(episode):
            G = self.gamma * G + reward
            if not (state, action) in [(x[0], x[1]) for x in episode[:-1]]:
                self.returns[(state, action)].append(G)
                self.Q[state][action] = np.mean(self.returns[(state, action)])
                best_action = np.argmax(self.Q[state])
                self.policy[state] = best_action

    def train(self, num_episodes=5000):
        for _ in trange(num_episodes):
            episode, rewards, time_steps = self.generate_episode()
            self.update_Q(episode)
            self.episode_rewards.append(rewards)
            self.time_steps_per_episode.append(time_steps)

if __name__ == "__main__":
    num_trials = 10
    num_episodes = 100
    episode_rewards = np.zeros((num_trials, num_episodes))
    all_time_steps = np.zeros((num_trials, num_episodes))

    for trial in range(num_trials):
        env = WindyGridEnv(goal_pos=(7, 3))
        monte_carlo_control = MonteCarloControl(env)
        monte_carlo_control.train(num_episodes=num_episodes)
        episode_rewards[trial, :] = monte_carlo_control.episode_rewards
        all_time_steps[trial, :] = monte_carlo_control.time_steps_per_episode

    # Calculate the mean and the standard error of the rewards across trials
    mean_rewards = np.mean(episode_rewards, axis=0)
    std_error = np.std(episode_rewards, axis=0) / np.sqrt(num_trials)
    average_time_steps = np.mean(all_time_steps, axis=0)

    # Plotting the results
    plt.figure(figsize=(12, 8))
    plt.plot(mean_rewards, label='Average Rewards')
    plt.fill_between(range(num_episodes), 
                    mean_rewards - 1.96 * std_error, 
                    mean_rewards + 1.96 * std_error, 
                    color='blue', alpha=0.2, label='95% Confidence Interval')
    plt.title('Performance of On-policy Monte-Carlo Control across Episodes')
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    plt.legend()
    plt.grid(True)
    plt.show()

    cumulative_time_steps = np.cumsum(average_time_steps)

    plt.figure(figsize=(6, 4)) 
    plt.plot(cumulative_time_steps, np.arange(1, len(cumulative_time_steps) + 1), color='red')  
    plt.xlabel('Time steps')
    plt.ylabel('Episodes')
    plt.title('Learning Curve') 
    plt.tight_layout() 
    plt.show()
