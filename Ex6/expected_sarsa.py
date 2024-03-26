import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from collections import defaultdict
from windy_env import *
from tqdm import tqdm, trange

class ExpectedSARSAAgent:
    def __init__(self, environment, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.env = environment
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = defaultdict(lambda: np.zeros(len(Action)))
        self.episode_rewards = []
        self.time_steps_per_episode = []

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(list(Action))
        else:
            return np.argmax(self.Q[state])

    def expected_Q(self, next_state):
        action_probs = np.ones(len(Action)) * self.epsilon / len(Action)
        best_action = np.argmax(self.Q[next_state])
        action_probs[best_action] += (1.0 - self.epsilon)
        expected_q = np.dot(self.Q[next_state], action_probs)
        return expected_q

    def update_Q_expected_sarsa(self, state, action, reward, next_state):
        predict = self.Q[state][action]
        expected_q = self.expected_Q(next_state)
        target = reward + self.gamma * expected_q
        self.Q[state][action] += self.alpha * (target - predict)

    def train(self, num_episodes=1000):
        for _ in trange(num_episodes):
            state = self.env.reset()
            done = False
            time_steps = 0
            rewards = 0

            while not done:
                action = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(Action(action))
                rewards += reward
                self.update_Q_expected_sarsa(state, action, reward, next_state)
                
                state = next_state
                time_steps += 1
            self.episode_rewards.append(rewards)
            self.time_steps_per_episode.append(time_steps)

if __name__ == "__main__":
    num_trials = 10
    num_episodes = 500
    episode_rewards = np.zeros((num_trials, num_episodes))
    all_time_steps = np.zeros((num_trials, num_episodes))


    for trial in range(num_trials):
        env = WindyGridEnv(goal_pos=(7, 3))
        sarsa_agent = ExpectedSARSAAgent(env)
        sarsa_agent.train(num_episodes=num_episodes)
        episode_rewards[trial, :] = sarsa_agent.episode_rewards
        all_time_steps[trial, :] = sarsa_agent.time_steps_per_episode


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
    plt.title('Performance of Expected SARSA across Episodes')
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

