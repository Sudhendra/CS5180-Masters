import matplotlib.pyplot as plt
from collections import defaultdict, deque
import numpy as np
from collections import defaultdict
from windy_env import *
from tqdm import tqdm, trange

class NStepSARSAAgent:
    def __init__(self, environment, n=4, alpha=0.5, gamma=0.9, epsilon=0.1):
        self.env = environment
        self.n = n 
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

    def train(self, num_episodes=1000):
        for _ in tqdm(range(num_episodes)):
            state = self.env.reset()
            done = False
            total_reward = 0
            time_steps = 0

            states = deque(maxlen=self.n+1)
            actions = deque(maxlen=self.n+1)
            rewards = deque(maxlen=self.n)
            
            # Start the first action
            action = self.choose_action(state)
            states.append(state)
            actions.append(action)

            while not done or len(rewards) > 0:
                if not done:
                    next_state, reward, done, _ = self.env.step(action)
                    next_action = self.choose_action(next_state) if not done else None

                    # Append to deques
                    states.append(next_state)
                    actions.append(next_action)
                    rewards.append(reward)

                    total_reward += reward
                    time_steps += 1
                
                if len(rewards) == self.n or done:
                    G = sum([self.gamma**i * rewards[i] for i in range(len(rewards))])
                    if not done:
                        G += (self.gamma**len(rewards)) * self.Q[next_state][next_action]
                    first_state, first_action = states.popleft(), actions.popleft()
                    self.Q[first_state][first_action] += self.alpha * (G - self.Q[first_state][first_action])
                    rewards.popleft() 

                if not done:
                    state, action = next_state, next_action

            self.episode_rewards.append(total_reward)
            self.time_steps_per_episode.append(time_steps)

if __name__ == "__main__":
    num_trials = 10
    num_episodes = 500
    episode_rewards = np.zeros((num_trials, num_episodes))
    all_time_steps = np.zeros((num_trials, num_episodes))


    for trial in range(num_trials):
        env = WindyGridEnv(goal_pos=(7, 3))
        n_step_sarsa_agent = NStepSARSAAgent(env)
        n_step_sarsa_agent.train(num_episodes=num_episodes)
        episode_rewards[trial, :] = n_step_sarsa_agent.episode_rewards
        all_time_steps[trial, :] = n_step_sarsa_agent.time_steps_per_episode

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
    plt.title('Performance of On-policy SARSA across Episodes')
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

