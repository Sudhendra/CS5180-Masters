import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm, trange
from env import FourRoomsEnv, Action
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt

def create_random_policy(action_space_n: int):
    def policy(state):
        return np.random.choice(action_space_n)
    return policy

def create_epsilon_greedy_policy(Q, epsilon, action_space_n):
    def policy(state):
        if np.random.rand() < epsilon:
            return np.random.choice(action_space_n)
        else:
            return np.argmax(Q[state])
    return policy

def on_policy_mc_control(env, gamma=0.99, num_episodes=int(100), every_visit=True):
    A = [Action.LEFT, Action.DOWN, Action.RIGHT, Action.UP]
    Q = defaultdict(lambda: np.zeros(len(A)))
    N = defaultdict(lambda: np.zeros(len(A)))
    epsilon = 0.8
    episodes = []
    action_probabilities = []

    policy = create_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

    for _ in trange(num_episodes):
        s = env.reset()
        episode_data = []
        episode_probabilities = []
        done = False
        while not done:
            a = policy(s)
            next_s, r, done, _ = env.step(a)
            episode_data.append((s, a, r))
            episode_probabilities.append(1.0 / len(A))  # Each action has equal probability
            s = next_s

        episodes.append(episode_data)
        action_probabilities.append(episode_probabilities)

        G = 0
        for t, (s, a, r) in enumerate(reversed(episode_data)):
            G = gamma * G + r
            if every_visit or (s, a) not in [(x[0], x[1]) for x in episode_data[:-t-1]]:
                N[s][a] += 1
                Q[s][a] += (1 / N[s][a]) * (G - Q[s][a])

    return Q, policy

def off_policy_mc_prediction(episodes, behavior_policy_probabilities, gamma=0.99):
    Q = defaultdict(lambda: np.zeros(len(Action)))
    C = defaultdict(lambda: np.zeros(len(Action)))

    for episode, behavior_probs in tqdm(zip(episodes, behavior_policy_probabilities), desc="Off-policy MC Control"):
        G = 0
        W = 1
        for t, (state, action, reward) in enumerate(reversed(episode)):
            G = gamma * G + reward
            C[state][action] += W
            Q[state][action] += (W / C[state][action]) * (G - Q[state][action])
            if action != np.argmax(Q[state]): 
                break
            W *= 1./behavior_probs[t] 
    return Q

def compute_greedy_policy(Q):
    policy = {}
    for state, actions in Q.items():
        policy[state] = np.argmax(actions)
    return policy

def generate_episodes_with_probabilities(env, policy_func, epsilon=None, num_episodes=100):
    episodes = []
    behavior_probabilities = []
    action_space_n = env.action_space.n 

    for _ in trange(num_episodes, desc=f"Generating episodes for {policy_func.__name__}"):
        episode = []
        episode_probs = []
        state = env.reset()
        done = False
        while not done:
            if epsilon is not None and policy_func is not None:  # For ε-greedy policy
                action_probs = np.ones(action_space_n) * epsilon / action_space_n
                best_action = policy_func(state)
                action_probs[best_action] += (1.0 - epsilon)
                action = np.random.choice(np.arange(action_space_n), p=action_probs)
                prob = action_probs[action]
            else:  # For random policy
                action = np.random.choice(action_space_n)
                prob = 1.0 / action_space_n

            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            episode_probs.append(prob)
            state = next_state

        episodes.append(episode)
        behavior_probabilities.append(episode_probs)

    return episodes, behavior_probabilities


# ============================ Plotting Functions =====================================
def plot_policy(Q, shape, title):
    """
    Plots the policy derived from Q-values.

    Args:
    - Q (dict): A dictionary mapping states to arrays of action values.
    - shape (tuple): The shape of the gridworld, e.g., (rows, cols).
    """
    policy = np.array([np.argmax(Q.get(state, np.ones(4))) for state in range(np.prod(shape))]).reshape(shape)
    fig, ax = plt.subplots()
    ax.matshow(policy, cmap='coolwarm')
    for i in range(policy.shape[0]):
        for j in range(policy.shape[1]):
            ax.text(j, i, ['←', '↓', '→', '↑'][policy[i, j]], ha='center', va='center', color='white')
    plt.title(f'Policy Visualization {title}')
    plt.xlabel('State X')
    plt.ylabel('State Y')
    plt.show()


def plot_value_function(V, shape, title):
    value_function = np.zeros(shape)
    for state, value in V.items():
        x, y = state  
        value_function[x, y] = value  
    fig, ax = plt.subplots()
    im = ax.matshow(value_function, cmap='viridis')
    for i in range(value_function.shape[0]):
        for j in range(value_function.shape[1]):
            ax.text(j, i, f'{value_function[i, j]:.2f}', ha='center', va='center', color='white')
    plt.title(f'Value Function {title}')
    plt.colorbar(im)
    plt.xlabel('State X')
    plt.ylabel('State Y')
    plt.show()


if __name__ == "__main__":
    env = FourRoomsEnv(goal_pos=(10, 10))  # Initialize the environment
    

    random_policy = create_random_policy(env.action_space.n)
    episodes_random, probabilities_random = generate_episodes_with_probabilities(env, random_policy, num_episodes=100)
    
    Q_epsilon_greedy, on_policy = on_policy_mc_control(env, num_episodes=100)
    epsilon_greedy_policy = create_epsilon_greedy_policy(Q_epsilon_greedy, epsilon=0.1, action_space_n=env.action_space.n)
    episodes_epsilon_greedy, probabilities_epsilon_greedy = generate_episodes_with_probabilities(env, epsilon_greedy_policy, epsilon=0.1, num_episodes=100)
    
    Q_off_random = off_policy_mc_prediction(episodes_random, probabilities_random)
    Q_off_epsilon_greedy = off_policy_mc_prediction(episodes_epsilon_greedy, probabilities_epsilon_greedy)
    

    greedy_policy_random = compute_greedy_policy(Q_off_random)
    greedy_policy_epsilon_greedy = compute_greedy_policy(Q_off_epsilon_greedy)

    env_shape = (11,11) 
    
    # Plotting
    plot_policy(greedy_policy_random, env_shape, 'Off policy pi_random')
    plot_policy(greedy_policy_epsilon_greedy, env_shape, 'Off policy pi_greedy')

    V_random = {s: np.max(Q_off_random[s]) for s in Q_off_random}
    V_epsilon_greedy = {s: np.max(Q_off_epsilon_greedy[s]) for s in Q_off_epsilon_greedy}
    
    plot_value_function(V_random, env_shape, 'Off policy pi_random')
    plot_value_function(V_epsilon_greedy, env_shape, 'Off policy pi_greedy')

    # on policy
    greedy_policy_on = compute_greedy_policy(Q_epsilon_greedy)
    plot_policy(greedy_policy_on, env_shape, 'On policy pi_greedy')

    V_epsilon_greedy_on = {s: np.max(Q_epsilon_greedy[s]) for s in Q_epsilon_greedy}
    plot_value_function(V_epsilon_greedy_on, env_shape, 'On policy pi_greedy')
