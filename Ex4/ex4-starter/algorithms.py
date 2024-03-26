import gym
from typing import Callable, Tuple
from collections import defaultdict
from tqdm import trange
import numpy as np
from policy import create_blackjack_policy, create_epsilon_policy

def standardize_state_info(state):
    if isinstance(state, tuple) and isinstance(state[0], tuple):
        standardized_state = state[0]  
    elif isinstance(state, tuple):
        standardized_state = state
    else:
        raise ValueError("Unexpected state format returned by environment")
    return standardized_state

def generate_episode(env: gym.Env, policy: Callable, es: bool = False):
    """A function to generate one episode and collect the sequence of (s, a, r) tuples

    This function will be useful for implementing the MC methods

    Args:
        env (gym.Env): a Gym API compatible environment
        policy (Callable): A function that represents the policy.
        es (bool): Whether to use exploring starts or not
    """
    episode = []
    obtained_state = env.reset()
    # Account for dict appearing in the state information
    state = standardize_state_info(obtained_state)

    while True:
        if es and len(episode) == 0:
            action = env.action_space.sample()
        else:
            action = policy(state)
        observed_next_state, reward, done, _ = env.step(action)
        next_state = standardize_state_info(observed_next_state)

        episode.append((state, action, reward))
        if done:
            break
        state = next_state

    return episode


def on_policy_mc_evaluation(
    env: gym.Env,
    policy: Callable,
    num_episodes: int,
    gamma: float,
) -> defaultdict:
    """On-policy Monte Carlo policy evaluation. First visits will be used.

    Args:
        env (gym.Env): a Gym API compatible environment
        policy (Callable): A function that represents the policy.
        num_episodes (int): Number of episodes
        gamma (float): Discount factor of MDP

    Returns:
        V (defaultdict): The values for each state. V[state] = value.
    """
    # We use defaultdicts here for both V and N for convenience. The states will be the keys.
    V = defaultdict(float)
    N = defaultdict(int)

    for _ in trange(num_episodes, desc="Episode"):
        episode = generate_episode(env, policy)

        G = 0
        visited_states = set()
        for t in range(len(episode) - 1, -1, -1):
            # TODO Q3a
            state, action, reward = episode[t]
            G = gamma * G + reward

            # Update V and N here according to first visit MC
            if state not in visited_states:
                N[state] += 1 
                V[state] += (G - V[state]) / N[state] # Incremental formula to update value
                visited_states.add(state)

    return V


def on_policy_mc_control_es(
    env: gym.Env, num_episodes: int, gamma: float
) -> Tuple[defaultdict, Callable]:
    """On-policy Monte Carlo control with exploring starts for Blackjack

    Args:
        env (gym.Env): a Gym API compatible environment
        num_episodes (int): Number of episodes
        gamma (float): Discount factor of MDP
    """
    # We use defaultdicts here for both Q and N for convenience. The states will be the keys and the values will be numpy arrays with length = num actions
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    N = defaultdict(lambda: np.zeros(env.action_space.n))

    # If the state was seen, use the greedy action using Q values.
    # Else, default to the original policy of sticking to 20 or 21.
    policy = create_blackjack_policy(Q)

    for _ in trange(num_episodes, desc="Episode"):
        # TODO Q3b
        G = 0 # initialize the expected return
        observed_state = env.reset() # exploration at the beginning of each episode
        state = standardize_state_info(observed_state)
        initial_action = env.action_space.sample() # Ensure all actions are selected with probability > 0 by choosing an action randomly for the first step
        episode = [(state, initial_action, 0)] # dummy reward for first action

        # Generate an episode following the policy after the first step
        state, action, reward = episode[0]
        while True:
            observed_next_state, reward, done, _, _ = env.step(action)
            next_state = standardize_state_info(observed_next_state)
            episode.append((next_state, policy(next_state), reward)) if not done else episode.append((next_state, None, reward))
            if done:
                break
            state = next_state
            action = policy(state)
        # Note there is no need to update the policy here directly.
        # By updating Q, the policy will automatically be updated.
        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            G = gamma * G + reward  # Update return
            if not any(s == state and a == action for s, a, _ in episode[:t]):
                N[state][action] += 1
                Q[state][action] += (G - Q[state][action]) / N[state][action]

        # Update the policy to be greedy with respect to the current Q
        policy = create_blackjack_policy(Q)
    return Q, policy

# def on_policy_mc_control_epsilon_soft(env, num_episodes, discount_factor=1.0, epsilon=0.1):
#     """
#     Implements the Monte Carlo Control for Epsilon-Soft policies.
    
#     Args:
#         env: The environment.
#         num_episodes: Number of episodes to run for training.
#         discount_factor: Gamma discount factor.
#         epsilon: The probability of choosing a random action, float between 0 and 1.
        
#     Returns:
#         A tuple (Q, policy, episode_rewards) where
#          - Q is a dictionary mapping state -> action values.
#          - policy is a function that takes an observation as argument and returns action probabilities.
#          - episode_rewards is a list of total rewards received for each episode.
#     """
    
#     # Initializes the action-value function
#     Q = defaultdict(lambda: np.zeros(env.action_space.n))
#     returns_sum = defaultdict(float)
#     returns_count = defaultdict(int)
#     episode_rewards = np.zeros(num_episodes)  # Track rewards for each episode

#     # The policy we're following
#     def policy(observation):
#         A = np.ones(env.action_space.n, dtype=float) * epsilon / env.action_space.n
#         best_action = np.argmax(Q[observation])
#         A[best_action] += (1.0 - epsilon)
#         return A

#     for i_episode in range(num_episodes):
#         # Generate an episode.
#         episode = []
#         state = env.reset()
#         total_reward = 0  # Initialize total reward for the episode
#         for t in range(100):  # Limit the number of steps per episode
#             probs = policy(state)
#             action = np.random.choice(np.arange(len(probs)), p=probs)
#             next_state, reward, done, _ = env.step(action)
#             episode.append((state, action, reward))
#             total_reward += reward
#             if done:
#                 break
#             state = next_state

#         episode_rewards[i_episode] = total_reward  # Store total reward for this episode
        
#         sa_in_episode = set([(tuple(x[0]), x[1]) for x in episode])
#         for state, action in sa_in_episode:
#             sa_pair = (state, action)
#             first_occurrence_idx = next(i for i,x in enumerate(episode) if x[0] == state and x[1] == action)
#             G = sum([x[2]*(discount_factor**i) for i,x in enumerate(episode[first_occurrence_idx:])])
#             returns_sum[sa_pair] += G
#             returns_count[sa_pair] += 1
#             Q[state][action] = returns_sum[sa_pair] / returns_count[sa_pair]

#     def final_policy(observation):
#         return np.argmax(Q[observation])
    
#     return Q, final_policy, episode_rewards
def make_epsilon_greedy_policy(Q, epsilon, nA):
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn

def on_policy_mc_control_epsilon_soft(env, num_episodes, discount_factor=1.0, epsilon=0.1):
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    returns_sum = defaultdict(float)
    returns_count = defaultdict(int)
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

    episode_rewards = np.zeros(num_episodes)  # To track rewards for plotting

    for i_episode in range(num_episodes):
        episode = []
        state = env.reset()
        for t in range(100):  # Assuming 100 is the max episode length
            action_probs = policy(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            if done:
                break
            state = next_state
        
        states_in_episode = set([tuple(x[0]) for x in episode])
        for state in states_in_episode:
            state_action_rewards = [x[2] for x in episode if x[0] == state]
            for action in range(env.action_space.n):
                sa_pair = (state, action)
                sa_rewards = [sum(state_action_rewards[i:]) for i, x in enumerate(episode) if x[0] == state and x[1] == action]
                if sa_rewards:
                    returns_sum[sa_pair] += sum(sa_rewards)
                    returns_count[sa_pair] += len(sa_rewards)
                    Q[state][action] = returns_sum[sa_pair] / returns_count[sa_pair]

        # Update rewards for plotting
        episode_rewards[i_episode] = sum([x[2] for x in episode])
    
    return Q, policy, episode_rewards