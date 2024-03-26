import numpy as np
import matplotlib.pyplot as plt

def random_walk(state, n_states, left_reward=-1):
    """Returns the next state and reward given the current state, with modified left-side outcome."""
    if np.random.rand() < 0.5:
        next_state = state + 1 
    else:
        next_state = state - 1 
    
    if next_state == 0:
        return next_state, left_reward 
    elif next_state == n_states + 1:
        return next_state, 1 
    else:
        return next_state, 0 

def run_td_0(V, alpha, gamma, episodes, states, n_states, left_reward=-1):
    for episode in range(episodes):
        state = np.random.choice(states)  
        while state not in [0, n_states + 1]:  
            next_state, reward = random_walk(state, n_states, left_reward)
            V[state] += alpha * (reward + gamma * V[next_state] - V[state])
            state = next_state
    return V

def value_estimates_for_different_walks():
    for n_states, left_reward in [(19, -1), (3, -1), (19, 0), (3, 0)]:
        states = np.arange(1, n_states + 1)
        alpha = 0.1
        gamma = 1
        episodes_to_track = [0, 1, 10, 100, 1000]
        V_values = {}

        for ep in episodes_to_track:
            V = np.zeros(n_states + 2)
            V[1:-1] = 0.5
            V = run_td_0(V, alpha, gamma, ep, states, n_states, left_reward)
            V_values[ep] = V[1:-1]

        plt.figure(figsize=(10, 6))
        for ep, values in V_values.items():
            plt.plot(states, values, label=f'Episodes: {ep}', marker='o')
        plt.title(f'Value Estimates for {n_states} States, Left Reward: {left_reward}')
        plt.xlabel('State')
        plt.ylabel('Estimated Value')
        plt.legend()
        plt.show()

value_estimates_for_different_walks()
