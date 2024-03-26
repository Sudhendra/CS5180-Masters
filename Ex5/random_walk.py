import numpy as np
import matplotlib.pyplot as plt


# Random Walk Function
def random_walk(state, n_states):
    """Returns the next state and reward given the current state."""
    if np.random.rand() < 0.5:
        next_state = state + 1  # Move right
    else:
        next_state = state - 1  # Move left
    
    if next_state == 0:
        return next_state, 0  # Terminal state on the left
    elif next_state == n_states + 1:
        return next_state, 1  # Terminal state on the right
    else:
        return next_state, 0  # Non-terminal states
    
# Function to run the random walk and update V using TD(0)
def run_td_0(V, alpha, gamma, episodes, states, n_states):
    for episode in range(episodes):
        state = np.random.choice(states)  # Start from a random non-terminal state
        while state not in [0, n_states + 1]:  # Until terminal state is reached
            next_state, reward = random_walk(state, n_states)
            V[state] += alpha * (reward + gamma * V[next_state] - V[state])
            state = next_state
    return V

def value_estimates():
    # Parameters for TD(0)
    n_states = 5  # Number of non-terminal states
    states = np.arange(1, n_states + 1)  # States 1 through 5
    alpha = 0.1  # Step-size parameter
    gamma = 1  # Discount rate
    episodes_to_track = [0, 1, 10, 100, 1000]  # Episodes at which to track and plot V
    # Initialize V for each tracking point and run TD(0)
    V_values = {}
    for ep in episodes_to_track:
        # Reset V to initial values for each run
        V = np.zeros(n_states + 2)  # Value estimates, including terminal states at 0 and 6
        V[1:-1] = 0.5  # Initialize non-terminal states to 0.5
        
        # Run TD(0)
        V = run_td_0(V, alpha, gamma, ep, states, n_states)
        V_values[ep] = V[1:-1]  # Store the non-terminal states' values after ep episodes

    # True values for V
    true_values = [1/6, 2/6, 3/6, 4/6, 5/6]

    # State labels
    state_labels = ['A', 'B', 'C', 'D', 'E']

    # Plotting
    plt.figure(figsize=(10, 6))
    for ep, values in V_values.items():
        plt.plot(state_labels, values, label=f'Estimated value ({ep} episodes)', marker='o')

    # Plot the true values
    plt.plot(state_labels, true_values, label='True values', marker='o', color='red', linewidth=2)

    # Adding plot title and legend
    plt.title('Estimated Value vs. State for Different Episodes')
    plt.xlabel('State')
    plt.ylabel('Estimated Value')
    plt.legend()

    # Show the plot
    plt.show()


# Function to calculate RMS error
def rms_error(V, true_V):
    return np.sqrt(np.mean((V - true_V)**2))

# TD(0) Learning Function
def td_learning(alpha, gamma, n_episodes, true_values, states, n_states):
    V = np.full(n_states + 2, 0.5)  # Initialize all state values to 0.5
    V[0] = 0  # Set terminal state values to 0
    V[-1] = 0
    errors = np.zeros(n_episodes)
    for episode in range(n_episodes):
        state = np.random.choice(states)  # Start from a random non-terminal state
        while state not in [0, n_states + 1]:  # Until terminal state is reached
            next_state, reward = random_walk(state)
            V[state] += alpha * (reward + gamma * V[next_state] - V[state])
            state = next_state
        errors[episode] = rms_error(V[1:-1], true_values)
    return errors

# Monte Carlo Learning Function
def mc_learning(alpha, gamma, n_episodes, true_values, states, n_states):
    V = np.full(n_states + 2, 0.5)  # Initialize all state values to 0.5
    V[0] = 0  # Set terminal state values to 0
    V[-1] = 0
    errors = np.zeros(n_episodes)
    for episode in range(n_episodes):
        # Generate an episode
        states_visited = []
        rewards = []
        state = np.random.choice(states)
        while state not in [0, n_states + 1]:
            states_visited.append(state)
            next_state, reward = random_walk(state)
            rewards.append(reward)
            state = next_state
        G = 0
        for state, reward in zip(reversed(states_visited), reversed(rewards)):
            G = gamma * G + reward  # Update the return
            # For every visit MC, update the state value with the average return
            V[state] = V[state] + alpha * (G - V[state])
        errors[episode] = rms_error(V[1:-1], true_values)
    return errors

def rmse_estimates():
    # Initialize the parameters for the simulation
    n_states = 5  # Number of non-terminal states
    states = np.arange(1, n_states + 1)  # States 1 through 5
    true_values = np.array([1/6, 2/6, 3/6, 4/6, 5/6])  # True values of the states
    gamma = 1  # Discount rate
    n_runs = 100  # Number of runs to average over
    n_episodes = 100  # Number of episodes for each run
    alpha_list_td = [0.15, 0.1, 0.05]  # Alphas for TD
    alpha_list_mc = [0.01, 0.02, 0.03, 0.04]  # Alphas for MC
    # Initialize error accumulators
    td_errors = np.zeros((len(alpha_list_td), n_episodes))
    mc_errors = np.zeros((len(alpha_list_mc), n_episodes))

    # Run simulations
    for i, alpha in enumerate(alpha_list_td):
        for run in range(n_runs):
            td_errors[i] += td_learning(alpha, gamma, n_episodes, true_values, states, n_states)
        td_errors[i] /= n_runs

    for i, alpha in enumerate(alpha_list_mc):
        for run in range(n_runs):
            mc_errors[i] += mc_learning(alpha, gamma, n_episodes, true_values, states, n_states)
        mc_errors[i] /= n_runs

    # Plotting
    plt.figure(figsize=(10, 8))
    episodes = np.arange(1, n_episodes + 1)
    for i, alpha in enumerate(alpha_list_td):
        plt.plot(episodes, td_errors[i], label=f'TD α={alpha}')
    for i, alpha in enumerate(alpha_list_mc):
        plt.plot(episodes, mc_errors[i], label=f'MC α={alpha}', linestyle='--')

    plt.title('Empirical RMS error, averaged over states and runs')
    plt.xlabel('Episodes')
    plt.ylabel('RMS error')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    value_estimates()
    rmse_estimates()
