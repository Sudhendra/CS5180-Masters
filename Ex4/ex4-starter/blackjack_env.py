import gym
import numpy as np
import matplotlib.pyplot as plt

from algorithms import on_policy_mc_evaluation, on_policy_mc_control_es
from policy import default_blackjack_policy

def Q3b(n_episodes, discount_factor, env):
    # TODO: Task B: Q3 part b
    Q, optimal_policy = on_policy_mc_control_es(env, n_episodes, discount_factor)
    return Q, optimal_policy

def Q3a(n_episodes, discount_fcator, env):
    # Task A: Use stick only on 20 or 21 policy to policy and evaluate the first-visit-MC
    state_value_function = on_policy_mc_evaluation(env, default_blackjack_policy, n_episodes, discount_fcator)
    return state_value_function

#  plotting functions
def plot_state_value_function(V, filename, title="State Value Function"):
    player_sum = np.arange(11, 22)
    dealer_show = np.arange(1, 11)
    usable_ace = [False, True]
    
    fig, axes = plt.subplots(nrows=2, figsize=(10, 10), constrained_layout=True)
    fig.suptitle(title, fontsize=16)
    
    for i, ace in enumerate(usable_ace):
        state_values = np.zeros((len(player_sum), len(dealer_show)))
        for j, player in enumerate(player_sum):
            for k, dealer in enumerate(dealer_show):
                state_values[j, k] = V.get((player, dealer, ace), 0)
        
        im = axes[i].imshow(state_values, cmap='coolwarm', extent=[0.5, 10.5, 21.5, 11.5, 10])
        axes[i].set_title(f"Usable Ace: {ace}")
        axes[i].set_xlabel("Dealer Showing")
        axes[i].set_ylabel("Player Sum")
        axes[i].set_xticks(range(1, 11))
        axes[i].set_yticks(range(11, 22))
        axes[i].invert_yaxis()

    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.45, label='State Value')
    plt.draw()
    plt.savefig(filename)

def plot_policy(Q, filename, title="Optimal Policy"):
    policy = {state: np.argmax(actions) for state, actions in Q.items()}
    
    player_sum = np.arange(11, 22)
    dealer_show = np.arange(1, 11)
    usable_ace = [False, True]
    
    fig, axes = plt.subplots(nrows=2, figsize=(10, 10), constrained_layout=True)
    fig.suptitle(title, fontsize=16)
    
    for i, ace in enumerate(usable_ace):
        policy_grid = np.zeros((len(player_sum), len(dealer_show)))
        for j, player in enumerate(player_sum):
            for k, dealer in enumerate(dealer_show):
                policy_grid[j, k] = policy.get((player, dealer, ace), 0)
        
        axes[i].matshow(policy_grid, cmap='coolwarm', extent=[0.5, 10.5, 21.5, 11.5, 10])
        axes[i].set_title(f"Usable Ace: {ace}")
        axes[i].set_xlabel("Dealer Showing")
        axes[i].set_ylabel("Player Sum")
        axes[i].set_xticks(range(1, 11))
        axes[i].set_yticks(range(11, 22))
        axes[i].invert_yaxis()

    plt.draw()
    plt.savefig(filename)

def plot_optimal_value_function(Q, filename, title="Optimal Value Function"):
    player_sum = np.arange(11, 22)
    dealer_show = np.arange(1, 11)
    usable_ace = [False, True]
    
    fig, axes = plt.subplots(nrows=2, figsize=(10, 10), constrained_layout=True)
    fig.suptitle(title, fontsize=16)
    
    for i, ace in enumerate(usable_ace):
        value_function = np.zeros((len(player_sum), len(dealer_show)))
        for j, player in enumerate(player_sum):
            for k, dealer in enumerate(dealer_show):
                state = (player, dealer, ace)
                value_function[j, k] = max(Q[state]) if state in Q else 0  # Take max over all actions
        
        im = axes[i].imshow(value_function, cmap='coolwarm', extent=[0.5, 10.5, 21.5, 11.5, 10], aspect='auto')
        axes[i].set_title(f"Usable Ace: {ace}")
        axes[i].set_xlabel("Dealer Showing")
        axes[i].set_ylabel("Player Sum")
        axes[i].set_xticks(range(1, 11))
        axes[i].set_yticks(range(11, 22))
        axes[i].invert_yaxis()  # Invert Y-axis
        
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.45, label='State Value')
    plt.draw()
    plt.savefig(filename)


if __name__ == "__main__":
    # define the environment
    blackjack_env = gym.make('Blackjack-v1', natural=False, sab=False)
    gamma = 1.0

    # ================== Q3 part a ============================= 
    # 10,000 episodes
    outfile = "q3a_1.png"
    state_value_function_10k = Q3a(10000, gamma, blackjack_env)
    # plot_state_value_function(state_value_function_10k, outfile)
    # 500,000 episodes
    outfile = "q3a_2.png"
    # state_value_function_500k = Q3a(500000, gamma, blackjack_env)
    # plot_state_value_function(state_value_function_500k, outfile)

    # ================== Q3 part b   ============================= 
    outfile = "q3b_1.png"
    # Q_optimal, optimal_policy = Q3b(1000000, gamma, blackjack_env)
    # plot_policy(Q_optimal, outfile)
    outfile = "q3b_2.png"
    # plot_optimal_value_function(Q_optimal, outfile)

    # ================== Q2 part c   ============================= 
    # Run Monte Carlo Control and collect data
    # num_episodes = 500000
    # Q, average_rewards = mc_control_collect_data(blackjack_env, default_blackjack_policy, num_episodes, gamma)

    # # Plot the learning curve
    # plot_learning_curve(average_rewards, "learning_curve.png", "Learning Progress over Episodes")


