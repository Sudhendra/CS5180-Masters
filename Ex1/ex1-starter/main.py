from env import BanditEnv, NonStationaryBanditEnv, BanditEnvWithSwitchingCost
from agent import EpsilonGreedy, UCB, argmax
from tqdm import trange
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def q4(k: int, num_samples: int):
    """Q4

    Structure:
        1. Create multi-armed bandit env
        2. Pull each arm `num_samples` times and record the rewards
        3. Plot the rewards (e.g. violinplot, stripplot)

    Args:
        k (int): Number of arms in bandit environment
        num_samples (int): number of samples to take for each arm
    """

    env = BanditEnv(k=k)
    env.reset()

    # Initialize rewards for each arm
    rewards_for_strip_plot = []
    rewards_for_violin_plot = [[] for arm in range(k)]
    arms = []

    for arm in range(k):
        for pull in range(num_samples):
            rewards_for_violin_plot[arm].append(env.step(arm))
            rewards_for_strip_plot.append(env.step(arm))
            arms.append(arm)

    # Plot the rewards
    # Strip Plot
    plt.figure(figsize=(10, 6))
    sns.stripplot(x=arms, y=rewards_for_strip_plot, jitter=True)
    plt.title("Rewards Distribution for Each Arm")
    plt.xlabel("Arm")
    plt.ylabel("Reward")
    plt.savefig("q4stripplot.png")

    # Violin Plot
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=rewards_for_violin_plot)
    plt.title("Rewards Distribution for Each Arm")
    plt.xlabel("Arm")
    plt.ylabel("Reward")
    plt.savefig("q4violinplot.png")

def q6(k: int, trials: int, steps: int):
    """Q6

    Implement epsilon greedy bandit agents with an initial estimate of 0

    Args:
        k (int): number of arms in bandit environment
        trials (int): number of trials
        steps (int): total number of steps for each trial
    """
    # initialize env and agents here
    env = BanditEnv(k=k)

    epsilon_values = [0, 0.01, 0.1]
    agents = [EpsilonGreedy(k=k, init=0, epsilon=epsilon) for epsilon in epsilon_values]

    rewards = np.zeros((trials, steps, len(agents)))
    optimal_action_counts = np.zeros((steps, len(agents)))

    upper_bound_values = []

    # Loop over trials
    for t in trange(trials, desc="Trials"):
        # Reset environment and agents after every trial
        env.reset()
        optimal_action = np.argmax(env.means) # the index of the Qmax value i.e., the optimal arm to pull in all k=10 arms (sampled from the normal distribution of expected rewards).
        upper_bound = np.max(env.means) # Max opf the expected average rewards.
        upper_bound_values.append(upper_bound)
        for each_agent in agents: 
            each_agent.reset()

        # Learning
        for step in range(steps):
            for i, agent in enumerate(agents):
                action = agent.choose_action()
                reward = env.step(action)
                agent.update(action, reward)
                rewards[t, step, i] = reward
                if action == optimal_action:
                    optimal_action_counts[step, i] += 1

    average_rewards = np.mean(rewards, axis=0)
    std_errors = np.std(rewards, axis=0) / np.sqrt(trials)
    percentage_optimal_action = (optimal_action_counts / trials) * 100

    # Plotting
    plt.figure(figsize=(12, 8))
    for i, epsilon in enumerate(epsilon_values):
        plt.plot(average_rewards[:, i], label=f'ε = {epsilon}')
        plt.fill_between(range(steps), 
                         average_rewards[:, i] - 1.96 * std_errors[:, i], 
                         average_rewards[:, i] + 1.96 * std_errors[:, i], alpha=0.2)
    plt.axhline(y=np.mean(upper_bound_values), color='r', linestyle='--', label='Upper Bound')
    plt.xlabel('Steps')
    plt.ylabel('Average Reward')
    plt.title('Average Rewards vs Steps for Each Epsilon-Greedy Agent')
    plt.legend()
    plt.savefig("q6averagerewardsplot.png")

    # Plotting % optimal action
    plt.figure(figsize=(12, 8))
    for i, agent in enumerate(agents):
        label = f'ε = {agent.epsilon}, q={agent.init}'
        plt.plot(percentage_optimal_action[:, i], label=label)
    plt.xlabel('Steps')
    plt.ylabel('% Optimal Action')
    plt.title(f'% Optimal Action vs Steps for Each Epsilon-Greedy Agent')
    plt.legend()
    plt.savefig(f"q6optimal_action_plot.png")


def q7(k: int, trials: int, steps: int, fig: int):
    """Q7

    Compare epsilon greedy bandit agents and UCB agents

    Args:
        k (int): number of arms in bandit environment
        trials (int): number of trials
        steps (int): total number of steps for each trial
    """
    # initialize env and agents here
    env = BanditEnv(k=k)

    if fig == 0:
        experiment = "Each Epsilon-Greedy Agent"
        hyperparameters = [(5,0), (0,0.1), (5,0.1), (0,0)]
        agents = [EpsilonGreedy(k=k, init=q, epsilon=epsilon) for q, epsilon in hyperparameters]
    if fig == 1:
        experiment = "Epsilon Greedy Agent & UCB agent"
        agents = [EpsilonGreedy(k=k, init=0, epsilon=0.1),  # Example ε value
                  UCB(k=k, init=0, c=2, step_size=1)]  # Example c value for UCB

    rewards = np.zeros((trials, steps, len(agents)))
    optimal_action_counts = np.zeros((steps, len(agents)))

    upper_bound_values = []

    # Loop over trials
    for t in trange(trials, desc="Trials"):
        # Reset environment and agents after every trial
        env.reset()
        optimal_action = np.argmax(env.means) # the index of the Qmax value i.e., the optimal arm to pull in all k=10 arms (sampled from the normal distribution of expected rewards).
        upper_bound = np.max(env.means) # Max opf the expected average rewards.
        upper_bound_values.append(upper_bound)

        for each_agent in agents: 
            each_agent.reset()

        # Learning
        for step in range(steps):
            for i, agent in enumerate(agents):
                action = agent.choose_action()
                reward = env.step(action)
                agent.update(action, reward)
                rewards[t, step, i] = reward
                if action == optimal_action:
                    optimal_action_counts[step, i] += 1

    average_rewards = np.mean(rewards, axis=0)
    std_errors = np.std(rewards, axis=0) / np.sqrt(trials)
    percentage_optimal_action = (optimal_action_counts / trials) * 100

    # Plotting
    plt.figure(figsize=(12, 8))
    if fig == 0:
        for i, (q, epsilon) in enumerate(hyperparameters):
            plt.plot(average_rewards[:, i], label=f'ε = {epsilon}, q={q}')
            plt.fill_between(range(steps), 
                            average_rewards[:, i] - 1.96 * std_errors[:, i], 
                            average_rewards[:, i] + 1.96 * std_errors[:, i], alpha=0.2)
    elif fig == 1:
        for i, agent in enumerate(agents):
            plt.plot(average_rewards[:, i], label=type(agent).__name__)
            plt.fill_between(range(steps), 
                            average_rewards[:, i] - 1.96 * std_errors[:, i], 
                            average_rewards[:, i] + 1.96 * std_errors[:, i], alpha=0.2)
            
    plt.axhline(y=np.mean(upper_bound_values), color='r', linestyle='--', label='Upper Bound')
    plt.xlabel('Steps')
    plt.ylabel('Average Reward')
    plt.title(f'Average Rewards vs Steps for {experiment}')
    plt.legend()
    plt.savefig(f"q7fig{fig}plot.png")

    # Plotting % optimal action
    plt.figure(figsize=(12, 8))
    for i, agent in enumerate(agents):
        label = f'ε = {agent.epsilon}, q={agent.init}' if fig == 0 else type(agent).__name__
        plt.plot(percentage_optimal_action[:, i], label=label)
    plt.xlabel('Steps')
    plt.ylabel('% Optimal Action')
    plt.title(f'% Optimal Action vs Steps for {experiment}')
    plt.legend()
    plt.savefig(f"q7fig{fig}optimal_action_plot.png")

# extra credit - Q8
def q8(k: int, trials: int, steps: int):
    """Q8

    Implement epsilon greedy bandit agents with a non stationary Bandit env.

    Args:
        k (int): number of arms in bandit environment
        trials (int): number of trials
        steps (int): total number of steps for each trial
    """
    env = NonStationaryBanditEnv(k=k)

    agents = [EpsilonGreedy(k=k, init=0, epsilon=0.1), EpsilonGreedy(k=k, init=0, epsilon=0.1, step_size=0.1)]

    rewards = np.zeros((trials, steps, len(agents)))
    optimal_action_counts = np.zeros((steps, len(agents)))
    max_mean_rewards = np.zeros((trials, steps))

    # Loop over trials
    for t in trange(trials, desc="Trials"):
        # Reset environment and agents after every trial
        env.reset()
        optimal_action = np.argmax(env.means) # the index of the Qmax value i.e., the optimal arm to pull in all k=10 arms (sampled from the normal distribution of expected rewards).
        for each_agent in agents: 
            each_agent.reset()

        # Learning
        for step in range(steps):
            upper_bound = np.max(env.means) # Max of the expected average rewards.
            max_mean_rewards[t, step] = upper_bound
            for i, agent in enumerate(agents):
                action = agent.choose_action()
                reward = env.step(action)
                agent.update(action, reward)
                rewards[t, step, i] = reward
                if action == optimal_action:
                    optimal_action_counts[step, i] += 1
                    
    average_rewards = np.mean(rewards, axis=0)
    std_errors = np.std(rewards, axis=0) / np.sqrt(trials)
    percentage_optimal_action = (optimal_action_counts / trials) * 100
    average_upper_bound = np.mean(max_mean_rewards, axis=0)

    # Plotting
    plt.figure(figsize=(12, 8))
    for i, agent in enumerate(agents):
        plt.plot(average_rewards[:, i], label=f'ε = {agent.epsilon}, α = {agent.step_size}')
        plt.fill_between(range(steps), 
                         average_rewards[:, i] - 1.96 * std_errors[:, i], 
                         average_rewards[:, i] + 1.96 * std_errors[:, i], alpha=0.2)
    plt.axhline(y=np.mean(average_upper_bound), color='r', linestyle='--', label='Upper Bound')
    plt.xlabel('Steps')
    plt.ylabel('Average Reward')
    plt.title('Average Rewards vs Steps for Each Epsilon-Greedy Agent in a Non stationary environment')
    plt.legend()
    plt.savefig("q8averagerewardsplot.png")

    # Plotting % optimal action
    plt.figure(figsize=(12, 8))
    for i, agent in enumerate(agents):
        label = f'ε = {agent.epsilon}, α = {agent.step_size}'
        plt.plot(percentage_optimal_action[:, i], label=label)
    plt.xlabel('Steps')
    plt.ylabel('% Optimal Action')
    plt.title(f'% Optimal Action vs Steps for Each Epsilon-Greedy Agent')
    plt.legend()
    plt.savefig(f"q8optimal_action_plot.png")

# Extra credit - Q9
def q9(k: int, trials: int, steps: int, switch_cost: float):
    """Q9

    Implement Epsilon Greedy & UCB agents where the agent incurs a cost for switchinga arms.

    Args:
        k (int): number of arms in bandit environment
        trials (int): number of trials
        steps (int): total number of steps for each trial
    """
    env = BanditEnvWithSwitchingCost(k=k, switch_cost=switch_cost)
    agents = [EpsilonGreedy(k=k, init=5, epsilon=0.01), UCB(k=k, init=5, c=2, step_size=1)]

    rewards = np.zeros((trials, steps, len(agents))) # track all rewards over all the trials
    switches = np.zeros((trials, steps, len(agents))) # track all instances the agent decides to switch from one arm to another.
    optimal_action_counts = np.zeros((steps, len(agents)))
    upper_bound_values = []

    # Loop over trials
    for t in trange(trials, desc="Trials"):
        # Reset environment and agents after every trial
        env.reset()
        optimal_action = np.argmax(env.means)
        upper_bound = np.max(env.means)
        upper_bound_values.append(upper_bound)
        for each_agent in agents: 
            each_agent.reset()

        last_action = None
        
        for step in range(steps):
            for i, agent in enumerate(agents):
                action = agent.choose_action()
                reward = env.step(action)
                agent.update(action, reward)
                rewards[t, step, i] = reward

                if action == optimal_action:
                    optimal_action_counts[step, i] += 1

                if last_action is not None and last_action != action:
                    switches[t, step, i] = 1
                last_action = action

    average_rewards = np.mean(rewards, axis=0)
    std_errors = np.std(rewards, axis=0) / np.sqrt(trials)
    percentage_optimal_action = (optimal_action_counts / trials) * 100
    switch_percentage = np.mean(switches, axis=0) * 100

    # Plotting
    # Average Rewards
    plt.figure(figsize=(12, 8))
    for i, agent in enumerate(agents):
        if type(agent).__name__ == "EpsilonGreedy":
            label = f'Agent = {type(agent).__name__} ε = {agent.epsilon}, q = {agent.init}, α = {agent.step_size}'
        elif type(agent).__name__ == "UCB":
            label=f'Agent = {type(agent).__name__} c = {agent.c}, q = {agent.init}, α = {agent.step_size}'

        plt.plot(average_rewards[:, i], label=label)
        plt.fill_between(range(steps), 
                         average_rewards[:, i] - 1.96 * std_errors[:, i], 
                         average_rewards[:, i] + 1.96 * std_errors[:, i], alpha=0.2)
    plt.axhline(y=np.mean(upper_bound_values), color='r', linestyle='--', label='Upper Bound')
    plt.xlabel('Steps')
    plt.ylabel('Average Reward')
    plt.title(f'Average Rewards vs Steps with Switching Cost = {switch_cost}')
    plt.legend()
    plt.savefig("q9averagerewardsplot.png")

    # Plotting switching percentage
    plt.figure(figsize=(12, 8))
    for i, agent in enumerate(agents):
        plt.plot(switch_percentage[:, i], label=type(agent).__name__)
    plt.xlabel('Steps')
    plt.ylabel('Switching Percentage')
    plt.title(f'Percentage of Switching Arms vs Steps with Switching Cost = {switch_cost}')
    plt.legend()
    plt.savefig("q9switchingPercentageplot.png")

    # Plotting % optimal action
    plt.figure(figsize=(12, 8))
    for i, agent in enumerate(agents):
        plt.plot(percentage_optimal_action[:, i], label=type(agent).__name__)
    plt.xlabel('Steps')
    plt.ylabel('% Optimal Action')
    plt.title(f'% Optimal Action vs Steps with Switching Cost = {switch_cost}')
    plt.legend()
    plt.savefig(f"q9optimalActionplot.png")

def main():
    # q4(10, 10)
    # q6(10, 2000, 10000)
    # q7(10, 2000, 10000, fig=0) # reproducing fig 2.3 with requested modificatons
    # q7(10, 2000, 10000, fig=1) # reproducing fig 2.4 with requested modfications
    q8(10, 2000, 10000) # Epsilon Greedy Agents in a Non Stationary environment
    q9(10, 2000, 10000, switch_cost=0.1) # Epsilon Greedy & UCB agents in a Bandit env with a switching cost

if __name__ == "__main__":
    main()
