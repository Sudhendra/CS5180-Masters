
import random
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from copy import deepcopy
from collections import defaultdict

from env import *


def plot_curves(arr_list, legend_list, color_list, ylabel, fig_title):
    """
    Args:
        arr_list (list): list of results arrays to plot
        legend_list (list): list of legends corresponding to each result array
        color_list (list): list of color corresponding to each result array
        ylabel (string): label of the Y axis

        Note that, make sure the elements in the arr_list, legend_list and color_list are associated with each other correctly.
        Do not forget to change the ylabel for different plots.
    """
    # set the figure type
    fig, ax = plt.subplots(figsize=(12, 8))

    # PLEASE NOTE: Change the labels for different plots
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Time Steps")

    # ploth results
    h_list = []
    for arr, legend, color in zip(arr_list, legend_list, color_list):
        # compute the standard error
        arr_err = arr.std(axis=0) / np.sqrt(arr.shape[0])
        # plot the mean
        h, = ax.plot(range(arr.shape[1]), arr.mean(axis=0), color=color, label=legend)
        # plot the confidence band
        arr_err *= 1.96
        ax.fill_between(range(arr.shape[1]), arr.mean(axis=0) - arr_err, arr.mean(axis=0) + arr_err, alpha=0.3,
                        color=color)
        # save the plot handle
        h_list.append(h)

    # plot legends
    ax.set_title(f"{fig_title}")
    ax.legend(handles=h_list)

    # save the figure
    # plt.savefig(f"{fig_title}.png", dpi=200)

    plt.show()


class SemiGradientSARSAAgent(object):
    def __init__(self, env, info):
        """
        Function to initialize the semi-gradient SARSA agent
        Args:
            env: the environment that the agent interacts with
            info (dict): a dictionary variable contains all necessary parameters.

        Note that: In this question, we will implement a simple state aggregation strategies.
                   Specifically, we design the following function approximation:
                   1. Feature: we use the one-hot encoding to compute the feature for each state-action pair.
                               E.g., state = [0, 0] and action = "Up" (state-action pair) will correspond to
                               a unique one-hot representation $f(s, a) = [0, 0, 0, 1, 0, ..., 0]$.
                   2. Weights: we define a weight vector $w$ having the sample shape as the feature vector.
                               Specifically, the Q(s, a) can be estimated by Q(s, a) = w^{T} * f(s, a)

        Importantly, as described in the question, we only aggregate the states.
        """
        # Store the environment
        self.env = env

        """ Learning parameters for semi-gradient SARSA """
        # Store the number of learning episodes
        self.episode_num = info['episode_num']

        # Store the update step size alpha
        self.alpha = info['alpha']

        # Store the discount factor
        self.gamma = info['gamma']

        # Initialize the epsilon
        self.epsilon = info['epsilon']

        # Store the other hyerparameters
        self.params = info

        """ State aggregation for semi-gradient SARSA """
        # Compute the total number of state after the aggregation
        self.state_space, self.state_num = self.create_state_aggregation()

        # Compute the total number of the actions
        self.action_num = len(self.env.action_names)

        """ Function approximation for semi-gradient SARSA"""
        # We create a weight with shape |S| * |A|
        self.weights_fn = np.zeros((self.state_num * self.action_num))

        # We construct a numpy array that contains the one-hot features for all state-action pairs.
        # The size is (|S| * |A|) x (|S| * |A|).
        # Each i-th row is the one-hot encoding for state-action pair with index i.
        self.feature_arr = np.eye(self.state_num * self.action_num)

    def create_state_aggregation(self):
        """
        Function that returns the aggregated state space and the number of the aggregated states.
        """
        
        """CODE HERE: your state aggregation strategy.
           
           You have to return:
            1. the aggregated state space (Any data structure that you find it easier to render the index of the 
               aggregated state.)
            2. the number of the aggregated states (int)
        """
        
        # compute the aggregated state space based on the state aggregation strategy
        aggregated_state_space = deepcopy(self.env.observation_space)

        # compute the number of the state in the aggregated state space
        aggregate_state_num = len(aggregated_state_space)

        return aggregated_state_space, aggregate_state_num

    def _aggregate_state_idx(self, state):
        """
        Function returns the index of aggregated state given an original state

        Args:
            state (list): original state
        """
        
        """CODE HERE: based on your state aggregation, return the index of the aggregated state given an original
           state. 
           
           You have to return:
           1. index (int) of the aggregated state given the original state
        """
        state_idx = self.state_space.index(state)
        
        return state_idx

    def _aggregate_action_idx(self, action):
        """
        Function returns the index of aggregated action.
        Args:
            action (string): name of the action

        To be simple, here, one action only aggregates to itself
        """
        return self.env.action_names.index(action)

    def _get_state_action_feature(self, state, action):
        """
        Function that returns the one-hot feature given a state-action pair.

        Args:
            state (list): original state
            action (string): name of the action
        """
        # Get the unique index of the aggregated state
        state_index = self._aggregate_state_idx(state)
        # Get the unique index of the aggregated action
        action_index = self._aggregate_action_idx(action)
        # Compute the state(aggregated)-action index
        state_action_index = self.state_num * action_index + state_index
        # Get the one-hot feature of the state
        return self.feature_arr[state_action_index]

    def function_approximation(self, state, action):
        """
        Function that computes the Q value given a state-action pair using linear function approximation.
        Args:
            state (list): original state
            action (string): name of the action
        """
        state_action_feature = self._get_state_action_feature(state, action)
        return np.matmul(state_action_feature.T, self.weights_fn)

    def render_q_value(self, state, action):
        """
        Function that returns the Q value given a state-action pair

        Args:
            state (list): original state
            action (string): name of the action
        """
        return self.function_approximation(state, action)

    def epsilon_greedy_policy(self, state):
        """
        Function implements the epsilon-greedy policy
        Args:
            state (list): original state
        """
        """CODE HERE:
           implement the epsilon-greedy policy using function approximation. Break ties if happens 
           
           You should return:
           1. name of the action (string)
        """
        l = len(self.env.action_names)
        
        probs = np.ones(l) * (self.epsilon/l)
        Q = []
        
        for a in self.env.action_names:
            Q.append(self.render_q_value(state, a))
            
        A_opt = np.argmax(Q)
    
        if Q.count(Q[A_opt]) > 1:
            tie = [i for i in range(len(Q)) if Q[i] == Q[A_opt]]
            tie_idx = random.choice(tie)
            A_opt = tie_idx

        probs[A_opt] += 1 - self.epsilon
    
        action = np.random.choice(np.arange(l), p=probs)
        
        return(self.env.action_names[action])

    def update_weights(self, s, a, r, s_prime, a_prime):
        """
        Function that updates the weights using semi-gradients

        Args:
            s (list): original state
            a (string): action name
            r (float): reward
            s_prime (list): original next state
            a_prime (string): next action name
        """
        """ CODE HERE:
            implement the update of the semi-gradient SARSA
            
            You should update "self.weights_fn"
        """
        # check if s_prime is the termination state
        if s_prime == self.env.goal_location:
            # update the weights
            self.weights_fn = self.weights_fn + (self.alpha*(r - self.render_q_value(s, a)))*self._get_state_action_feature(s, a)
        else:
            self.weights_fn = self.weights_fn + (self.alpha*(r + self.gamma*self.render_q_value(s_prime, a_prime) - self.render_q_value(s, a)))*self._get_state_action_feature(s, a)

    def run(self):
        # Save the discounted return for each episode
        discounted_returns = []

        # Semi-gradient SARSA starts
        for ep in tqdm.trange(self.episode_num):
            """CODE HERE:
               Implement the pseudocode of the Semi-gradient SARSA
            """
            # Reset the agent to initial STATE at the beginning of every episode
            state, _ = self.env.reset()
            state = tuple(state)

            # Render an ACTION based on the initial STATE
            action = self.epsilon_greedy_policy(state)

            # Store rewards to compute return G for the current episode.
            reward_list = []
            
            # Loop the episode
            for t in range(self.env.max_time_steps):
                # Take the ACTION and observe REWARD and NEXT STATE
                next_state, reward, done, _, _ = self.env.step(action)
                next_state = tuple(next_state)
                
                # Given the NEXT STATE, choose the NEXT ACTION
                next_action = self.epsilon_greedy_policy(next_state)

                # Update the weights of the function using semi-gradient SARSA
                # Using STATE, ACTION, REWARD, NEXT STATE, NEXT ACTION
                self.update_weights(state, action, reward, next_state, next_action)
                
                """DO NOT CHANGE BELOW"""
                # Save the reward for plotting
                reward_list.append(reward)

                # Reset the environment
                if done:
                    break
                else:
                    state = next_state
                    action = next_action

            # compute the discounted return for the current episode
            G = 0
            for reward in reversed(reward_list):
                G = reward + self.gamma * G
            discounted_returns.append(G)

        return discounted_returns


""" Implement the Tile-based Agent here. We inherit it from the "SemiGradientSARSAAgent" above
"""
class TileAgent(SemiGradientSARSAAgent):
    def __init__(self, env, info):
        """
        Function to initialize the semi-gradient SARSA agent
        Args:
            env: the environment that the agent interacts with
            info (dict): a dictionary variable contains all necessary parameters.

        Note that: In this question, we will implement a simple state aggregation strategies.
                   Specifically, we design the following function approximation:
                   1. Feature: we use the one-hot encoding to compute the feature for each state-action pair.
                               E.g., state = [0, 0] and action = "Up" (state-action pair) will correspond to
                               a unique one-hot representation $f(s, a) = [0, 0, 0, 1, 0, ..., 0]$.
                   2. Weights: we define a weight vector $w$ having the sample shape as the feature vector.
                               Specifically, the Q(s, a) can be estimated by Q(s, a) = w^{T} * f(s, a)

        Importantly, as described in the question, we only aggregate the states.
        """
        super().__init__(env, info)
        
        """ State aggregation for semi-gradient SARSA """
        # Compute the total number of state after the aggregation
        self.state_space, self.state_num = self.create_state_aggregation()

        # Compute the total number of the actions
        self.action_num = len(self.env.action_names)

        """ Function approximation for semi-gradient SARSA"""
        # We create a weight with shape |S| * |A|
        self.weights_fn = np.zeros((self.state_num * self.action_num))

        # We construct a numpy array that contains the one-hot features for all state-action pairs.
        # The size is (|S| * |A|) x (|S| * |A|).
        # Each i-th row is the one-hot encoding for state-action pair with index i.
        self.feature_arr = np.eye(self.state_num * self.action_num)

    def create_state_aggregation(self):
        """
        Function that returns the aggregated state space and the number of the aggregated states.
        """
        
        """CODE HERE: Implement the Tile-based state aggregation here.
           Hint: you can manually discretize the original states using the Tile-based method.
           For example, you can copy the grid from the Four Rooms environment
           and manually aggregate the states (value = 0) in the grid. 
           
           You have to return:
            1. the aggregated state space (Any data structure that you find it easier to render the index of the 
               aggregated state.)
            2. the number of the aggregated states (int)
        """
        aggregated_state_space = defaultdict(list)
        #storing the tiles into the state space by running over 2 for loops, using a dictionary to store the state-space with keys being the indices
        idx = 0

        for i in range(0,11,2):
            for j in range(0,11,2):
                if(i==10 and j==10):
                    aggregated_state_space[idx] = [(i,j)]
                elif(j==10):
                    aggregated_state_space[idx] = [(i,j), (i+1,j)]
                elif(i==10):
                    aggregated_state_space[idx] = [(i,j), (i,j+1)]
                else:
                    neighbours = [(i,j), (i,j+1), (i+1,j), (i+1,j+1)]
                    aggregated_state_space[idx] = neighbours
                idx += 1

        # aggregate the state space
        aggregate_state_num = len(aggregated_state_space.keys())

        return aggregated_state_space, aggregate_state_num

    def _aggregate_state_idx(self, state):
        """
        Function returns the index of aggregated state given an original state

        Args:
            state (list): original state
        """
        """CODE HERE: based on your state aggregation, return the index of the aggregated state given an original
           state. 
           
           You have to return:
           1. index (int) of the aggregated state given the original state
        """
        
        # render the index of the aggregated state
        for i in self.state_space.keys():
            if state in self.state_space[i]:
                state_idx = i
        return state_idx



""" Implement the Room-based Agent here. We inherit it from the "SemiGradientSARSAAgent" above
"""
class RoomAgent(SemiGradientSARSAAgent):
    def __init__(self, env, info):
        """
        Function to initialize the semi-gradient SARSA agent
        Args:
            env: the environment that the agent interacts with
            info (dict): a dictionary variable contains all necessary parameters.

        Note that: In this question, we will implement a simple state aggregation strategies.
                   Specifically, we design the following function approximation:
                   1. Feature: we use the one-hot encoding to compute the feature for each state-action pair.
                               E.g., state = [0, 0] and action = "Up" (state-action pair) will correspond to
                               a unique one-hot representation $f(s, a) = [0, 0, 0, 1, 0, ..., 0]$.
                   2. Weights: we define a weight vector $w$ having the sample shape as the feature vector.
                               Specifically, the Q(s, a) can be estimated by Q(s, a) = w^{T} * f(s, a)

        Importantly, as described in the question, we only aggregate the states.
        """
        super().__init__(env, info)
        
        """ State aggregation for semi-gradient SARSA """
        # Compute the total number of state after the aggregation, using a dictionary to store the state-space with keys being the indices
        self.state_space, self.state_num = self.create_state_aggregation()

        # Compute the total number of the actions
        self.action_num = len(self.env.action_names)

        """ Function approximation for semi-gradient SARSA"""
        # We create a weight with shape |S| * |A|
        self.weights_fn = np.zeros((self.state_num * self.action_num))

        # We construct a numpy array that contains the one-hot features for all state-action pairs.
        # The size is (|S| * |A|) x (|S| * |A|).
        # Each i-th row is the one-hot encoding for state-action pair with index i.
        self.feature_arr = np.eye(self.state_num * self.action_num)

    def create_state_aggregation(self):
        """
        Function that returns the aggregated state space and the number of the aggregated states.
        """
        """CODE HERE: your state aggregation strategy. Hint: you can start with a simple state aggregation
           that just aggregate each state to itself. In other words, the aggregated state space is just
           the original state space.
           
           You have to return:
            1. the aggregated state space (Any data structure that you find it easier to render the index of the 
               aggregated state.)
            2. the number of the aggregated states (int)
        """
        aggregated_state_space = defaultdict(list)
        
        #storing the rooms into the state space by running over 2 for loops and clubbing the entire set of grids
        neighbours1, neighbours2, neighbours3, neighbours4 = [],[],[],[]

        for i in range(0,11):
            for j in range(0,11):
                if(i<6 and j<6):
                    neighbours1.append((i,j))
                elif(i<6 and j>6):
                    neighbours2.append((i,j))
                elif(i>6 and j<6):
                    neighbours3.append((i,j))
                else:
                    neighbours4.append((i,j))
        
        aggregated_state_space[0] = neighbours1
        aggregated_state_space[1] = neighbours2
        aggregated_state_space[2] = neighbours3
        aggregated_state_space[3] = neighbours4
            
        aggregate_state_num = len(aggregated_state_space.keys())
        
        return aggregated_state_space, aggregate_state_num

    def _aggregate_state_idx(self, state):
        """
        Function returns the index of aggregated state given an original state

        Args:
            state (list): original state
        """
        """CODE HERE: based on your state aggregation, return the index of the aggregated state given an original
           state. 
           
           You have to return:
           1. index (int) of the aggregated state given the original state
        """
        # render the index of the aggregated state
        for i in self.state_space.keys():
            if state in self.state_space[i]:
                state_idx = i
        
        return state_idx


import math
class RBFAgent(SemiGradientSARSAAgent):
    def __init__(self, env, info):
        super().__init__(env, info)
        
        """ State aggregation for semi-gradient SARSA """
        # Compute the total number of state after the aggregation
        self.state_space, self.state_num = self.create_state_aggregation()

        # Compute the total number of the actions
        self.action_num = len(self.env.action_names)

        """ Function approximation for semi-gradient SARSA"""
        # We create a weight with shape |S| * |A|
        self.weights_fn = np.zeros((self.state_num * self.action_num))

        # We construct a numpy array that contains the one-hot features for all state-action pairs.
        # The size is (|S| * |A|) x (|S| * |A|).
        # Each i-th row is the one-hot encoding for state-action pair with index i.
        self.feature_arr = np.eye(self.state_num * self.action_num)

    def create_state_aggregation(self):
        """
        Create the aggregated state space and the number of aggregated states using RBF Coarse Coding method.
        """
        # Define the number of radial basis functions (RBFs)
        num_rbfs = 5
        
        # Initialize a dictionary to store the RBF-based state space with keys being the indices
        rbf_based_state_space = defaultdict(list)
        
        # Compute the centers and widths of the RBFs
        centers_x = np.linspace(0, self.env.observation_space[0][1], num_rbfs)
        centers_y = np.linspace(0, self.env.observation_space[0][0], num_rbfs)
        width_x = (self.env.observation_space[0][1] - 0) / (num_rbfs - 1)
        width_y = (self.env.observation_space[0][0] - 0) / (num_rbfs - 1)
        
        # Iterate over all states and assign them to RBFs based on their distances to the centers
        for state in self.env.observation_space:
            # Compute the distances to the centers of the RBFs
            distances_x = np.abs(state[1] - centers_x)
            distances_y = np.abs(state[0] - centers_y)
            
            # Compute the activations of the RBFs
            activations_x = np.exp(-0.5 * (distances_x / (width_x + 1e-8)) ** 2)
            activations_y = np.exp(-0.5 * (distances_y / (width_y + 1e-8)) ** 2)
            
            # Compute the combined activations
            combined_activations = np.outer(activations_x, activations_y).flatten()
            
            # Add the state to the corresponding RBF
            for idx, activation in enumerate(combined_activations):
                rbf_based_state_space[idx].append(state)
        
        # Compute the number of RBFs
        num_rbfs_total = len(rbf_based_state_space.keys())
        
        return rbf_based_state_space, num_rbfs_total
    
    def _aggregate_state_idx(self, state):
        """
        Return the index of the aggregated state given an original state using RBF Coarse Coding method.

        Args:
            state (list): Original state.

        Returns:
            int: Index of the aggregated state given the original state.
        """
        # Find the index of the RBF containing the original state
        for i, states in self.state_space.items():
            if state in states:
                state_idx = i
        
        return state_idx
    


class CoordinateFeatureSemiGradientSARSAAgent(SemiGradientSARSAAgent):
    def __init__(self, env, info):
        super().__init__(env, info)

    def _get_state_action_feature(self, state, action):
        """
        Override the method to compute the feature vector using the (x, y) coordinates of the agent's location.
        Args:
            state (list): original state
            action (string): name of the action
        """
        # Extract x and y coordinates from the state
        x, y = state

        # Create the feature vector with (x, y, 1)
        return np.array([x, y, 1])

    def function_approximation(self, state, action):
        """
        Override the method to compute the Q value using linear function approximation.
        Args:
            state (list): original state
            action (string): name of the action
        """
        state_action_feature = self._get_state_action_feature(state, action)

        print(state_action_feature)

        # Reshape the weights to match the dimensions of the feature vector

        print(self.weights_fn)
        weights = self.weights_fn.reshape(len(state_action_feature), -1)
        return np.dot(state_action_feature, weights)



if __name__ == "__main__":
    # set the random seed
    np.random.seed(1234)
    random.seed(1234)

    # set hyper-parameters
    params = {
        "episode_num": 100,
        "alpha": 0.1,
        "gamma": 0.99,
        "epsilon": 0.1,
        "bin_radius":0.1
    }

    # set running trials. You can try run_trial = 5 for debugging
    run_trial = 10

    # # run multiple trials
    # results = []
    # for _ in range(run_trial):
    #     # create the environment
    #     my_env = FourRooms()

    #     # run semi-gradient SARSA
    #     my_agent = SemiGradientSARSAAgent(my_env, params)
    #     res = my_agent.run()

    #     # save result for each running trial
    #     results.append(np.array(res))


    # plot_curves([np.array(results)], ["semi-gradient SARSA"], ["b"],
    #         "Averaged discounted return", "Q3 - (a): semi-gradient SARSA")
    


    # run experiment for the Tile-based method
    results_tile = []
    for _ in range(run_trial):        
        # create the environment
        my_env = FourRooms()

        # run semi-gradient SARSA with Tile-based method with tile size n = 2
        my_agent = TileAgent(my_env, params)
        res = my_agent.run()

        # save result for each running trial
        results_tile.append(np.array(res))



    # run experiment for the Room-based method
    results_room = []
    for _ in range(run_trial):        
        # create the environment
        my_env = FourRooms()

        # run semi-gradient SARSA with Room-based method
        my_agent = RoomAgent(my_env, params)
        res = my_agent.run()

        # save result for each running trial
        results_room.append(np.array(res))





    # run experiment for the Room-based method
    results_rbg = []
    for _ in range(run_trial):        
        # create the environment
        my_env = FourRooms()

        # run semi-gradient SARSA with Room-based method
        my_agent = RBFAgent(my_env, params)
        res = my_agent.run()

        # save result for each running trial
        results_rbg.append(np.array(res))

    plot_curves([np.array(results_tile), np.array(results_room), np.array(results_rbg)],
                ["semi-gradient SARSA with tile aggregation", "semi-gradient SARSA with room aggregation", "semi-gradient SARSA with rbg aggregation"],
                ["b", "g", "r"],
                "Averaged discounted return", "Q3 - (b): semi-gradient SARSA with aggregation")
    

