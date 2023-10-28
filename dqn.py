# Date: 8/4/2023
# Author: Xubin Zhang
# Description: This file contains the implementation of...
import pandas as pd
import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from env_dqn import rp_env
from way_info import way, reset_df
from global_var import initial_data_p, initial_data_ch, data_p, data_ch

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sys
import os

if len(sys.argv) > 1:
    try_numbers = int(sys.argv[1])
else:
    print("No value for try_numbers provided.")
    sys.exit(1)


# try_numbers = 74 #test
# load_weights_path =f"/home/utlck/PycharmProjects/Tunning_results/weights_{(try_numbers - 1):03d}.pth"
# load_weights_path =f"/home/utlck/PycharmProjects/Tunning_results/weights_047_901.pth"
# load_weights_path ="/home/utlck/PycharmProjects/route-planning/weights_044.pth"


# original_stdout = sys.stdout
# with open(f"output_{try_numbers:03d}.txt", 'w') as file:
#     sys.stdout = file

if torch.cuda.is_available():
    num_episodes = 500
else:
    num_episodes = 100

env = rp_env()
env.w_distance = 6000  # value range -1~+1
env.w_energy = 1000  # -6~1
env.w_driving = 10  # -100~0 , 1
env.w_charge = 10  # -250~0 , 1
env.w_parking = 1  # -100~0
env.w_target = 0  # 1 or 0
# env.w_loop = 0 # 1 or -1000
env.w_power = 0 # 1 0.5 0.1 -1
w_num_charges = 0  # number of charges
env.w_end_soc = 0  # 1/soc

theway = way()
# theway.n_ch = 6  # Number of nearest charging station
# theway.n_p = 4  # Number of nearest parking lots
# theway.n_pois = 10

steps_max = 500
# REPLAYBUFFER = 10000
REPLAYBUFFER = 10000
# result_path = f"{try_numbers:03d}.png"
# weights_path = f"weights_{try_numbers:03d}.pth"
# folder_path = r'/home/utlck/PycharmProjects/Tunning_results'
## Linux
# result_path = os.path.join(folder_path, f"{try_numbers:03d}.png")
# weights_path = os.path.join(folder_path, f"weights_{try_numbers:03d}.pth")
result_path = f"/home/utlck/PycharmProjects/Tunning_results/{try_numbers:03d}.png"
result_path_step = f"/home/utlck/PycharmProjects/Tunning_results/{try_numbers:03d}_step.png"
## windows
# result_path = f"{folder_path}\\{try_numbers:03d}.png"
# weights_path = f"{folder_path}\\weights_{try_numbers:03d}.pth"

BATCH_SIZE = 128  # BATCH_SIZE is the number of transitions sampled from the replay buffer
GAMMA = 0.99  # GAMMA is the discount factor as mentioned in the previous section
EPS_START = 0.9  # EPS_START is the starting value of epsilon
EPS_END = 0.1  # EPS_END is the final value of epsilon
EPS_DECAY = 9618  # EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# EPS_END = 0.05  # EPS_END is the final value of epsilon
# EPS_DECAY = 10000  # EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
TAU = 0.005  # TAU is the update rate of the target network
LR = 1e-4  # LR is the learning rate of the ``AdamW`` optimizer


SGD = False
Adam = True
AdamW = False

SmoothL1Loss = True
MSE = False
MAE = False


#Use a cyclic buffer of bounded size that holds the transitions observed recently.
class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Structure of DQN
class DQN(nn.Module):

    #Q-Network with 2 hidden layers, 128 neurons
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, BATCH_SIZE)
        self.layer2 = nn.Linear(BATCH_SIZE, BATCH_SIZE)
        self.layer3 = nn.Linear(BATCH_SIZE, n_actions)

    # Forward propagation with ReLU
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


##Training Phase
# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transition: A named tuple representing a single transition in an environment
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Get number of actions from gym action space
n_actions = env.df_actions.shape[0]
# Get the number of state observations
state, info = env.reset()
n_observations = len(state)

policy_net = DQN(n_observations, n_actions).to(device)

# checkpoint = torch.load(load_weights_path)
# # print(checkpoint)
# policy_net.load_state_dict(checkpoint)

target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

if AdamW:
    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
if Adam:
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
if SGD:
    optimizer = optim.SGD(policy_net.parameters(), lr=LR)

memory = ReplayMemory(REPLAYBUFFER)

steps_done = 0

print("Number of episodes = ", num_episodes, "\n")
print("Batchsize, Gamma, EPS_start, EPS_end, EPS_decay, TAU, LR, Replaybuffer, actions, oberservations = ", BATCH_SIZE, GAMMA, EPS_START, EPS_END, EPS_DECAY, TAU, LR, REPLAYBUFFER, n_actions, n_observations, "\n")


# Select action by Epsilon-Greedy Policy according to state
def select_action(state, eps_flag):
    global steps_done, data_p, data_ch
    sample = random.random() # range 0~1

    #Epsilon-Greedy Policy
    # Start with threshold=0.9,exploits most of the time with a small chance of exploring.
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1 # The value of threshold decreases, increasing the chance of exploration

    # # In the last episode, there is only exploitation
    # if eps_flag == num_episodes - 10:
    #    eps_threshold == 0.05
    # if eps_flag == num_episodes - 1:
    #    eps_threshold == 0

    if sample >= eps_threshold:
        # Exploitation, chooses the greedy action to get the most reward
        # by exploiting the agent’s current action-value estimates
        with torch.no_grad():
            # Use Q-network to calculate the max. Q-value
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            print("Exploitation, chooses the greedy action to get the most reward")
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        # Exploration, sample from the action space randomly
        # print(torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long))
        print("Exploration, sample from the action space randomly")
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

average_rewards = [] # A list that keeps track of the average reward of each episode for analysis after training is complete.
# Plot the average reward of an episodes
def plot_average_reward():
    average_rewards_t = torch.tensor(average_rewards, dtype=torch.float)
    plt.figure(figsize=(10, 6))
    plt.xlabel('Episode')
    plt.ylabel('Average Reward per Episode')
    # plt.plot(average_rewards_t.numpy())
    plt.plot(range(len(average_rewards_t)), average_rewards_t.numpy(), linestyle='-')

    plt.grid()
    plt.savefig(result_path)
    plt.close()

step_reward = []
def plot_step_reward():
    plt.figure(figsize=(10, 6))
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.plot(range(len(step_reward)), step_reward, linestyle='-')

    plt.grid()
    plt.savefig(result_path_step)
    plt.close()

# A single step of the optimization
def optimize_model():
    #Determine whether resampling is required
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)  # random sampling
    # First sample a batch of data from the memory, the size is BATCH_SIZE
    # Each data contains a state, an action, a reward and a next state
    # Unpack this tuple into four single lists, state, action, reward, and next state
    batch = Transition(*zip(*transitions))
    # print(batch)

    # Check states that in batch, create a boolean mask that identifies which states are non-final
    # Final state: False
    # Non-final state: Ture
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    # Select next_state that is non-final state
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)  # All states in batch
    action_batch = torch.cat(batch.action)  # All actions
    reward_batch = torch.cat(batch.reward)  # All rewards

    # Compute Q(s_t, a) by policy_net, then select the columns of actions taken.
    # These are the actions which would've been taken for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states by target-net.
    # Expected values of actions for non_final_next_states are computed based on the "older" target_net;
    # selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad(): # target_net
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0] # max. state-action value
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    # The Huber loss acts like the mean squared error when the error is small,
    # but like the mean absolute error when the error is large
    # This makes it more robust to outliers when the estimates of Q are very noisy.
    if SmoothL1Loss:
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    if MSE:
        criterion = nn.MSELoss() ## Mean Squared Error, MSE
        loss = criterion(state_action_values.float(), expected_state_action_values.unsqueeze(1).float())
    if MAE:
        criterion = nn.L1Loss() ##Mean Absolute Error, MAE
        loss = criterion(state_action_values.float(), expected_state_action_values.unsqueeze(1).float())

    # Optimize the model
    optimizer.zero_grad() #The gradient needs to be cleared before updating the parameters each time,
    # so as not to affect the next update due to the superposition of gradient information
    loss.backward() # Back-propagation: According to the previously calculated loss value,
    # the gradient is calculated by the chain rule, which is used to update the model parameters
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100) #Clips gradient to ensure that the absolute value
    # of the gradient does not exceed 100, used to prevent the gradient explosion problem
    optimizer.step() # update weights


## Main Training Loop
for i_episode in range(num_episodes):
    # Initialize the reward for the number of chanrging
    r_num_charges = 0
    # Initialize the sum_reward in an episode
    sum_reward = 0
    # Initialize the environment and get it's state
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    print("state_reset = ", state, "\n")
    # # clear loop_pois.csv
    # env.clear_loop_file()

    for t in range(steps_max):
        action = select_action(state, i_episode)
        observation, reward, terminated = env.step(action) # observation is next state
        # in current step arrival target
        node_current, index_current, soc, t_stay, t_secd_current, t_secp_current, t_secch_current = observation
        if int(index_current) == theway.closest_index_ch or int(index_current) == theway.closest_index_p:
            terminated = True
        # print("observation, reward, terminated = ", observation, reward, terminated, "\n")

        if 0 <= node_current < 6 and t_stay != 0:
            r_num_charges += 1
        reward = reward + (r_num_charges * w_num_charges)
        print(f"r_num_charges = {r_num_charges * w_num_charges}")

        step_reward.append(reward)
        sum_reward = sum_reward + reward
        reward = torch.tensor([reward], device=device)
        # done = terminated

        if terminated:
            next_state = None # Stop Episode
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)
        print("state, action, next_state, reward = ", state, action, next_state, reward, "\n")

        # Move to the next state
        state = next_state

        # Perform one step of the optimization, just on the policy_net
        optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        print(f"###Episode {i_episode} step {t} done ###")

        if t == steps_max - 1:
            terminated = True
            print(f"Terminated: Can not arrival target after {steps_max} steps, stop the episode\n")

        if terminated: ## episode done
            # episode_durations.append(t + 1)
            # plot_durations()
            print("Number of steps in an episode:", t+1)
            print("Sum reward:", sum_reward)
            average_reward = sum_reward / (t+1)
            print("Average reward:", average_reward)
            average_rewards.append(average_reward)
            if (i_episode + 1) % 100 == 0:
                weights_path = f"/home/utlck/PycharmProjects/Tunning_results/weights_{try_numbers:03d}_{int(i_episode + 1)}epis.pth"
                torch.save(policy_net.state_dict(), weights_path)
                plot_average_reward()
                plot_step_reward()
            # reset data_ch, data_p
            reset_df()
            print(f"**************************************Episode {i_episode}done**************************************\n")
            break

print("average_rewards:", average_rewards)
# torch.save(policy_net.state_dict(), weights_path)
plot_average_reward()
plot_step_reward()
print('Complete')

# sys.stdout = original_stdout
