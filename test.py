import math
import torch
from Environment import Environment
import torch.optim as optim
from DQN import Transition, ReplayMemory
from DQN import DQN
import random
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib
from itertools import count
from os.path import exists

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# env = Environment()
# n_actions = env.num_action_space()
# n_observations = env.num_state_space()
#
# model = DQN(n_observations , n_actions).to(device)
#
# model.load_state_dict(torch.load('net.pth'))
#
# NUM_EPISODES = 1400
#
# reward_plt = []
# def plot_rewards(show_results = False):
#     plt.figure(1)
#     reward_plt_t = torch.tensor(reward_plt , dtype=torch.float)
#
#
#     if show_results:
#         plt.title('result')
#     else:
#         plt.clf()
#         plt.xlabel('Episodes')
#         plt.ylabel('Sum reward')
#         plt.plot(reward_plt_t.numpy())
#         if len(reward_plt) >= 100:
#             means = reward_plt_t.unfold(0 , 100 , 1).mean(1).view(-1)
#             means = torch.cat((torch.zeros(99) , means))
#             plt.plot(means.numpy())
#
#     plt.pause(0.001)
#
# for i_episode in range(NUM_EPISODES):
#     state = env.reset()
#     state = torch.tensor(state , dtype=torch.float32 , device = device).unsqueeze(0)
#     sum_reward = 0.
#     for t in count():
#         with torch.no_grad():
#             action = model(state).max(1)[1].view(1,1)
#         k , P = env.map_action(action)
#         observation , reward, done = env.step(action.item())
#
#         if done:
#             next_state = None
#         else :
#             next_state = torch.tensor(observation , dtype=torch.float32 , device = device).unsqueeze(0)
#             # next_state = torch.round(next_state , decimals=2)
#
#         state = next_state
#         sum_reward += reward
#         if done:
#             reward_plt.append(sum_reward)
#             plot_rewards()
#             break
#
# plot_rewards(True)
# plt.show()

