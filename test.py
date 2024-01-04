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
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



TimeSlot = 20
N_Episodes = 2000

def plot_rewards(show_result = False):
    plt.figure(num=1)
    #rewards_t = torch.tensor(sum_rewards , dtype=torch.float)
    #epsilon_t = torch.tensor(epsilon_values ,dtype=torch.float)
    #P_action_t = torch.tensor(P_action , dtype = torch.float)
    #Rho_action_t = torch.tensor(Rho_action , dtype=torch.float)
    Foul_bins = [0,1,2,3,4]

    rate_means = []
    reward_means = []
    Foul_means = []

    y = 0.
    for idx in range(len(sum_rewards)):
        y += sum_rewards[idx]
        leng = min(100 , (idx + 1))
        if(idx + 1 > 100):
            y -= sum_rewards[idx - 100]
        reward_means.append(y / leng)

    y = 0.
    for idx in range(len(SU_rates)):
        y += SU_rates[idx]
        leng = min(100, (idx + 1))
        if (idx + 1 > 100):
            y -= SU_rates[idx - 100]
        rate_means.append(y / leng)

    y = 0.
    for idx in range(len(Foul_count)):
        y += Foul_count[idx]
        leng = min(100 , (idx + 1))
        if(idx + 1 > 100):
            y -= Foul_count[idx - 100]
        Foul_means.append(y / leng)

    rate_means = np.array(rate_means)
    reward_means = np.array(reward_means)
    Foul_means = np.array(Foul_means)

    if show_result:
        plt.suptitle('Result')
    else:
        plt.clf()
        plt.suptitle('Training...')
        plt.subplot(231)
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        plt.plot(sum_rewards)
        # if len(sum_rewards) >= 100:
        #     plt.plot(means.numpy())
        plt.plot(reward_means)
        plt.subplot(232)
        plt.xlabel('Episode')
        plt.ylabel('Average Rate')
        plt.plot(SU_rates)
        plt.plot(rate_means)
        plt.subplot(233)
        plt.xlabel('Time Slot')
        plt.ylabel('Epsilon value')
        plt.plot(epsilon_values)
        plt.subplot(234)
        plt.xlabel('P value')
        plt.hist(P_action, bins=10)
        plt.subplot(235)
        plt.xlabel('Rho value')
        plt.hist(Rho_action , bins=11)


        #plt.subplot(236)
        #plt.title('Foul statistics')
        #plt.hist( Foul_bins[:-1] , Foul_bins ,weights=Foul_hist)


        plt.subplot(236)
        plt.title('Foul per episode')
        plt.xlabel('episode')
        plt.ylabel('num foul')
        plt.plot(Foul_count)
        plt.plot(Foul_means)
        plt.tight_layout()

        # plt.subplot(132)
        # plt.xlabel('Each optimize step')
        # plt.ylabel('loss value')
        # plt.plot(loss_t.numpy())
        # plt.subplot(133)
        # plt.xlabel('Each episode')
        # plt.ylabel('max reward')
        # plt.plot(max_t.numpy())
    plt.pause(0.0001)
    #plt.show()

SU_rates_loader = np.loadtxt('rate_values.txt', delimiter= ' ',dtype=float)
sum_rewards_loader = np.loadtxt('reward_values.txt' , delimiter=' ',dtype=float)
Foul_hist_loader = np.loadtxt('Foul_hist_values.txt', delimiter=' ',dtype=float)
epsilon_values_loader = np.loadtxt('epsilon_values.txt', delimiter= ' ',dtype=float)
Rho_action_loader = np.loadtxt('Rho_values.txt', delimiter= ' ',dtype=float)
P_action_loader = np.loadtxt('P_values.txt' , delimiter = ' ',dtype=float)
Foul_per_episode_loader = np.loadtxt('Foul_per_episode_values.txt' , delimiter=' ' , dtype=float)

SU_rates=[]
sum_rewards=[]
Foul_hist=[]
epsilon_values=[]
Rho_action=[]
P_action=[]
Foul_count = []
for i in range(N_Episodes):
    if(i % 25 != 0 and i != N_Episodes - 1):
        continue
    Foul_count = Foul_per_episode_loader[:i+1]
    SU_rates = SU_rates_loader[:i+1]
    sum_rewards=sum_rewards_loader[:i+1]
    Foul_hist=Foul_hist_loader[i]
    epsilon_values=epsilon_values_loader[:(i+1) * TimeSlot]
    Rho_action=Rho_action_loader[:(i+1)*TimeSlot]
    P_action=P_action_loader[:(i+1)*TimeSlot]
    plot_rewards()
plot_rewards(show_result=True)
plt.show()
