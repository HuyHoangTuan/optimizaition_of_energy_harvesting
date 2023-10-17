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

NUM_EPISODES = 1400
NUM_STEP = 20
BATCH_SIZE = 128
ALPHA = 0.003
EPS_MAX = 1.
EPS_MIN = 0.01
EPS_DECAY = 0.001
GAMMA = 0.99
TAU = 0.005

steps_done = 0

env = Environment()
n_actions = env.num_action_space()
n_observations = env.num_state_space()

policy_net = DQN(n_observations , n_actions).to(device)
target_net = DQN(n_observations , n_actions).to(device)

if exists('net.pth'):
    policy_net.load_state_dict(torch.load('net.pth'))

target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.SGD(policy_net.parameters(), lr = ALPHA)
memory = ReplayMemory(10000)


num_rewards = []
max_rewards = []
num_loss = []
epsilon_values = []
SU_rewards = []
def select_action(state , t=0):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_MIN + (EPS_MAX - EPS_MIN) * math.exp(-1. * EPS_DECAY * steps_done)
    #eps_threshold = EPS_MIN + (EPS_MAX - EPS_MIN) * math.exp(-1. * EPS_DECAY * t)
    epsilon_values.append(eps_threshold)
    steps_done += 1

    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1,1)
    else :
        return torch.tensor([[env.action_sampling()]] , dtype=torch.long ,device = device)


def plot_rewards(show_result = False):
    #plt.figure(num=1,figsize=(1,3))
    plt.figure(num =1 , figsize=(1,2))
    rewards_t = torch.tensor(num_rewards , dtype=torch.float)
    #loss_t = torch.tensor(num_loss , dtype=torch.float)
    #max_t = torch.tensor(max_rewards , dtype=torch.float)
    #epsilon_t = torch.tensor(epsilon_values ,dtype=torch.float)
    SU_rewards_t = torch.tensor(SU_rewards , dtype=torch.float)
    means = []
    if len(num_rewards) >= 100:
        means = rewards_t.unfold(0 , 100 , 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99) , means))
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
        plt.subplot(121)
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        plt.plot(rewards_t.numpy())
        if len(num_rewards) >= 100:
            plt.plot(means.numpy())
        plt.subplot(122)
        plt.xlabel('Episode')
        plt.ylabel('Average Rate')
        plt.plot(SU_rewards_t.numpy())
        #plt.subplot(122)
        #plt.xlabel('Time Slot')
        #plt.ylabel('Epsilon value')
        #plt.plot(epsilon_t.numpy())
        # plt.subplot(132)
        # plt.xlabel('Each optimize step')
        # plt.ylabel('loss value')
        # plt.plot(loss_t.numpy())
        # plt.subplot(133)
        # plt.xlabel('Each episode')
        # plt.ylabel('max reward')
        # plt.plot(max_t.numpy())
    plt.pause(0.001)

count_optim = 0
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return

    transitions = memory.sample(BATCH_SIZE)

    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s : s is not None , batch.next_state)), device=device , dtype = torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    y = policy_net(state_batch)
    state_action_values = y.gather(1,action_batch)

    next_state_values = torch.zeros(BATCH_SIZE , device = device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    MSE = nn.MSELoss()
    loss = MSE(state_action_values, expected_state_action_values.unsqueeze(1))
    num_loss.append(loss.item())

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


for i_episode in range(NUM_EPISODES):
    # if(i_episode == 1400):
    #     env.TimeSlot = -1
    state = env.reset()
    state = torch.tensor(state , dtype=torch.float32 , device=device).unsqueeze(0)
    sum_reward = 0.
    max_r = 0.
    sum_rate = 0.
    for t in count():
        action = select_action(state , i_episode)
        observation , reward , done = env.step(action.item())
        reward = torch.tensor([reward] , device = device, dtype=torch.float32)
        if done:
            next_state = None
        else:
            next_state = torch.tensor(observation , dtype=torch.float32,device=device).unsqueeze(0)

        reward_item = reward.squeeze(0).item()
        sum_reward += reward_item

        sum_rate += reward_item if reward_item >= 0 else 0
        max_r = max(max_r , reward)

        state = torch.round(state , decimals=2)
        if next_state != None:
            next_state = torch.round(next_state , decimals=2)

        #print('state ' , state)
        #print('next state ' , next_state)
        memory.push(state , action , next_state , reward)
        state = next_state

        optimize_model()

        if t % 5 == 4:
            # target_net_state_dict = target_net.state_dict()
            # policy_net_state_dict = policy_net.state_dict()
            #
            # for key in policy_net_state_dict:
            #     target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)

            target_net.load_state_dict(policy_net.state_dict())

        if done:
            num_rewards.append(sum_reward / NUM_STEP)
            max_rewards.append(max_r)
            SU_rewards.append(sum_rate / NUM_STEP)
            plot_rewards()
            break

torch.save(target_net.state_dict() , 'net.pth')
print('Complete')
plot_rewards(show_result=True)
plt.show()