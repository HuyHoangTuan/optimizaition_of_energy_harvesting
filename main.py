from Environment import Environment
from DQN import DQN
import torch
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


NUM_EPISODES = 2000

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = Environment()
    u = env.map_to_exponential_function(0)
    v = env.map_to_exponential_function(1.5)
    v = env.map_to_exponential_function(1.3)
    n_actions = env.num_action_space()
    n_observations = env.num_state_space()

    model = DQN(n_observations, n_actions).to(device)

    model.load_state_dict(torch.load('net.pth'))
    model.eval()

    for i_episode in range(NUM_EPISODES):
        state = env.reset()

        g_s = 10.
        g_p1r = 10.
        g_p2r = 10.
        g_p1s = 10.
        g_p2s = 10.
        g_sp1 = 10.
        g_sp2 = 10.

        (E_prev, C, P1, P2, *g) = state
        g = np.array([g_s, g_sp1, g_sp2, g_p1s, g_p2s, g_p1r, g_p2r])
        state = (E_prev, C, P1, P2, *g)

        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        for t in count():

            with torch.no_grad():
                action = model(state).max(1)[1].view(1, 1)

            k, Rho, P = env.map_action(action.item())

            observation, reward, done = env.step(action.item())

            (E_prev, C, P1, P2, *g_prev, Foul, Foul_cnt, Rate) = observation

            g_prev = np.array(g_prev)
            g = g + g_prev
            if (t == 0):
                g = g_prev
            observation = (E_prev, C, P1, P2, *(g / (t + 1)))

            reward = torch.tensor([reward], device=device, dtype=torch.float32)
            if done:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            reward_item = reward.squeeze(0).item()
            state = next_state

            if done:
                break