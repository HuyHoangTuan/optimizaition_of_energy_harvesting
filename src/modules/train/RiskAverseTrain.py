import math
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
import matplotlib.pyplot as plt
from itertools import count

from utils import LogUtils
from utils import RandomUtils
from modules.environment import Environment
from modules.model import DQNModel, QLearning
from modules.relay import ReplayMemory, Transition

class RiskAverseTrain:
    def __init__(
            self,
            episodes = 1600,
            eps_max = 1,
            eps_min = 0.01,
            eps_decay = 0.001,
            alpha = 0.003,
            batch_size = 64
    ):

        self._episodes = episodes
        self._eps_max = eps_max
        self._eps_min = eps_min
        self._eps_decay = eps_decay
        self._eps_drop_rate = 0.0
        self._alpha = alpha
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._batch_size = batch_size
        self._init_env()

    def _select_action(self, net, state):
        sample = RandomUtils.custom_random()
        eps_threshold = self._eps_min + (self._eps_max - self._eps_min) * math.exp(
            -1. * self._eps_drop_rate * self._eps_decay
        )
        if sample > eps_threshold:
            with torch.no_grad():
                return torch.argmax(
                    net(state)
                )
        else:
            _ = [s for s in range(self._env.get_num_actions())]
            return torch.tensor(RandomUtils.sample(_, 1), dtype = torch.long, device = self._device)

    def _init_env(self):
        self._env = Environment(
            Episode = self._episodes
        )

        n_actions = self._env.get_num_actions()
        n_observations = self._env.get_num_states()

        self._memory = ReplayMemory(10000)

        self._num_Q_Functions = 2
        self._policy_net = QLearning(n_observations, n_actions, self._num_Q_Functions)
        self._target_net = QLearning(n_observations, n_actions, self._num_Q_Functions)

    def _optimize_model(self, state, action, next_state, reward):
        self._memory.push(state, action, next_state, reward)

        if len(self._memory) < self._batch_size:
            return 0

        transitions = self._memory.sample(self._batch_size)
        # batch =

        pass

    def start_train(self):
        LogUtils.info('TRAIN_RISK_AVERSE', f'CUDA: {torch.cuda.is_available()}')

        for i_episode in range(self._episodes):
            state, _ = self._env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=self._device)

            for t in count():
                action = self._select_action(state)
                observation, (k, P, Rho), (reward, reward_type), time_slot = self._env.step(action.item(), i_episode)

                done = True if time_slot >= self._env.N else False

                if done == True:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=self._device)

                loss = self._optimize_model(state, action, next_state, reward)
