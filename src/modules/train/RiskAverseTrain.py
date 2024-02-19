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
            batch_size = 64,
            beta = -0.5,
            risk_control_parameter = 0.5
    ):

        self._episodes = episodes
        self._eps_max = eps_max
        self._eps_min = eps_min
        self._eps_decay = eps_decay
        self._eps_drop_rate = 0.0
        self._lambdaP = risk_control_parameter
        self._alpha = alpha
        self._beta = beta
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

        # self._memory = ReplayMemory(10000)

        self._num_Q_Functions = 2
        self._Q = QLearning(n_observations, n_actions, self._num_Q_Functions)
        self._Q_hat = QLearning(n_observations, n_actions, 1)

    def _utility_func(self, value):
        return -1.0 * math.exp(self._beta * value)

    def _optimize_model(self, state, action, next_state, reward):
        M = RandomUtils.poisson(Lambda = 1.0, size = self._num_Q_Functions)
        for i in range(self._num_Q_Functions):
            if M[i] == 1:
                Q = self._Q.get_Q(i, state, action)

                max_Q_next = -math.inf
                for action in range(self._env.get_num_actions):
                    max_Q_next = max(max_Q_next, self._Q.get_Q(i, next_state, action))

                x0 = -1
                learning_rate = self._Q.get_alpha(i, state, action)

                value = Q + learning_rate * (self._utility_func(reward + max_Q_next - Q) - x0)
                self._Q.set_Q(i, state, action, value)


        return 0

    def _update_Q_Hat(self, H, state):
        for action in range(self._env.get_num_actions()):
            # calc average Q
            Q_minus = 0
            for i in range(self._num_Q_Functions):
                Q_minus = Q_minus + self._Q.get(i, state, action)
            Q_minus = Q_minus / self._num_Q_Functions

            # calc standard deviation
            standard_deviation = 0
            for i in range(self._num_Q_Functions):
                standard_deviation = standard_deviation + (self._Q.get(i, state, action) - Q_minus) ** 2
            standard_deviation = standard_deviation / (self._num_Q_Functions - 1)

            # update Q_hat
            Q = self._Q.get_Q(H, state, action)
            self._Q_hat.set(0, state, action, Q - self._lambdaP * standard_deviation)
            
        return 0

    def start_train(self):
        # LogUtils.info('TRAIN_RISK_AVERSE', f'CUDA: {torch.cuda.is_available()}')

        for i_episode in range(self._episodes):
            state, _ = self._env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=self._device)

            for t in count():
                # update Q Hat
                H = math.floor(RandomUtils.custom_random() * self._Q.num_models)
                self._update_Q_Hat(H, state)

                # select action according to Q Hat
                action = self._select_action(self._Q_hat, state)

                # update Q Function
                observation, (k, P, Rho), (reward, reward_type), time_slot = self._env.step(action.item(), i_episode)
                done = True if time_slot >= self._env.N else False
                next_state = None

                if done is not True:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=self._device)

                loss = self._optimize_model(state, action, next_state, reward)


                if done:
                    break
