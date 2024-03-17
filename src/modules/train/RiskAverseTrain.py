import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
import matplotlib.pyplot as plt
from itertools import count

from utils import LogUtils
from utils import RandomUtils
from utils import Parser

from modules.environment import Environment
from modules.model import QLearning

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display


class RiskAverseTrain:
    def __init__(
            self,
            episodes=1600,
            eps_max=1,
            eps_min=0.01,
            eps_decay=0.001,
            alpha=0.003,
            gamma=0.99,
            batch_size=64,
            beta=-0.5,
            risk_control_parameter=0.5,
            is_dynamic_rho=False
    ):

        self._episodes = episodes
        self._eps_max = eps_max
        self._eps_min = eps_min
        self._eps_decay = eps_decay
        self._eps_drop_rate = 0.0
        self._eps_threshold = 500
        self._lambdaP = risk_control_parameter
        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._batch_size = batch_size

        self._env = Environment(
            Episode=self._episodes,
            Dynamic_Rho=is_dynamic_rho
        )

        # visualization
        self._f = 100

        self._samples_plt = []
        self._eps_plt = []

        self._rewards_plt = []
        self._mean_rewards_plt = []

        self._episodes_plt = []
        self._mean_episodes_plt = []

        self._rates_plt = []
        self._mean_rates_plt = []

        self._td_errors_plt = []
        self._mean_td_errors_plt = []

        self._rhos_plt = []
        self._mean_rhos_plt = []

        self._init_net()

    def _init_net(self):
        self._num_Q_Functions = 2
        self._Q = QLearning(
            self._device,
            self._env.get_num_states(),
            self._env.get_num_actions(),
            self._num_Q_Functions
        )
        self._Q_hat = QLearning(
            self._device,
            self._env.get_num_states(),
            self._env.get_num_actions(),
            1
        )

    def _select_action(self, Q, state, episode=0):
        sample = RandomUtils.custom_random()
        eps_threshold = self._eps_min + (self._eps_max - self._eps_min) * math.exp(
            -1. * self._eps_drop_rate * self._eps_decay
        )
        if episode < self._eps_threshold:
            sample = 1.0
            self._eps_drop_rate = 0
        else:
            self._eps_drop_rate += 1

        self._eps_plt.append(sample)
        self._samples_plt.append(eps_threshold)

        if sample > eps_threshold:
            with torch.no_grad():
                # print(Q(state))
                return torch.argmax(
                    Q(state)
                )
        else:
            _ = [s for s in range(self._env.get_num_actions())]
            return torch.tensor(RandomUtils.sample(_, 1), dtype=torch.long, device=self._device)

    def _utility_func(self, value):
        return torch.tensor(-1.0 * math.exp(self._beta * value.item()), dtype=torch.float32, device=self._device)

    def _optimize_model(self, state, action, next_state, reward):
        TD_Errors = 0
        M = RandomUtils.poisson(Lambda=1.0, size=self._num_Q_Functions)
        for idx in range(self._num_Q_Functions):
            if M[idx] == 1:
                Q = self._Q(idx)(state)[action]
                max_Q_Next = torch.max(self._Q(idx)(next_state))
                x0 = torch.tensor(-1, dtype=torch.float32, device=self._device)
                learning_rate = self._Q(idx).get_learning_rate(state, action)

                TD_Error = self._utility_func(reward + self._gamma * max_Q_Next - Q)
                value = Q + learning_rate * (TD_Error - x0)
                TD_Errors += TD_Error.item()
                self._Q(idx)(state)[action] = value
                self._Q(idx).update_learning_rate(state, action)

        return TD_Errors / self._num_Q_Functions

    def _update_Q_Hat(self, H, state):
        Q = self._Q(H)(state)

        Q_stacked = torch.stack([self._Q(idx)(state) for idx in range(self._num_Q_Functions)])
        Q_Minus = torch.mean(Q_stacked, dim=0)

        # Calculate the standard deviation for each action
        # Note: std deviation formula is sqrt(sum((x - mean)^2) / (n-1)), but here we need the squared term
        std_deviation_squared = torch.div(
            torch.sum(torch.pow(Q_stacked - Q_Minus, 2), dim=0),
            self._num_Q_Functions - 1
        )

        self._Q_hat(0)(state)[:] = Q - self._lambdaP * std_deviation_squared

    def _plot(self, show_result=False):
        plt.figure(num=1, figsize=(16, 9), dpi=120)
        parser = None
        if not show_result:
            self._mean_rewards_plt.append(torch.mean(torch.tensor(self._rewards_plt[-self._f:], dtype=torch.float32)))
            self._mean_rates_plt.append(torch.mean(torch.tensor(self._rates_plt[-self._f:], dtype=torch.float32)))
            self._mean_td_errors_plt.append(
                torch.mean(torch.tensor(self._td_errors_plt[-self._f:], dtype=torch.float32)))
            self._mean_rhos_plt.append(torch.mean(torch.tensor(self._rhos_plt[-self._f:], dtype=torch.float32)))
            plt.clf()
        else:
            parser = Parser('dqn', 'res/log/dqn_2024_02_28_21_07_50.log')
            plt.clf()

        # Plot reward
        # ---------------------------------------------------------------------
        plt.subplot(3, 2, 1)

        plt.title('Rewards')
        plt.ylabel('Reward')

        if show_result == False:
            plt.plot(torch.tensor(self._rewards_plt, dtype=torch.float32).numpy())
        else:
            if parser != None:
                _dqn_mean_rewards_plt = []
                _dqn_rewards = parser.get_rewards()
                for i in range(len(_dqn_rewards)):
                    _dqn_mean_rewards_plt.append(torch.mean(torch.tensor(_dqn_rewards[:i][-self._f:], dtype=torch.float32)))
                plt.plot(torch.tensor(_dqn_mean_rewards_plt, dtype=torch.float32).numpy(), label='Proposed DQN')
        plt.plot(torch.tensor(self._mean_rewards_plt, dtype=torch.float32).numpy(), label='Risk Averse')
        if show_result == True:
            plt.legend(loc='best')
        # ---------------------------------------------------------------------

        # Plot rate
        # ---------------------------------------------------------------------
        plt.subplot(3, 2, 2)

        plt.title('Rates')
        plt.ylabel('Rate')

        if show_result == False:
            plt.plot(torch.tensor(self._rates_plt, dtype=torch.float32).numpy())
        else:
            if parser != None:
                _dqn_mean_rates_plt = []
                _dqn_rates = parser.get_rates()
                for i in range(len(_dqn_rates)):
                    _dqn_mean_rates_plt.append(torch.mean(torch.tensor(_dqn_rates[:i][-self._f:], dtype=torch.float32)))
                plt.plot(torch.tensor(_dqn_mean_rates_plt, dtype=torch.float32).numpy(), label='Proposed DQN')
        plt.plot(torch.tensor(self._mean_rates_plt, dtype=torch.float32).numpy(), label='Risk Averse')
        if show_result == True:
            plt.legend(loc='best')
        # ---------------------------------------------------------------------

        # Plot td error
        # ---------------------------------------------------------------------
        plt.subplot(3, 2, 3)

        plt.title('TD Errors')
        plt.ylabel('Value')

        plt.plot(torch.tensor(self._td_errors_plt, dtype=torch.float32).numpy())
        plt.plot(torch.tensor(self._mean_td_errors_plt, dtype=torch.float32).numpy())
        # ---------------------------------------------------------------------

        # Plot eps
        # ---------------------------------------------------------------------
        plt.subplot(3, 2, 4)

        plt.title('Epsilons')
        plt.ylabel('Value')

        plt.plot(torch.tensor(self._samples_plt, dtype=torch.float32).numpy())
        # plt.plot(torch.tensor(self._eps_plt, dtype=torch.float32).numpy())
        # ---------------------------------------------------------------------

        # Plot rho
        # ---------------------------------------------------------------------
        plt.subplot(3, 2, 5)

        plt.title('Rhos')
        plt.ylabel('Value')

        plt.plot(torch.tensor(self._rhos_plt, dtype=torch.float32).numpy())
        plt.plot(torch.tensor(self._mean_rhos_plt, dtype=torch.float32).numpy())
        # ---------------------------------------------------------------------

        plt.tight_layout()
        plt.pause(1 / 1024)

        if is_ipython:
            if not show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())

    def start_train(self):
        LogUtils.info('TRAIN_RISK_AVERSE', f'CUDA: {torch.cuda.is_available()}')
        for i_episode in range(self._episodes):
            state, _ = self._env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=self._device)

            sum_reward = 0
            sum_td_error = 0
            sum_rate = 0
            sum_rho = 0

            for t in count():
                # update Q Hat
                H = math.floor(RandomUtils.custom_random() * self._num_Q_Functions)
                self._update_Q_Hat(H, state)

                # select action according to Q Hat
                action = self._select_action(self._Q_hat(0), state, i_episode)
                action = action.item()

                # update Q Function
                observation, (k, P, Rho), (reward, reward_type), time_slot = self._env.step(action, i_episode)
                done = True if time_slot >= self._env.N else False
                next_state = None

                if done is not True:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=self._device)

                reward = torch.tensor(reward, dtype=torch.float32, device=self._device)
                td_error = self._optimize_model(state, action, next_state, reward)

                
                sum_td_error += td_error
                sum_reward += reward.item()
                sum_rate += 0 if reward <= 0 else reward
                sum_rho += Rho

                state = next_state
                # LogUtils.info(
                #     'TRAIN_EPISODE',
                #     f'({i_episode + 1}): '
                #     f'action: {action}, '
                #     f'observation: {observation}, '
                #     f'action_value: {k}, {P}, {Rho}, '
                #     f'reward: {reward.item()} - {reward_type}, '
                #     f'rho: {Rho}'
                # )

                if done:
                    break

            sum_td_error /= self._env.N
            sum_rho = round(sum_rho/self._env.N, 2)

            LogUtils.info(
                'TRAIN',
                f'({i_episode + 1}/{self._episodes}): '
                f'reward: {sum_reward}, '
                f'rate: {sum_rate}, '
                f'td_error: {sum_td_error}, '
                f'rho: {sum_rho}'
            )

            self._rewards_plt.append(sum_reward)
            self._rates_plt.append(sum_rate)
            self._td_errors_plt.append(sum_td_error)
            self._rhos_plt.append(sum_rho)

            self._plot(show_result=False)

        self._plot(show_result=True)
        plt.ioff()
        plt.show()
