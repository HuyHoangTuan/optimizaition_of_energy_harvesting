import math
import torch
import matplotlib
import matplotlib.pyplot as plt
from itertools import count
from utils import LogUtils
from utils import RandomUtils
from modules.environment import Environment
from modules.model import DQNs

from src.modules.replay import ReplayMemory

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

POLICY = 'policy'
TARGET = 'target'

class RA_DQNTrain:
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

        self._memory = ReplayMemory(100000)
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

        self._num_DQN = 2
        self._Q = DQNs(
            self._device,
            self._env.get_num_states(),
            self._env.get_num_actions(),
            self._num_DQN
        )

        self._Q_hat = DQNs(
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

    def _update_Q_Hat(self, H, state):
        # Q_stacked = torch.stack([[] for key in self._Q(idx, POLICY).state_dict() for idx in range(self._num_DQN)])
        # Q_Minus = torch.mean(Q_stacked, dim=0)
        # print(Q_Minus)
        # Calculate the standard deviation for each action
        # Note: std deviation formula is sqrt(sum((x - mean)^2) / (n-1)), but here we need the squared term
        # std_deviation_squared = torch.div(
        #     torch.sum(torch.pow(Q_stacked - Q_Minus, 2), dim=0),
        #     self._num_DQN - 1
        # )
        #
        # self._Q_hat(0, POLICY)(state)[:] = Q - self._lambdaP * std_deviation_squared
        pass

    def start_train(self):
        LogUtils.info('TRAIN_RISK_AVERSE_DQN', f'CUDA: {torch.cuda.is_available()}')
        self._update_Q_Hat(0,0)
        # for i_episode in range(self._episodes):
        #     state, _ = self._env.reset()
        #     state = torch.tensor(state, dtype=torch.float32, device=self._device)
        #
        #     sum_reward = 0
        #     sum_td_error = 0
        #     sum_rate = 0
        #     sum_rho = 0
        #
        #     for t in count():
        #         # update Q Hat
        #         H = math.floor(RandomUtils.custom_random() * self._num_Q_Functions)
        #         self._update_Q_Hat(H, state)
        #
        #         # select action according to Q Hat
        #         action = self._select_action(self._Q_hat(0), state, i_episode)
        #         action = action.item()
        #
        #         # update Q Function
        #         observation, (k, P, Rho), (reward, reward_type), time_slot = self._env.step(action, i_episode)
        #         done = True if time_slot >= self._env.N else False
        #         next_state = None
        #
        #         if done is not True:
        #             next_state = torch.tensor(observation, dtype=torch.float32, device=self._device)
        #
        #         reward = torch.tensor(reward, dtype=torch.float32, device=self._device)
        #         td_error = self._optimize_model(state, action, next_state, reward)
        #
        #         sum_td_error += td_error
        #         sum_reward += reward.item()
        #         sum_rate += 0 if reward <= 0 else reward
        #         sum_rho += Rho
        #
        #         # LogUtils.info(
        #         #     'TRAIN_EPISODE',
        #         #     f'({i_episode + 1}): '
        #         #     f'action: {action}, '
        #         #     f'observation: {observation}, '
        #         #     f'action_value: {k}, {P}, {Rho}, '
        #         #     f'reward: {reward.item()} - {reward_type}, '
        #         #     f'rho: {Rho}'
        #         # )
        #
        #         if done:
        #             break
        #
        #     sum_td_error /= self._env.N
        #     sum_rho = round(sum_rho/self._env.N, 2)
        #
        #     LogUtils.info(
        #         'TRAIN',
        #         f'({i_episode + 1}/{self._episodes}): '
        #         f'reward: {sum_reward}, '
        #         f'rate: {sum_rate}, '
        #         f'td_error: {sum_td_error}, '
        #         f'rho: {sum_rho}'
        #     )
        #
        #     self._rewards_plt.append(sum_reward)
        #     self._rates_plt.append(sum_rate)
        #     self._td_errors_plt.append(sum_td_error)
        #     self._rhos_plt.append(sum_rho)
        #
        #     self._plot(show_result=False)