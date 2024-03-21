import math
import torch
import matplotlib
import matplotlib.pyplot as plt
from itertools import count
from utils import LogUtils
from utils import RandomUtils
from utils import Parser
from modules.environment import Environment
from modules.model import DQNs

from src.modules.replay import ReplayMemory, Transition

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
            beta = -0.5,
            batch_size=64,
            risk_control_parameter=0.001,
            is_dynamic_rho=False,
    ):
        self._episodes = episodes
        self._eps_max = eps_max
        self._eps_min = eps_min
        self._eps_decay = eps_decay
        self._eps_drop_rate = 0.0
        self._beta = beta
        self._eps_threshold = 500
        self._lambdaP = risk_control_parameter
        self._alpha = alpha
        self._gamma = gamma
        self._is_dynamic_rho = is_dynamic_rho
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

        self._loss_plt = []
        self._mean_loss_plt = []

        self._rhos_plt = []
        self._mean_rhos_plt = []

        self._num_DQN = 5
        self._Q = DQNs(
            self._device,
            self._env.get_num_states(),
            self._env.get_num_actions(),
            self._alpha,
            self._num_DQN
        )

        self._Q_hat = DQNs(
            self._device,
            self._env.get_num_states(),
            self._env.get_num_actions(),
            self._alpha,
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
                return Q(state).max(1).indices.view(1, 1)
                # return torch.argmax(
                #     Q(state)
                # )
        else:
            _ = [s for s in range(self._env.get_num_actions())]
            return torch.tensor([[RandomUtils.sample(_, 1)[0]]], dtype=torch.long, device=self._device)

    with torch.no_grad():
        def _update_Q_Hat(self, H, state):
            Q = self._Q(H)
            for key, _ in Q.named_parameters():
                params_stacked = torch.stack([self._Q(idx).state_dict()[key] for idx in range(self._num_DQN)], dim=0)
                params_minus = torch.mean(params_stacked, dim=0)

                std_deviation_squared = torch.div(
                    torch.sum(torch.pow(params_stacked - params_minus, 2), dim=0),
                    self._num_DQN - 1
                )
                self._Q_hat(0).state_dict()[key] = Q.state_dict()[key] - self._lambdaP * std_deviation_squared

    def _plot(self, show_result=False):
        plt.figure(num=1, figsize=(16, 9), dpi=120)
        parser = None
        if not show_result:
            self._mean_rewards_plt.append(torch.mean(torch.tensor(self._rewards_plt[-self._f:], dtype=torch.float32)))
            self._mean_rates_plt.append(torch.mean(torch.tensor(self._rates_plt[-self._f:], dtype=torch.float32)))
            self._mean_loss_plt.append(
                torch.mean(torch.tensor(self._loss_plt[-self._f:], dtype=torch.float32)))
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
        plt.plot(torch.tensor(self._mean_rewards_plt, dtype=torch.float32).numpy(), label='Risk Averse DQN')
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
        plt.plot(torch.tensor(self._mean_rates_plt, dtype=torch.float32).numpy(), label='Risk Averse DQN')
        if show_result == True:
            plt.legend(loc='best')
        # ---------------------------------------------------------------------

        # Plot td error
        # ---------------------------------------------------------------------
        plt.subplot(3, 2, 3)

        plt.title('Loss')
        plt.ylabel('Value')

        plt.plot(torch.tensor(self._loss_plt, dtype=torch.float32).numpy())
        plt.plot(torch.tensor(self._mean_loss_plt, dtype=torch.float32).numpy())
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
        if self._is_dynamic_rho is True:
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

    def _utility_function(self, value):
        return -1.0 * torch.exp(self._beta * value)

    def _optimize_model(self, state, action, next_state, reward):
        self._memory.push(state, action, next_state, reward)

        if len(self._memory) < self._batch_size:
            return 0
        transitions = self._memory.sample(self._batch_size)
        batch = Transition(*zip(*transitions))
        non_final_mask = torch.tensor(
            tuple(
                map(
                    lambda s: s is not None,
                    batch.next_state
                )
            ),
            device = self._device,
            dtype = torch.bool
        )
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        M = RandomUtils.poisson(Lambda=1.0, size=self._num_DQN)
        loss = 0
        cnt = 0
        for idx in range(self._num_DQN):
            if M[idx] == 1:
                state_action_values = self._Q(idx)(state_batch).gather(1, action_batch)
                next_state_values = torch.zeros(self._batch_size, device=self._device)
                with torch.no_grad():
                    next_state_values[non_final_mask] = self._Q_hat(0)(non_final_next_states).max(1).values

                expected_state_action_values = state_action_values + (self._utility_function(
                        reward_batch + next_state_values.unsqueeze(1) * self._gamma - state_action_values
                ) + 1.0)
                loss = loss + self._Q.loss(idx, state_action_values, expected_state_action_values)
                cnt += 1
        return 0 if cnt == 0 else loss / cnt

    def start_train(self):
        LogUtils.info('TRAIN_RISK_AVERSE_DQN', f'CUDA: {torch.cuda.is_available()}')
        for i_episode in range(self._episodes):
            state, _ = self._env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=self._device).unsqueeze(0)
        
            sum_reward = 0
            sum_loss = 0
            sum_rate = 0
            sum_rho = 0
        
            for t in count():
                # update Q Hat
                H = math.floor(RandomUtils.custom_random() * self._num_DQN)
                self._update_Q_Hat(H, state)
        
                # select action according to Q Hat
                action = self._select_action(self._Q_hat(0), state, i_episode)
                # print(action)
                # action = action.item()
        
                # update Q Function
                observation, (k, P, Rho), (reward, reward_type), time_slot = self._env.step(action, i_episode)
                done = True if time_slot >= self._env.N else False
                next_state = None
        
                if done is not True:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=self._device)
                    next_state = next_state.unsqueeze(0)
                reward = torch.tensor([reward], dtype=torch.float32, device=self._device)
                loss = self._optimize_model(state, action, next_state, reward)
                state = next_state
                sum_loss += loss
                sum_reward += reward.item()
                sum_rate += 0 if reward.item() <= 0 else reward.item()
                sum_rho += Rho
        
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
        
            sum_loss /= self._env.N
            sum_rho = round(sum_rho/self._env.N, 2)
        
            LogUtils.info(
                'TRAIN',
                f'({i_episode + 1}/{self._episodes}): '
                f'reward: {sum_reward}, '
                f'rate: {sum_rate}, '
                f'td_error: {sum_loss}, '
                f'rho: {sum_rho}'
            )
        
            self._rewards_plt.append(sum_reward)
            self._rates_plt.append(sum_rate)
            self._loss_plt.append(sum_loss)
            self._rhos_plt.append(sum_rho)
        
            self._plot(show_result=False)

        self._plot(show_result=True)
        plt.ioff()
        plt.show()
