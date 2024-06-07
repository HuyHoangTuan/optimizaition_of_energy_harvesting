import math
import torch
import matplotlib
import matplotlib.pyplot as plt
from itertools import count
from src.utils import LogUtils
from src.utils import RandomUtils
from src.utils import Parser
from src.modules.environment import Environment
from src.modules.model import DQNs
from torch import nn
from src.modules.replay import ReplayMemory, Transition
import random

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

POLICY = 'policy'
TARGET = 'target'

class RA_DQNTrain:
    def __init__(
            self,
            episodes=1600,
            num_su=1,
            eps_max=1,
            eps_min=0.01,
            eps_decay=0.001,
            alpha=0.003,
            gamma=0.99,
            beta = -0.99,
            batch_size=128,
            risk_control_parameter=0.01,
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
            NumSU=num_su,
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

        self._transmit_actions_plt = []
        self._mean_transmit_actions_plt = []

        self._P_t_plt = []
        self._mean_P_t_plt = []

        self._num_DQN = 3
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

    def _update_Q_Hat(self, H, state):
        Q = self._Q(1, H)
        updated_params = {}  # Initialize a dictionary to hold updated parameters

        for key, _ in Q.named_parameters():
            params_stacked = torch.stack([self._Q(1, idx).state_dict()[key] for idx in range(self._num_DQN)], dim=0)
            params_mean = torch.mean(params_stacked, dim=0)

            std_deviation_squared = torch.sqrt(
                    torch.div(
                        torch.sum(torch.pow(params_stacked - params_mean.unsqueeze(0), 2), dim=0),
                        self._num_DQN - 1
                )
            )
            # Apply the update rule to compute the new parameter value
            updated_params[key] = Q.state_dict()[key] - self._lambdaP * std_deviation_squared

        # Use load_state_dict to apply the updated parameters to Q_hat
        self._Q_hat(0, 0).load_state_dict(updated_params)


    
    def _plot(self, **kwargs):
        
        title = kwargs['title']
        ylabel = kwargs['ylabel']
        is_need_mean = kwargs['is_need_mean']
        ylim = kwargs['ylim']
        is_need_extra_data = kwargs['is_need_extra_data']

        old_label = 'Baseline'
        old_line_style = '--'
        new_label = 'Proposed'
        new_line_style = '-'

        row = 3
        column = 2

        idx = kwargs['idx']
        data = kwargs['data']
        parser_data = kwargs['parser_data']
        extra_data = kwargs['extra_data']

        ax = plt.subplot(row, column, idx)
        if ylim != None:
            ax.set_ylim(0, 10)
        
        plt.title(title)
        plt.ylabel(ylabel)
        

        if parser_data == None:
            data_t = torch.tensor(data, dtype=torch.float)
            plt.plot(data_t.numpy())
        else:
            mean_parser_t = []
            if type(parser_data) == list:
                for i in range(len(parser_data)):
                    mean_parser_t.append(torch.mean(torch.tensor(parser_data[:i][-self._f:], dtype=torch.float32)))
                plt.plot(torch.tensor(mean_parser_t, dtype=torch.float32).numpy(), label=old_label, ls=old_line_style, linewidth=3)


        if is_need_mean is True:
            mean_t = []
            for i in range(len(data)):
                mean_t.append(torch.mean(torch.tensor(data[:i][-self._f:], dtype=torch.float32)))
            plt.plot(torch.tensor(mean_t, dtype=torch.float32).numpy(), label=new_label, ls=new_line_style, linewidth=3)

        if extra_data != None and is_need_mean is True and is_need_extra_data is True:
            if is_need_mean is True:
                mean_extra_t = []
                for i in range(len(extra_data)):
                    mean_extra_t.append(torch.mean(torch.tensor(extra_data[:i][-self._f:], dtype=torch.float32)))
                plt.plot(torch.tensor(mean_extra_t, dtype=torch.float32).numpy(), label='Bound', ls=':', linewidth=3)

        if parser_data != None:
            plt.legend(loc='best')
            ax.grid()

    def plot_rewards(self, show_result=False):
        plt.figure(num=1, figsize=(16, 9), dpi=120)
        parser = None

        base_rewards = None
        base_rates = None
        base_loss = None
        base_transmit_actions = None
        base_P = None
        base_rho = None

        if show_result is False:
            # self.SU_rewards_t.append(torch.mean(torch.tensor(self.SU_rewards, dtype=torch.float)))
            # self.mean_rhos_t.append(torch.mean(torch.tensor(self.mean_rhos, dtype=torch.float32)))
            # self.mean_sum_rates_t.append(torch.mean(torch.tensor(self.mean_sum_rates, dtype=torch.float)))
            # self.mean_gs_t.append(torch.mean(torch.tensor(self.mean_gs, dtype = torch.float)))
            # self.mean_transmit_action_t.append(torch.mean(torch.tensor(self.mean_transmit_action, dtype=torch.float)))
            plt.clf()
        else:
            # parser = Parser('dqn', f'res/result/base_result_{self.env.NumSU}.log')
            parser = Parser('dqn', f'res/result/base_result_dynamic_rho_{self._env.NumSU}.log')
            base_rewards, base_rates, base_loss, base_rho, base_transmit_actions, base_P = parser.get_data()
            plt.clf()
            
        self._plot(
            idx=1,
            is_need_mean=True,
            title='reward',
            ylabel='value',
            ylim=None,
            data=self._rewards_plt,
            parser_data=base_rewards,
            extra_data=None,
            is_need_extra_data=False
        )

        self._plot(
            idx=2,
            is_need_mean=True,
            title='rate',
            ylabel='value',
            ylim=None,
            data=self._rates_plt,
            parser_data=base_rates,
            extra_data=None,
            is_need_extra_data=(self._env.NumSU == 1)
        )
            
        self._plot(
            idx=3,
            is_need_mean=True,
            title='transmission action',
            ylabel='value',
            ylim=None,
            data=self._transmit_actions_plt,
            parser_data=base_transmit_actions,
            extra_data=None,
            is_need_extra_data=False
        )

        if self._is_dynamic_rho is True:
            print(base_rho)
            self._plot(
                idx=4,
                is_need_mean=True,
                title='rho',
                ylabel='value',
                ylim=None,
                data=self._rhos_plt,
                parser_data=base_rho,
                extra_data=None,
                is_need_extra_data=False
                )
        else:
            self._plot(
                idx=4,
                is_need_mean=False,
                title='epsilon',
                ylabel='value',
                ylim=None,
                data=self._eps_plt,
                parser_data=None,
                extra_data=None,
                is_need_extra_data=False
            )

        self._plot(
            idx=5,
            is_need_mean=True,
            title='loss',
            ylabel='value',
            ylim=10,
            data=self._loss_plt,
            parser_data=base_loss,
            extra_data=None,
            is_need_extra_data=False
        )

        self._plot(
            idx=6,
            is_need_mean=True,
            title='transmission power',
            ylabel='value',
            ylim=None,
            data=self._P_t_plt,
            parser_data=base_P,
            extra_data=None,
            is_need_extra_data=False
        )
        plt.tight_layout()
        plt.pause(1 / 1024)  # pause a bit so that plots are updated
        # ---------------------------------------------------------------------

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
                state_action_values = self._Q(0, idx)(state_batch).gather(1, action_batch)
                next_state_values = torch.zeros(self._batch_size, device=self._device)
                with torch.no_grad():
                    next_state_values[non_final_mask] = self._Q(1, idx)(non_final_next_states).max(1).values

                # _learning_rate = self._Q.get_learning_rate(idx, state_batch, action_batch)
                expected_state_action_values = (
                        state_action_values + (
                            self._utility_function(
                                (
                                    reward_batch.unsqueeze(1)
                                    + next_state_values.unsqueeze(1) * self._gamma
                                    - state_action_values
                                )
                            )
                            + 1.0
                        )
                )

                # print(self._utility_function(
                #                 (
                #                     reward_batch.unsqueeze(1)
                #                     + next_state_values.unsqueeze(1) * self._gamma
                #                     - state_action_values
                #                 )
                #             ))
                _loss = self._Q.loss(idx, state_action_values, expected_state_action_values)
                loss = loss + _loss
                cnt += 1
                self._Q.soft_update(idx)
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

            sum_transmit_actions_episode = 0

            _P_t = []

            # if i_episode % 5 == 0:
            #     # update Q Hat
            #     H = random.randint(0, self._num_DQN - 1)
            #     self._update_Q_Hat(H, state)

            for t in count():
                # update Q Hat
                H = random.randint(0, self._num_DQN - 1)
                self._update_Q_Hat(H, state)
                # select action according to Q Hat
                action = self._select_action(self._Q_hat(0, 0), state, i_episode)
                # print(action)
                # action = action.item()

                # update Q Function
                observation, (k, P, Rho), (reward, rate, _), time_slot = self._env.step(action, i_episode)
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
                sum_rate += rate
                sum_rho += Rho
                sum_transmit_actions_episode += 1 if k == 0 else 0
                _P_t.append(P)

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
                    # if i_episode % 5 == 0:
                    #     for i in range(self._num_DQN):
                    #         self._Q.soft_update(i)
                    break

            sum_loss /= self._env.N
            sum_rho = sum_rho/self._env.N

            LogUtils.info(
                'TRAIN',
                f'({i_episode + 1}/{self._episodes}): '
                f'reward: {sum_reward}, '
                f'rates: {sum_rate}, '
                f'loss: {sum_loss}, '
                f'rho: {sum_rho}, '
                f'transmit_actions: {sum_transmit_actions_episode}, '
                f'P: {_P_t}'
            )

            self._rewards_plt.append(sum_reward)
            self._rates_plt.append(sum_rate)
            self._loss_plt.append(sum_loss)
            self._rhos_plt.append(sum_rho)
            self._transmit_actions_plt.append(sum_transmit_actions_episode)
            
            self._P_t_plt.extend(_P_t)
            # for _P in _P_t:
            #     self._P_t_plt.append(_P)
            #     self._mean_P_t_plt.append(torch.mean(torch.tensor(self._P_t_plt[-self._f:], dtype=torch.float)))

            self.plot_rewards(show_result=False)

        self.plot_rewards(show_result=True)
        plt.ioff()
        plt.show()
