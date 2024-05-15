import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
from itertools import count
from os.path import exists

from src.utils import LogUtils, RandomUtils

from src.modules.environment import Environment
from src.modules.model import DQNModel
from src.modules.replay import ReplayMemory, Transition

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

class Train:
    def __init__(
            self,
            num_episode = 1600,
            is_dynamic_rho = False,
            reward_function_id = 0,
            batch_size = 64,
            gamma = 0.99,
            eps_max = 1,
            eps_min = 0.01,
            eps_decay = 0.001,
            tau = 0.001,
            learning_rate = 0.003,  # alpha
    ):
        self.is_dynamic_rho = is_dynamic_rho
        self.reward_function_id = reward_function_id
        self.num_episode = num_episode

        self.env = Environment(
            Episode = self.num_episode,
            Dynamic_Rho = self.is_dynamic_rho,
            reward_function_id = self.reward_function_id
        )

        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_max = eps_max
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.tau = tau
        self.learning_rate = learning_rate


        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._init_evn()

    def _init_evn(self):

        n_actions = self.env.get_num_actions()
        n_observations = self.env.get_num_states()
        LogUtils.info('TRAIN', f'actions: {n_actions}, observations: {n_observations}')

        self.policy_net = DQNModel(n_observations, n_actions).to(self.device)
        self.target_net = DQNModel(n_observations, n_actions).to(self.device)

        # if exists('res/check_point/target_model.pth'):
            # self.policy_net.load_state_dict(torch.load('res/check_point/dqn/target_model.pth'))

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.SGD(self.policy_net.parameters(), lr = self.learning_rate)
        self.memory = ReplayMemory(100000)

        self.eps_threshold = 500 if self.is_dynamic_rho is False else 400 # Episode
        self.steps_done = 0
        self.eps_drop_rate = 0
        # visualization
        self.num_to_get_mean = 100
        self.rewards = []
        self.SU_rewards = []
        self.R_0_types = []
        self.R_1_types = []
        self.R_2_types = []
        self.eps = []
        self.eps_e = []
        self.rhos = []
        self.mean_rhos = []
        self.sample = []
        self.sum_rates = []
        self.mean_sum_rates = []
        self.actions = []
        self.losses = []

        # plt
        self.SU_rewards_t = [0]
        self.mean_rhos_t = []
        self.mean_sum_rates_t = [0]
        # self.gs_t = [0]
        # self.mean_gs = []
        # self.mean_gs_t = [0]
        self.P_t = [0]
        self.mean_P_t = [0]
        self.transmit_action = [0]
        self.mean_transmit_action = []
        self.mean_transmit_action_t = [0]

    def select_action(self, state, episode = 0):
        sample = RandomUtils.custom_random()
        eps_threshold = self.eps_min + (self.eps_max - self.eps_min) * math.exp(
            -1. * self.eps_drop_rate * self.eps_decay
        )

        if episode < self.eps_threshold:
            sample = -1
            self.eps_drop_rate = 0
        else:
            self.eps_drop_rate += 1.5

        self.eps.append(eps_threshold)
        self.sample.append(sample)

        # self.steps_done += 1
        if sample > eps_threshold:
            # self.actions.append(1)
            with torch.no_grad():
                # action = torch.argmax(self.policy_net(state), dim = 1).item()
                return self.policy_net(state).max(1).indices.view(1, 1)
        else:
            # self.actions.append(0)
            return torch.tensor(
                [[RandomUtils.sample([s for s in range(len(self.env.actions_space))], 1)[0]]], device = self.device,
                dtype = torch.long
            )

    def plot_rewards(self, show_result = False):

        plt.figure(num = 1, figsize = (16, 9), dpi = 120)

        if show_result is False:
            self.SU_rewards_t.append(torch.mean(torch.tensor(self.SU_rewards, dtype = torch.float)))
            self.mean_rhos_t.append(torch.mean(torch.tensor(self.mean_rhos, dtype = torch.float32)))
            self.mean_sum_rates_t.append(torch.mean(torch.tensor(self.mean_sum_rates, dtype = torch.float)))
            # self.mean_gs_t.append(torch.mean(torch.tensor(self.mean_gs, dtype = torch.float)))
            self.mean_transmit_action_t.append(torch.mean(torch.tensor(self.mean_transmit_action, dtype = torch.float)))
            plt.clf()

        # Plot reward
        # ---------------------------------------------------------------------
        plt.subplot(3, 2, 1)

        plt.title('Rewards')
        plt.ylabel('Reward')

        rewards_t = torch.tensor(self.rewards, dtype = torch.float)
        plt.plot(rewards_t.numpy())

        SU_rewards_t = torch.tensor(self.SU_rewards_t, dtype = torch.float)
        plt.plot(SU_rewards_t.numpy())

        # ---------------------------------------------------------------------

        # Plot Sum rate
        # ---------------------------------------------------------------------
        plt.subplot(3, 2, 2)
        #
        plt.title('Rates')
        plt.ylabel('Rate value')

        rates_t = torch.tensor(self.sum_rates, dtype=torch.float)
        plt.plot(rates_t.numpy())

        mean_sum_rates_t = torch.tensor(self.mean_sum_rates_t, dtype=torch.float)
        plt.plot(mean_sum_rates_t.numpy())

        # ---------------------------------------------------------------------

        # Plot EPS
        # ---------------------------------------------------------------------
        plt.subplot(3, 2, 4)
        #
        plt.title('Epsilon')
        plt.ylabel('Epsilon value')

        eps_t = torch.tensor(self.eps_e, dtype = torch.float)
        plt.plot(eps_t.numpy())

        # ---------------------------------------------------------------------

        # Plot rho
        # ---------------------------------------------------------------------
        plt.subplot(3, 2, 3)

        plt.title('Transmit Action')
        plt.ylabel('Number')

        transmit_action_t = torch.tensor(self.transmit_action, dtype = torch.float)
        plt.plot(transmit_action_t.numpy())

        mean_transmit_action_t = torch.tensor(self.mean_transmit_action_t, dtype = torch.float)
        plt.plot(mean_transmit_action_t.numpy())

        # mean_rhos_t = torch.tensor(self.mean_rhos_t, dtype = torch.float32)
        # print(mean_rhos_t.numpy().astype(np.float32))
        # if self.is_dynamic_rho is True:
        #     plt.plot(mean_rhos_t.numpy())

        # ---------------------------------------------------------------------

        # Plot loss
        # ---------------------------------------------------------------------
        ax = plt.subplot(3, 2, 5)
        plt.title('Loss')
        plt.ylabel('Loss value')
        ax.set_ylim(0, 10)

        losses_t = torch.tensor(self.losses, dtype = torch.float)
        plt.plot(losses_t.numpy())
        # ---------------------------------------------------------------------

        # Plot Transmission Power between SU-Rx and SU-Tx
        # ---------------------------------------------------------------------
        plt.subplot(3, 2, 6)

        plt.title('P: SU-Rx <-> SU-Tx')
        plt.ylabel('Values')

        P_t = torch.tensor(self.P_t, dtype=torch.float)
        plt.plot(P_t.numpy())

        mean_P_t = torch.tensor(self.mean_P_t, dtype=torch.float)
        plt.plot(mean_P_t.numpy())

        # ---------------------------------------------------------------------

        # Plot Reward Type PU2
        # ---------------------------------------------------------------------
        # plt.subplot(3, 2, 6)
        #
        # if show_result is False:
        #     plt.title('PU2 Reward Type')
        #     # plt.ylim(0, self.env.N)
        # else:
        #     plt.title('PU2 Reward Type (Sum)')
        # plt.ylabel('Number')
        #
        # categories = ['0', '1', '2']
        #
        # PU2_R_O_types = np.array(self.R_0_types)[:, 0]
        # PU2_R_1_types = np.array(self.R_1_types)[:, 0]
        # PU2_R_2_types = np.array(self.R_2_types)[:, 0]
        # if show_result is False:
        #     PU2_R_types = np.array([PU2_R_O_types[-1], PU2_R_1_types[-1], PU2_R_2_types[-1]])
        # else:
        #     PU2_R_types = np.array([PU2_R_O_types.sum(), PU2_R_1_types.sum(), PU2_R_2_types.sum()])
        #
        # plt.bar(categories, PU2_R_types)
        # ---------------------------------------------------------------------

        plt.tight_layout()
        plt.pause(1/1024)  # pause a bit so that plots are updated
        # ---------------------------------------------------------------------

        if is_ipython:
            if not show_result:
                display.display(plt.gcf())
                display.clear_output(wait = True)
            else:
                display.display(plt.gcf())

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return 0

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        non_final_mask = torch.tensor(
            tuple(
                map(
                    lambda s: s is not None,
                    batch.next_state
                )
            ),
            device = self.device,
            dtype = torch.bool
        )
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(self.batch_size, device = self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values

        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        self.optimizer.zero_grad()
        criterion = nn.MSELoss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        loss.backward()

        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        self.steps_done += 1
        return loss

    def start_train(self):
        LogUtils.info('TRAIN', f'CUDA: {torch.cuda.is_available()}')
        best_reward = -1000
        sum_reward_episode = 0
        # scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma = 0.99)

        sum_actions = 0
        sum_transmit_actions = 0

        for i_episode in range(self.num_episode):
            state, _ = self.env.reset()
            state = torch.tensor(state, dtype = torch.float32, device = self.device).unsqueeze(0)

            sum_reward = 0
            sum_Rho = 0
            sum_rate = 0
            sum_loss = 0
            # sum_gs = 0
            sum_transmit_actions_episode = 0
            count_loss = 0
            _P_t = []

            for t in count():
                action = self.select_action(state, i_episode)
                observation, (k, P, Rho), (reward, rate, _), time_slot = self.env.step(action.item(), i_episode)
                reward = torch.tensor([reward], dtype = torch.float32, device = self.device)

                sum_rate += rate
                sum_reward += reward.squeeze(0).item()
                # sum_gs += gs
                sum_Rho += Rho
                # sum_actions += 1
                sum_transmit_actions += 1 if k == 0 else 0
                sum_transmit_actions_episode += 1 if k == 0 else 0
                _P_t.append(P)

                done = True if time_slot >= self.env.N else False

                if done == True:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype = torch.float32, device = self.device)
                    next_state = next_state.unsqueeze(0)

                self.memory.push(state, action, next_state, reward)

                state = next_state

                loss = self.optimize_model()
                if len(self.memory) >= self.batch_size:
                    sum_loss += loss.item()
                    count_loss += 1

                # if self.steps_done % 12000 == 0:
                #     self.target_net.load_state_dict(self.policy_net.state_dict())



                # LogUtils.info(
                #     'TRAIN_EPISODE',
                #     f'({i_episode + 1}): '
                #     f'action: {action.squeeze(0).item()}, '
                #     f'observation: {observation}, '
                #     f'action_value: {k}, {P}, {Rho}, '
                #     f'reward: {reward.squeeze(0).item()} - {reward_type}, '
                #     f'eps: {self.eps[-1]}, '
                #     f'sample: {self.sample[-1]}, '
                # )
                if done:
                    if i_episode % 5 == 0:
                        _dict = {}
                        for key in self.policy_net.state_dict():
                            _dict[key] = self.policy_net.state_dict()[key] * self.tau + self.target_net.state_dict()[
                                key] * (1 - self.tau)
                        self.target_net.load_state_dict(_dict)
                    break

            LogUtils.info(
                'TRAIN',
                f'({i_episode + 1}/{self.num_episode}): '
                f'reward: {sum_reward}, '
                f'rates: {sum_rate}, '
                f'loss: {0 if count_loss <=0 else sum_loss / count_loss}, '
                f'rho: {sum_Rho / self.env.N}'
            )

            self.losses.append(0 if count_loss <=0 else sum_loss / count_loss)

            self.sum_rates.append(sum_rate)

            self.rhos.append(sum_Rho / self.env.N)

            self.rewards.append(sum_reward)

            self.eps_e.append(torch.mean(torch.tensor(self.eps, dtype = torch.float)))
            self.eps = []

            self.mean_sum_rates.append(sum_rate)
            if len(self.mean_sum_rates) > self.num_to_get_mean:
                self.mean_sum_rates = self.mean_sum_rates[1:]

            self.SU_rewards.append(sum_reward)
            if len(self.SU_rewards) > self.num_to_get_mean:
                self.SU_rewards = self.SU_rewards[1:]

            self.mean_rhos.append(sum_Rho / self.env.N)
            if len(self.mean_rhos) > self.num_to_get_mean:
                self.mean_rhos = self.mean_rhos[1:]

            # self.gs_t.append(sum_gs / self.env.N)
            # self.mean_gs.append(sum_gs / self.env.N)
            # if len(self.mean_gs) > self.num_to_get_mean:
            #     self.mean_gs = self.mean_gs[1:]

            for _P in _P_t:
                self.P_t.append(_P)
                self.mean_P_t.append(torch.mean(torch.tensor(self.P_t[-self.num_to_get_mean:], dtype=torch.float)))

            self.transmit_action.append(sum_transmit_actions_episode)
            self.mean_transmit_action.append(sum_transmit_actions_episode)
            if len(self.mean_transmit_action) > self.num_to_get_mean:
                self.mean_transmit_action = self.mean_transmit_action[1:]

            self.plot_rewards()

            # scheduler.step()

        LogUtils.info('TRAIN_FINISHED', f'Transmit Action: {sum_transmit_actions}/{sum_actions}')
        self.plot_rewards(show_result = True)
        plt.ioff()
        plt.show()
        torch.save(self.target_net.state_dict(), 'res/check_point/dqn/target_model.pth')
