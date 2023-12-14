import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
from itertools import count
from os.path import exists

from utils import LogUtils, RandomUtils

from modules.environment import Environment
from modules.model import DQNModel
from modules.relay import ReplayMemory, Transition


is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

class Train:
    def __init__(
            self,
            env = Environment(),
            batch_size = 256,
            gamma = 0.99,
            eps_max = 1,
            eps_min = 0.01,
            eps_decay = 0.001,
            tau = 0.001,
            learning_rate = 0.003, #alpha
    ):
        self.env = env

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
        state, info = self.env.reset()

        n_actions = self.env.get_num_actions()
        n_observations = self.env.get_num_states()
        LogUtils.info('TRAIN', f'actions: {n_actions}, observations: {n_observations}')

        self.policy_net = DQNModel(n_observations, n_actions).to(self.device)
        self.target_net = DQNModel(n_observations, n_actions).to(self.device)

        if exists('res/check_point/policy_model.pth'):
            self.policy_net.load_state_dict(torch.load('res/check_point/policy_model.pth'))

        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.SGD(self.policy_net.parameters(), lr = self.learning_rate)
        self.memory = ReplayMemory(50000)

        # self.steps_done = 0
        self.rewards = []
        self.SU_rewards = []

    def select_action(self, state, time_slot = 0):
        sample = RandomUtils.custom_random()
        eps_threshold = self.eps_min + (self.eps_max - self.eps_min) * math.exp(
            -1. * time_slot * self.eps_decay
            )
        # self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                action = self.policy_net(state).max(1)[1].view(1, 1)
                # noise = torch.tensor([[RandomUtils.normal(0, 0.1)]], device = self.device, dtype = torch.long)
                # action = action + noise
                # action = torch.clamp(action, 0, len(self.env.actions_space) - 1)
                return action
        else:
            return torch.tensor(
                [[RandomUtils.sample([s for s in range(len(self.env.actions_space))], 1)[0]]], device = self.device,
                dtype = torch.long
                )

    def plot_rewards(self, show_result = False):

        plt.figure(num =1, figsize = (10, 6), dpi = 100)
        num_to_get_mean = 100

        if show_result is False:
            plt.clf()

        # Plot reward
        # ---------------------------------------------------------------------
        plt.subplot(1, 2, 1)

        plt.title('Rewards')

        rewards_t = torch.tensor(self.rewards, dtype = torch.float)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.plot(rewards_t.numpy())

        # ---------------------------------------------------------------------

        # Plot average SU reward
        # ---------------------------------------------------------------------
        plt.subplot(1, 2, 2)

        plt.title('Average SU Rewards')

        SU_rewards_t = torch.tensor(self.SU_rewards, dtype = torch.float)
        plt.xlabel('Episode')
        plt.ylabel('Average SU Reward (bits/s/Hz)')
        plt.plot(SU_rewards_t.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated
        # ---------------------------------------------------------------------

        if is_ipython:
            if not show_result:
                display.display(plt.gcf())
                display.clear_output(wait = True)
            else:
                display.display(plt.gcf())

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return

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
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]

        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        criterion = nn.MSELoss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def start_train(self):
        LogUtils.info('TRAIN', f'CUDA: {torch.cuda.is_available()}')
        if torch.cuda.is_available():
            num_episodes = 1500
        else:
            num_episodes = 300
        best_reward = -1000
        sum_reward_episode = 0
        scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma = 0.99)

        for i_episode in range(num_episodes):
            state, info = self.env.reset()
            state = torch.tensor(state, dtype = torch.float32, device = self.device).unsqueeze(0)
            # LogUtils.info('TRAIN', f'episode: {i_episode}, state: {state}')
            sum_reward = 0
            num_step = 0
            time_slot = 0
            for t in count():
                num_step += 1
                action = self.select_action(state, time_slot)
                observation, _action, reward, time_slot = self.env.step(action.item())
                # LogUtils.info('TRAIN', f'observation: {observation}\naction: {action}\nreward: {reward}\ntime_slot: {time_slot}')
                reward = torch.tensor([reward], device = self.device)

                reward_item = reward.squeeze(0).item()
                sum_reward += reward_item

                done = True if time_slot >= self.env.N else False

                if done == True:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype = torch.float32, device = self.device).unsqueeze(0)

                self.memory.push(state, action, next_state, reward)

                state = next_state

                self.optimize_model()

                if t % 5 == 4:
                    target_net_state_dict = self.target_net.state_dict()
                    policy_net_state_dict = self.policy_net.state_dict()

                    self.target_net.load_state_dict(target_net_state_dict)

                if done:
                    LogUtils.info('TRAIN', f'({i_episode + 1}/{num_episodes}) reward: {sum_reward}')
                    break

            sum_reward_episode += sum_reward
            self.rewards.append(sum_reward)

            avg_reward = sum_reward_episode / (i_episode + 1)
            if best_reward < avg_reward:
                best_reward = avg_reward
                torch.save(self.policy_net.state_dict(), 'res/check_point/policy_model.pth')
                LogUtils.info('TRAIN', f'SAVE MODEL: best_reward: {best_reward}')

            self.SU_rewards.append(avg_reward)
            self.plot_rewards()

            scheduler.step()

        self.plot_rewards(show_result = True)
        plt.ioff()
        plt.show()
