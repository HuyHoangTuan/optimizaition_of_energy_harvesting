import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
from itertools import count

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
            batch_size = 128,
            gamma = 0.99,
            eps_start = 0.9,
            eps_end = 0.05,
            eps_decay = 1000,
            tau = 0.005,
            learning_rate = 1e-4,
    ):
        self.env = env

        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
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
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr = self.learning_rate, amsgrad = True)
        self.memory = ReplayMemory(10000)

        self.steps_done = 0
        self.rewards = []
        self.SU_rewards = []

    def select_action(self, state):
        sample = RandomUtils.custom_random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(
            -1. * self.steps_done * self.eps_decay
            )
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor(
                [[RandomUtils.sample([s for s in range(len(self.env.actions_space))], 1)[0]]], device = self.device,
                dtype = torch.long
                )

    def plot_rewards(self, show_result = False):
        plt.figure(num = 1)
        rewards_t = torch.tensor(self.rewards, dtype = torch.float)

        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.plot(rewards_t.numpy())

        num_to_get_mean = 100

        # Take 100 episode averages and plot them too
        if len(rewards_t) >= num_to_get_mean:
            means = rewards_t.unfold(0, num_to_get_mean, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(num_to_get_mean), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated
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

        criterion = nn.SmoothL1Loss()
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

        for i_episode in range(num_episodes):
            state, info = self.env.reset()
            state = torch.tensor(state, dtype = torch.float32, device = self.device).unsqueeze(0)
            # LogUtils.info('TRAIN', f'episode: {i_episode}, state: {state}')
            sum_reward = 0
            num_step = 0

            for t in count():
                num_step += 1
                action = self.select_action(state)
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

                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()

                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key] * self.tau + target_net_state_dict[key] * (
                                1 - self.tau)

                self.target_net.load_state_dict(target_net_state_dict)

                # LogUtils.info('TRAIN', f'episode: {i_episode}_{t}, time_slot: {time_slot}, reward: {reward.item()}')
                if done:
                    avg_reward = sum_reward / num_step
                    if best_reward < avg_reward:
                        best_reward = avg_reward
                        torch.save(self.policy_net.state_dict(), 'res/check_point/policy_model.pth')
                        LogUtils.info('TRAIN', f'SAVE MODEL: best_reward: {best_reward}')
                    self.rewards.append(avg_reward)
                    self.plot_rewards()
                    break

            LogUtils.info('TRAIN', f'({i_episode}/{num_episodes}) reward: {sum_reward / num_step}')

        self.plot_rewards(show_result = True)
        plt.ioff()
        plt.show()
