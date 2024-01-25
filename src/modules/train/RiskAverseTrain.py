import math
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
import matplotlib.pyplot as plt

# from utils import LogUtils
from utils import RandomUtils
from modules.environment import Environment
from modules.model import DQNModel
from modules.relay import ReplayMemory, Transition

class RiskAverseTrain:
    def __init__(
            self,
            episodes = 1600,
            eps_max = 1,
            eps_min = 0.01,
            eps_decay = 0.001,
            alpha = 0.003
    ):

        self._episodes = episodes
        self._eps_max = eps_max
        self._eps_min = eps_min
        self._eps_decay = eps_decay
        self._eps_drop_rate = 0.0
        self._alpha = alpha
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._init_env()

    def _select_action(self, net, state):
        sample = RandomUtils.custom_random()
        eps_threshold = self._eps_min + (self._eps_max - self._eps_min) * math.exp(
            -1. * self._eps_drop_rate * self._eps_decay
        )
        if sample > eps_threshold:
            with torch.no_grad():
                return torch.argmax(
                    net['policy'](
                        torch.tensor(state, dtype = torch.float32, device = self._device)
                    ), dim = 1)
        else:
            _ = [s for s in range(net['env'].get_num_actions())]
            return torch.tensor(RandomUtils.sample(_, 1), dtype = torch.long, device = self._device)

    def _create_net(self, env):
        return {
            'env': env,
            'policy': DQNModel(
                n_observations = env.get_num_states(),
                n_actions = env.get_num_actions()
            ).to(self._device),
            'target': DQNModel(
                n_observations = env.get_num_states(),
                n_actions = env.get_num_actions()
            ).to(self._device),
            'relay': ReplayMemory(100000)
        }

    def _init_env(self):
        self._envs = [
            Environment(
                Episode = self._episodes
            )
        ]

        self._nets = []
        for i in range(len(self._envs)):
            self._nets.append(
                self._create_net(self._envs[i])
            )
