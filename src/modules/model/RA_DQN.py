from modules.model import DQNModel
import torch.optim as optim
from torch import nn
import torch


class DQNs:
    def __init__(
            self,
            device,
            n_observation,
            n_action,
            alpha = 0.003,
            num_models=2
    ):
        self._device = device
        self._DQNs = []
        self._optimizers = []
        self._n_observation = n_observation
        self._n_action = n_action

        for i in range(num_models):
            self._DQNs.append(
                DQNModel(self._n_observation, self._n_action).to(device)
            )
            self._optimizers.append(
                optim.SGD(self._DQNs[-1].parameters(), lr=alpha)
            )

    def __call__(self, idx = 0):
        return self._DQNs[idx]

    def loss(self, idx, values, expected_values):
        criterion = nn.MSELoss()
        _loss = criterion(values, expected_values)
        self._optimizers[idx].zero_grad()
        _loss.backward()
        torch.nn.utils.clip_grad_value_(self._DQNs[idx].parameters(), 100)
        self._optimizers[idx].step()

        return _loss