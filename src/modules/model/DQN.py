import torch
import torch.nn as nn

class DQNModel(nn.Module):
    def __init__(self, state_space, action_space):
        super(DQNModel, self).__init__()
        self.layers = [
            nn.Linear(state_space, 24),
            nn.Linear(24, 24),
            nn.Linear(24, action_space)
        ]

    def _get_layer(self, index):
        index -= 1
        if index < 0:
            return self.layers[0]

        if index > len(self.layer):
            return self.layer[-1]

        return self.layer[index]

    def forward(self, x):
        x = torch.relu(self._get_layer(1)(x))
        x = torch.relu(self._get_layer(2)(x))
        return self._get_layer(3)(x)