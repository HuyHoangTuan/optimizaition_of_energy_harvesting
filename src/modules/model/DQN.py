import torch
import torch.nn as nn
import torch.nn.functional as F

class DQNModel(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQNModel, self).__init__()
        self.layer1 = nn.Linear(n_observations, 24)
        self.layer2 = nn.Linear(24, 24)
        self.layer3 = nn.Linear(24, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
