import torch
import torch.nn as nn
import torch.nn.functional as F

class DQNModel(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQNModel, self).__init__()
        # lstm_hidden_size = 64
        # self.lstm = nn.LSTM(n_observations, lstm_hidden_size)

        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)  # Adjust the output size to match the number of actions

    def forward(self, x):
        # lstm_out, _ = self.lstm(x)

        # lstm_out = lstm_out.view(-1)

        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x
