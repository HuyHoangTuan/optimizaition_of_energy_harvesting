import torch
import torch.nn as nn
import torch.nn.functional as F

class DQNModel(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQNModel, self).__init__()
        lstm_hidden_size = 64
        

        self.layer1 = nn.Linear(n_observations, 128)
        self.lstm = nn.LSTM(128, lstm_hidden_size)
        self.layer2 = nn.Linear(lstm_hidden_size, 64)
        self.layer3 = nn.Linear(64, n_actions)  # Adjust the output size to match the number of actions

    def forward(self, x):
        # lstm_out, _ = self.lstm(x)
        # print(lstm_out.shape)
        # print(x.shape)
        x = F.relu(self.layer1(x))
        # print(x.shape)
        x, _ = self.lstm(x)
        x = F.relu(self.layer2(x))
        return self.layer3(x)
