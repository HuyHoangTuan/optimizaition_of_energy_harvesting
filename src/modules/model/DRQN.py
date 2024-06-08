import torch
import torch.nn as nn
import torch.nn.functional as F

class DRQNModel(nn.Module):
    def __init__(self, n_observations, n_actions, device):
        super(DRQNModel, self).__init__()
        # lstm_hidden_size = 64
        # self.lstm = nn.LSTM(n_observations, lstm_hidden_size)
        
        self.conv1 = nn.Conv1d(n_observations, 128, 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.gru = nn.GRU(128, 128)
        self.hidden_layer = torch.zeros(1, 128, device=device)
        self.layer1 = nn.Linear(128, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, n_actions)  # Adjust the output size to match the number of actions

    def forward(self, x):
        # lstm_out, _ = self.lstm(x)

        # lstm_out = lstm_out.view(-1)
        # print(x.shape)
        x = x.permute(1, 0)
        
        x = F.relu(self.conv1(x))
        # print(x.shape)
        x = F.relu(self.conv2(x))
        # print(x.shape)
        x = x.permute(1, 0)
        # print(x.shape)
        # print(x.shape)
        # print(self.hidden_layer)
        x, _ = self.gru(x)

        # print(x)
        x = F.relu(self.layer1(F.relu(x)))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        # print(x.shape)
        return x
