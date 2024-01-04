import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple , deque
import random

Transition = namedtuple('Transition' , ('state' , 'action' , 'next_state' , 'reward'))
class ReplayMemory(object):
  def __init__(self, capacity):
    self.memory = deque([] , maxlen=capacity)

  def push(self , *args):
    self.memory.append(Transition(*args))

  def sample(self, batch_size):
    return random.sample(self.memory , batch_size)

  def __len__(self):
    return len(self.memory)


class DQN(nn.Module):
  def __init__(self, n_observations , n_actions):
    super(DQN, self).__init__()
    #self.layer1 = nn.Linear(n_observations , 128)
    self.layer1 = nn.BatchNorm1d(n_observations)
    self.layer2 = nn.LSTM(n_observations , 128 , 10)
    #self.layer3 = nn.Linear(256, 256)
    #self.layer4 = nn.Linear(256 , 256)
    #self.layer5 = nn.Linear(256 , 256)
    self.layer6 = nn.Linear(128, n_actions)

  def forward(self, x):
    #x = F.relu(self.layer1(x))
    x = self.layer1(x)
    x, _ = self.layer2(x)
    #x = F.relu(self.layer2(x))
    #x = F.relu(self.layer3(x))
    #x = F.relu(self.layer4(x))
    #x = F.relu(self.layer5(x))
    return F.softmax(self.layer6(x), dim = 1)

