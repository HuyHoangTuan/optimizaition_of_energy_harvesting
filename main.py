from Environment import Environment
from DQN import DQN
import torch


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = Environment()
    model = DQN(env.num_state_space() , env.num_action_space()).to(device = device)
