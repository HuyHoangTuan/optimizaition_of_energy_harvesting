import numpy as np
class QLearning:
    def __init__(
            self,
            n_observations,
            n_actions,
            num_models = 2,
    ):
        self.Q_Functions = []
        self.N = []
        self.n_actions = n_actions
        self.num_models = num_models
        for i in range(num_models):
            self.Q_Functions.append({})
            self.N.append({})


    def set_Q(self, idx, state, action, value):
        if state not in self.Q_Functions:
            self.Q_Functions[idx][state] = np.zeros(self.n_actions)
        self.Q_Functions[idx][state][action] = value

    def get_Q(self, idx, state, action):
        if state not in self.Q_Functions:
            self.Q_Functions[idx][state] = np.zeros(self.n_actions)
        return self.Q_Functions[idx][state][action]

    def get_alpha(self, idx, state, action):
        if state not in self.N:
            self.N[idx][state] = np.zeros(self.n_actions)
        if self.N[idx][state] == 0:
            return 1
        else:
            return 1 / self.N[idx][state][action]