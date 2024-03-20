import numpy as np
import torch

class QFunction:
    def __init__(self, device, num_actions):
        self._device = device
        self.num_actions = num_actions
        self.states = {}
        self.N = {}

    def _check_state(self, state):
        if state not in self.states:
            self.states[state] = torch.zeros(self.num_actions,  dtype=torch.float32, device=self._device)

        if state not in self.N:
            self.N[state] = torch.zeros(self.num_actions,  dtype=torch.float32, device=self._device)

    def __call__(self, state):
        self._check_state(state)
        return self.states[state]

    def __add__(self, target):
        states = list(self.states.keys()) + list(target.states.keys())
        Q = QFunction(self._device, self.num_actions)
        for state in states:
            Q.states[state] = torch.add(self.__call__(state), target(state))
        return Q

    def get_learning_rate(self, state, action):
        self._check_state(state)
        _zero = torch.zeros(0, dtype=torch.float32, device=self._device)
        return torch.tensor(1, dtype=torch.float32, device=self._device) if torch.equal(self.N[state][action], _zero) else 1.0/self.N[state][action]

    def update_learning_rate(self, state, action):
        self._check_state(state)
        self.N[state][action] = torch.add(self.N[state][action], 1)


class QLearning:
    def __init__(
            self,
            device,
            n_observations,
            n_actions,
            num_models=2
    ):
        self._device = device
        self.QFunctions = []
        self.n_actions = n_actions
        for i in range(num_models):
            self.QFunctions.append(QFunction(self._device, n_actions))

    def __call__(self, idx):
        return self.QFunctions[idx]

    def get_Q(self):
        Q = QFunction(self._device, self.n_actions)
        for i in range(len(self.QFunctions)):
            Q = Q + self.QFunctions[i]

        for state in list(Q.states.keys()):
            Q.states[state] = torch.div(torch.Q.states[state], len(self.QFunctions))

        return Q
