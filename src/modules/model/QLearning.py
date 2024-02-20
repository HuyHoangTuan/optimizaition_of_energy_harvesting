import numpy as np


class QFunction:
    def __init__(self, num_actions):
        self.num_actions = num_actions
        self.states = {}
        self.N = {}

    def _check_state(self, state):
        if state not in self.states:
            self.states[state] = np.zeros(self.num_actions)

        if state not in self.N:
            self.N[state] = np.zeros(self.num_actions)

    def __call__(self, state):
        self._check_state(state)
        return self.states[state]

    def __add__(self, target):
        states = list(self.states.keys()) + list(target.states.keys())
        Q = QFunction(self.num_actions)
        for state in states:
            Q.states[state] = np.sum(self.__call__(state), target(state))

        return Q

    def get_learning_rate(self, state, action):
        self._check_state(state)
        return self.N[state][action] == 0 if self.N[state][action] == 0 else self.N[state][action]

    def update_learning_rate(self, state, action):
        self._check_state(state)
        self.N[state][action] = self.N[state][action] + 1


class QLearning:
    def __init__(
            self,
            n_observations,
            n_actions,
            num_models=2,
    ):
        self.QFunctions = []
        self.n_actions = n_actions
        for i in range(num_models):
            self.QFunctions.append(QFunction(n_actions))

    def __call__(self, idx):
        return self.QFunctions[idx]

    def get_Q(self):
        Q = QFunction(self.n_actions)
        for i in range(len(self.QFunctions)):
            Q = Q + self.QFunctions[i]

        for state in list(Q.states.keys()):
            Q.states[state] = Q.states[state] / len(self.QFunctions)

        return Q
