from modules.model import DQNModel

class DQNs:
    def __init__(
            self,
            device,
            n_observation,
            n_action,
            num_models=2
    ):
        self._device = device
        self._DQNs = []
        self._n_observation = n_observation
        self._n_action = n_action

        for i in range(num_models):
            self._DQNs.append(
                DQNModel(self._n_observation, self._n_action)
            )

    def __call__(self, idx = 0):
        return self._DQNs[idx]
