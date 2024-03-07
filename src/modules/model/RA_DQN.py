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
                {
                    "target": DQNModel(self._n_observation, self._n_action),
                    "policy": DQNModel(self._n_observation, self._n_action)
                }
            )
            self._DQNs[-1]["target"].load_state_dict(self._DQNs[-1]["policy"].state_dict())

    def __call__(self, idx, module):
        return self._DQNs[idx][module]
