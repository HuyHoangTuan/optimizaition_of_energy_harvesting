class QLearning:
    def __init__(
            self,
            n_observations,
            n_actions,
            num_models = 2,
    ):
        self.Q_Functions = []
        self.num_models = num_models
        for i in range(num_models):
            self.Q_Functions.append({})

