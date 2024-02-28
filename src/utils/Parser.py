class Parser:
    def __init__(self, model_name, log_path):
        self._model_name = model_name
        self._log_path = log_path
        self._rewards, self._rates = self._read_file()
    def _read_file(self):
        with open(self._log_path, 'r') as f:
            if self._model_name == 'dqn':
                return self._parse_for_dqn(f)
            elif self._model_name == 'risk_averse':
                return self._parse_for_risk_averse(f)

    def _parse_for_dqn(self, _file):
        lines = _file.readlines()
        rewards = []
        rates = []
        for line in lines:
            prefix = '[TRAIN]: ('
            if line.startswith(prefix):
                start_position = line.find(':', line.find('reward:')) + 1
                rewards.append(
                    float(line[start_position:line.find(',', start_position)].strip())
                )

                start_position = line.find(':', line.find('rates:')) + 1
                rates.append(
                    float(line[start_position:line.find(',', start_position)].strip())
                )
        return rewards, rates

    def _parse_for_risk_averse(self, _file):
        return [], []

    def get_rewards(self):
        return self._rewards

    def get_rates(self):
        return self._rates