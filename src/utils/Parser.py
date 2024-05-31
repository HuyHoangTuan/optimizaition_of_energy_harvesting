class Parser:
    def __init__(self, model_name, log_path):
        self._model_name = model_name
        self._log_path = log_path
        self._read_file()

    def _read_file(self):
        with open(self._log_path, 'r') as f:
            if self._model_name == 'dqn':
                return self._parse_for_dqn(f)
            elif self._model_name == 'risk_averse':
                return self._parse_for_risk_averse(f)

    def _parse_float(self, line, key):
        start_position = line.find(':', line.find(key)) + 1
        return float(line[start_position: line.find(',', start_position)].strip())

    def _parse_int(self, line, key):
        start_position = line.find(':', line.find(key)) + 1
        return int(line[start_position: line.find(',', start_position)].strip())

    def _parse_array(self, line, key):
        start_position = line.find('[', line.find(key)) + 1
        end_position = line.find(']', line.find(']', start_position))
        _str = line[(start_position + 1): (end_position - 1)]
        _list = _str.split(',')
        for i in range(0, len(_list)):
            _list[i] = float(_list[i])
        return _list

    def _parse_for_dqn(self, _file):
        lines = _file.readlines()
        rewards = []
        rates = []
        loss = []
        rho = []
        transmit_actions = []
        P = []
        for line in lines:
            prefix = '[TRAIN]: ('
            if line.startswith(prefix):
                rewards.append(
                    self._parse_float(line, 'reward:')
                )

                rates.append(
                    self._parse_float(line, 'rates:')
                )

                loss.append(self._parse_float(line, 'loss:'))

                rho.append(self._parse_float(line, 'rho:'))

                transmit_actions.append(self._parse_int(line, 'transmit_actions:'))

                P.extend(self._parse_array(line, 'P:'))

        self._rewards = rewards
        self._rates = rates
        self._loss = loss
        self._rho = rho
        self._transmit_actions = transmit_actions
        self._P = P

    def _parse_for_risk_averse(self, _file):
        return [], []

    def get_rewards(self):
        return self._rewards

    def get_rates(self):
        return self._rates

    def get_data(self):
        return self._rewards, self._rates, self._loss, self._rho, self._transmit_actions, self._P