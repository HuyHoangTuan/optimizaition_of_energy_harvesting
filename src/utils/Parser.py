import torch
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

    def _parse_array(self, line, key, is_float=True):
        if line.find(key) == -1:
            return []

        start_position = line.find('[', line.find(key)) + 1
        end_position = line.find(']', start_position)

        _str = line[start_position: end_position]
        _list = _str.split(',')
        if len(_str) == 0:
            return []

        for i in range(0, len(_list)):
            if is_float is True:
                _list[i] = float(_list[i])
            else:
                _list[i] = int(_list[i])
        return _list

    def _parse_for_dqn(self, _file):
        lines = _file.readlines()
        rewards = []
        rates = []
        loss = []
        rho = []
        transmit_actions = []
        P = []
        bound = []
        reward_t = []
        rate_t = []
        rho_t = []
        transmit_actions_t = []
        bound_t = []
        C_t = []
        P_t = []
        for line in lines:
            prefix = '[TRAIN]: ('
            if line.startswith(prefix):
                rewards.append(
                    self._parse_float(line, 'reward:')
                )
                reward_t.extend(
                    self._parse_array(line, 'reward_t:')
                )

                rates.append(
                    self._parse_float(line, 'rates:')
                )
                rate_t.append(
                    self._parse_array(line, 'rate_t:')
                )

                loss.append(self._parse_float(line, 'loss:'))

                rho.append(self._parse_float(line, 'rho:'))
                rho_t.append(
                    self._parse_array(line, 'rho_t:')
                )
                
                transmit_actions.append(
                    self._parse_int(line, 'transmit_actions:')
                )
                transmit_actions_t.append(
                    self._parse_array(line, 'transmit_actions_t:', is_float=False)
                )
                bound.append(
                    self._parse_float(line, 'bound:')
                )
                bound_t.append(
                    self._parse_array(line, 'bound_t:')
                )
                C_t.append(
                    self._parse_array(line, 'C: [')
                )

                # P.append(
                #     torch.mean(
                #         torch.tensor(
                #             self._parse_array(line, 'P:'),
                #             dtype=torch.float
                #         )
                #     ).item()
                # )
                P.extend(self._parse_array(line, 'P:'))
                P_t.append(self._parse_array(line, 'P:'))

        self._rewards = rewards
        self._reward_t = reward_t
        self._rates = rates
        self._rate_t = rate_t
        self._bound = bound
        self._bound_t = bound_t
        self._loss = loss
        self._rho = rho
        self._rho_t = rho_t
        self._transmit_actions = transmit_actions
        self._transmit_actions_t = transmit_actions_t
        self._P = P
        self._C = C_t
        self._P = P_t

    def _parse_for_risk_averse(self, _file):
        return [], []

    def get_rewards(self):
        return self._rewards

    def get_rates(self):
        return self._rates

    def get_C(self):
        return self._C, self._P

    def get_data(self):
        return self._rewards, self._reward_t, self._rates, self._rate_t, self._loss, self._rho, self._rho_t, self._bound, self._bound_t, self._transmit_actions, self._transmit_actions_t, self._P