import math
from utils import LogUtils, RandomUtils
class Environment:
    # [PU2, PU1]
    def __init__(
            self,
            P_max=1,
            Xi_s=0.1,
            Xi_pr=[0.1, 0.1],
            Xi_ps=[0.1, 0.1],
            Xi_p=[0.1, 0.1],
            Xi_sp=[0.1, 0.1],
            Lambda=[0.1, 0.1],
            N=20,
            I=[0.5, 0.5],
            Gamma=0.99,
            Alpha=0.003,
            Eta=0.9,
            T_s=1,
            N_0=1,
            E_max=0.2,
            Phi=1,
            Rho=0.4,
            A=10,
            C_max=0.5,
    ):
        # self.actions_space = [(0, s * 0.05) for s in range(1, N + 1, 1)] + [(1, s * 0.05) for s in range(1, N+1, 1)]
        self.P_max = P_max
        self.Xi_s = Xi_s
        self.Xi_pr = Xi_pr
        self.Xi_ps = Xi_ps
        self.Xi_p = Xi_p
        self.Xi_sp = Xi_sp
        self.Lambda = Lambda
        self.N = N
        self.I = I
        self.Gamma = Gamma
        self.Alpha = Alpha
        self.Eta = Eta
        self.T_s = T_s
        self.N_0 = N_0
        self.E_max = E_max
        self.Phi = Phi
        self.Rho = Rho
        self.A = A
        self.C_max = C_max

        # init
        self.init_state = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        self._P_p = [None, None]
        self._E_ambient = None
        self._g_s = None
        self._g_pr = [None, None]
        self._g_ps = [None, None]
        self._g_p = [None, None]
        self._g_sp = [None, None]
        self.reset()

        self.actions_space = []
        for i in range(0, 2 * self.N):
            pu = self._get_PU(None, (i % N) + 1)
            _range = 0
            _scale = 0
            if pu == 0:
                _range = self.C_max/(self.T_s)
                _scale = 0.1
            else:
                _range = self._get_I(None, (i % N) + 1)/(self.T_s)
                _scale = 0.01
            _P = RandomUtils.normal(_range, _scale)

            if i < self.N:
                self.actions_space.append((0, _P))
            else:
                self.actions_space.append((1, _P))
        RandomUtils.shuffle(self.actions_space)

    def _get_PU(self, action, time_slot):
        return 1 if time_slot > self.A else 0

    def _get_P(self, action, time_slot, pu=None):
        if pu is None:
            pu = self._get_PU(action, time_slot)

        # transmit power
        _, P = self.actions_space[action]
        # _P = 10 * math.log10(P/(10**-3))
        # return 10**(P / 10)
        return P
    def _get_P_p(self, action, time_slot, pu = None):
        if pu is None:
            pu = self._get_PU(action, time_slot)

        if self._P_p[pu] is None:
            self._P_p[pu] = RandomUtils.uniform(0, self.P_max, self.N)
        # print(f'time_slot = {time_slot}')
        # _P_p = 10 * math.log10(self._P_p[pu][time_slot - 1]/(10**-3))
        return self._P_p[pu][time_slot - 1]

    def _get_k(self, action, time_slot, pu=None):
        if pu is None:
            pu = self._get_PU(action, time_slot)

        # choice
        k, _ = self.actions_space[action]
        return k

    def _get_E_ambient(self, action, time_slot, pu=None):
        if pu is None:
            pu = self._get_PU(action, time_slot)

        if self._E_ambient is None:
            self._E_ambient = RandomUtils.uniform(0, self.E_max, self.N)

        # ambient resources
        return self._E_ambient[time_slot - 1]

    def _get_E_TS(self, action, time_slot, pu=None):
        if pu is None:
            pu = self._get_PU(action, time_slot)

        # energy harvested from the PU(time_slot)
        return self.Rho * self.T_s * self._get_P_p(action, time_slot, pu) * self.Eta * self._get_g_ps(action, time_slot, pu)

    def _get_g_s(self, action, time_slot, pu=None):
        if pu is None:
            pu = self._get_PU(action, time_slot)

        if self._g_s is None:
            self._g_s = RandomUtils.exponential(1.0 / self.Xi_s, self.N)

        return self._g_s[time_slot - 1]

    def _get_g_pr(self, action, time_slot, pu=None):
        if pu is None:
            pu = self._get_PU(action, time_slot)

        if self._g_pr[pu] is None:
            self._g_pr[pu] = RandomUtils.exponential(1.0 / self.Xi_pr[pu], self.N)
        return self._g_pr[pu][time_slot - 1]

    def _get_g_ps(self, action, time_slot, pu=None):
        if pu is None:
            pu = self._get_PU(action, time_slot)

        if self._g_ps[pu] is None:
            self._g_ps[pu] = RandomUtils.exponential(1.0 / self.Xi_ps[pu], self.N)

        return self._g_ps[pu][time_slot - 1]

    def _get_g_p(self, action, time_slot, pu=None):
        if pu is None:
            pu = self._get_PU(action, time_slot)

        if self._g_p[pu] is None:
            self._g_p[pu] = RandomUtils.exponential(1.0 / self.Xi_p[pu], self.N)

        return self._g_p[pu][time_slot - 1]

    def _get_g_sp(self, action, time_slot, pu=None):
        if pu is None:
            pu = self._get_PU(action, time_slot)

        if self._g_sp[pu] is None:
            self._g_sp[pu] = RandomUtils.exponential(1.0 / self.Xi_sp[pu], self.N)
        return self._g_sp[pu][time_slot - 1]

    def _get_mu(self, action, time_slot, pu=None):
        if pu is None:
            pu = self._get_PU(action, time_slot)

        Lambda = self.Lambda[pu]
        if self._get_P_p(action, time_slot, pu) >= Lambda:
            return 1 - self.Rho
        else:
            return 1

    def _get_Hs(self, action, time_slot, pu=None):
        if pu is None:
            pu = self._get_PU(action, time_slot)

        Lambda = self.Lambda[pu]
        if self._get_P_p(action, time_slot, pu) >= Lambda:
            return self._get_E_TS(action, time_slot, pu) + self._get_E_ambient(action, time_slot, pu)
        else:
            return self._get_E_ambient(action, time_slot, pu)

    def _get_I(self, action, time_slot, pu=None):
        if pu is None:
            pu = self._get_PU(action, time_slot)

        return self.I[pu]

    def _harvest_energy(self, action, time_slot, pu=None):
        if pu is None:
            pu = self._get_PU(action, time_slot)

        return self._get_Hs(action, time_slot, pu)

    def _get_record(self, action, time_slot):
        # record: (v, k, mu, E, C, P)

        if time_slot <= 0:
            return 0, 0, 0, 0, 0, 0
        else:
            if time_slot > len(self.records):
                return None
            else:
                return self.records[time_slot - 1]

    def _add_record(self, record):
        self.records.append(record)

    def _get_reward(self, action, time_slot, pu=None):
        if pu is None:
            pu = self._get_PU(action, time_slot)

        if pu == 0:
            inside_log = 1 + self._get_P(action, time_slot, pu) * self._get_g_s(action, time_slot, pu) / self.N_0
            print(inside_log)
            return self._get_mu(action, time_slot, pu) * self.T_s * math.log(inside_log, 2)
        else:
            inside_log = 1 + self._get_P(action, time_slot, pu) * self._get_g_s(action, time_slot, pu) / (
                        self.N_0 + self._get_P_p(action, time_slot, pu) * self._get_g_pr(action, time_slot, pu))
            return self._get_mu(action, time_slot, pu) * self.T_s * math.log(inside_log, 2)

    def reset(self):
        self.records = []
        self.TimeSlot = 0



        return self.init_state, None

    def step(self, action):
        # LogUtils.info('Environment', f'action: {action}, time_slot: {self.TimeSlot}')
        # return: state, action, reward, time_slot
        prev_v, prev_k, prev_mu, prev_E, prev_C, prev_P = self._get_record(action, self.TimeSlot)

        self.TimeSlot += 1
        C = min(prev_C + prev_k * prev_E - (1 - prev_k) * prev_mu * prev_P * self.T_s, self.C_max)
        v = self._get_PU(action, self.TimeSlot)

        E = self._harvest_energy(action, self.TimeSlot, v)
        k = self._get_k(action, self.TimeSlot, v)
        mu = self._get_mu(action, self.TimeSlot, v)
        P = self._get_P(action, self.TimeSlot, v)
        g_sp = self._get_g_sp(action, self.TimeSlot, v)
        I = self._get_I(action, self.TimeSlot, v)
        # print(f'k = {k}, E = {E}, mu = {mu}, P = {P}, C = {C}, k*E = {k*E}, (1-k)*mu*P*T_s = {(1-k)*mu*P*self.T_s}')

        R = None
        if k == 0:
            if P * self.T_s <= C and v * P * g_sp <= I:
                R = self._get_reward(action, self.TimeSlot, v)
        else:
            if P * self.T_s > C:
                R = 0

        if R is None:
            R = -self.Phi
        print(f'k = {k}, v = {v}, P = {P}, P*T_s = {P * self.T_s}, v*P*g_sp = {v*P * g_sp}, E = {E}, I = {I}, C = {C}, R = {R}')
        R = self._reward_shift(R)

        state = (
            v,
            prev_E,
            C,
            self._get_g_s(action, self.TimeSlot),
            self._get_g_pr(action, self.TimeSlot, 1),
            self._get_g_ps(action, self.TimeSlot, 1),
            self._get_g_ps(action, self.TimeSlot, 0),
            self._get_g_sp(action, self.TimeSlot, 1),
            self._get_g_sp(action, self.TimeSlot, 0),
            self._get_g_p(action, self.TimeSlot, 1),
            self._get_g_p(action, self.TimeSlot, 0)
        )

        # print(f'g_sp_1 = {self._g_sp[1]}\ng_sp_2 = {self._g_sp[0]}')
        # print(f'g_s = {self._g_s}')
        record = (v, k, mu, E, C, P)
        self._add_record(record)
        return state, (k, P), R, self.TimeSlot

    def get_num_states(self):
        return 11

    def get_num_actions(self):
        return len(self.actions_space)

    def _reward_shift(self, reward):
        return reward