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
        self.actions_space = [(0, s * 0.05) for s in range(1, 11, 1)] + [(1, s * 0.05) for s in range(1, 11, 1)]
        self.records = []

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
        self.TimeSlot = 0
        self.init_state = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

        # init
        self._E_ambient = None
        self._g_s = None
        self._g_pr = [None, None]
        self._g_ps = [None, None]
        self._g_p = [None, None]
        self._g_sp = [None, None]


    def _get_PU(self, action, time_slot):
        return 0 if time_slot > self.A else 1

    def _get_P(self, action, time_slot, pu=None):
        if pu is None:
            pu = self._get_PU(action, time_slot)

        # transmit power
        _, P = self.actions_space[action]
        return 10**(P / 10)
        # return P

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
        return self.Rho * self.T_s * self._get_P(action, time_slot, pu) * self.Eta * self._get_g_ps(action, time_slot,
                                                                                                    pu)

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
        if self._get_P(action, time_slot, pu) >= Lambda:
            return 1 - self.Rho
        else:
            return 1

    def _get_Hs(self, action, time_slot, pu=None):
        if pu is None:
            pu = self._get_PU(action, time_slot)

        Lambda = self.Lambda[pu]
        if self._get_P(action, time_slot, pu) >= Lambda:
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
            return self._get_mu(action, time_slot, pu) * self.T_s * math.log(inside_log, 2)
        else:
            inside_log = 1 + self._get_P(action, time_slot, pu) * self._get_g_s(action, time_slot, pu) / (
                        self.N_0 + self._get_P(action, time_slot, pu) * self._get_g_pr(action, time_slot, pu))
            return self._get_mu(action, time_slot, pu) * self.T_s * math.log(inside_log, 2)

    def reset(self):

        self.TimeSlot = 0
        self.records = []

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

        R = None
        if k == 0:
            if P * self.T_s <= C and P * g_sp <= I:
                R = self._get_reward(action, self.TimeSlot, v)
        else:
            if P * self.T_s > C:
                R = 0

        if R is None:
            R = -self.Phi

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
        record = (v, k, mu, E, C, P)
        self._add_record(record)
        return state, (k, P), R, self.TimeSlot

    def get_num_states(self):
        return 11

    def get_num_actions(self):
        return len(self.actions_space)

    def _reward_shift(self, reward):
        return reward + self.Phi