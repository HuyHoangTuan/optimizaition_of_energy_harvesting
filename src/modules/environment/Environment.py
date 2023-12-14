import math
from utils import LogUtils, RandomUtils
class Environment:
    # [PU2, PU1]
    def __init__(
            self,
            Num_PU=2,
            P_max=1,
            Lambda=0.1,
            N=20,
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
        self.Lambda = Lambda
        self.N = N
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
        self.init_state = 0, 0, [], []

        self._P_p = []
        for i in range(0, Num_PU):
            self._P_p.append(RandomUtils.uniform(0, self.P_max, self.N))

        self._E_ambient = None
        self.reset()

        # todo: gen new actions_space
        self.actions_space = [(0, s * 0.01) for s in range(1, 53, 1)] + [(1, s * 0.01) for s in range(1, 53, 1)]
        RandomUtils.shuffle(self.actions_space)

    def _get_k(self, action):
        k, _ = self.actions_space[action]
        return k

    def _get_Rho(self, action):
        k = self._get_k(action)
        if k == 0:
            return self.Rho
        return 0

    def _get_mu(self, rho):
        return 1 - rho

    def _get_P(self, action):
        _, P = self.actions_space[action]
        return P

    def _get_P_p(self, time_slot):
        return self._P_p[time_slot - 1]

    def _get_E_ambient(self, time_slot):
        if self._E_ambient is None:
            self._E_ambient = RandomUtils.uniform(0, self.E_max, self.N)
        return self._E_ambient[time_slot - 1]

    def _get_E_TS(self, rho, P_p):
        E_TS = 0
        for i in range(0, len(P_p)):
            E_TS += rho * self.T_s * P_p[i] * self.Eta
        return E_TS

    def _get_I(self, rho, P_p, g):
        I = 0
        for i in range(0, len(P_p)):
            I += rho * self.T_s * P_p[i] * g[i]

        return I

    def _get_record(self, action, time_slot):
        # record: (v, k, mu, E, C, P)

        if time_slot <= 0:
            return 0, 0, 0, 0, 0
        else:
            if time_slot > len(self.records):
                return None
            else:
                return self.records[time_slot - 1]

    def _add_record(self, record):
        self.records.append(record)

    def reset(self):
        self.records = []
        self.TimeSlot = 0

        return self.init_state, None

    def step(self, action):
        # LogUtils.info('Environment', f'action: {action}, time_slot: {self.TimeSlot}')
        # return: state, action, reward, time_slot
        prev_k, prev_mu, prev_E, prev_C, prev_P = self._get_record(action, self.TimeSlot)

        self.TimeSlot += 1
        C = min(prev_C + prev_E - (1 - prev_k) * prev_mu * prev_P * self.T_s, self.C_max)

        k = self._get_k(action)

        Rho = self._get_Rho(action)
        mu = self._get_mu(Rho)
        P_p = self._P_p[:, self.TimeSlot - 1]

        P = self._get_P(action)

        E_ambient = self._get_E_ambient(self.TimeSlot)
        E_TS = self._get_E_TS(Rho, P_p)
        E = E_TS + E_ambient

        I = self._get_I(Rho, P_p, g)

        R = None
        if k == 0 and mu * P * self.T_s <= C:
            P_dbw = 10 * math.log(P * 1000, 10)
            P_dbw = 10 ** (P_dbw / 10)
            inside_log = 1 + P_dbw * g_s / (self.N_0 + I)
            R = mu * self.T_s * math.log(inside_log,2)
        else:
            if k == 1:
                R = 0

        if R is None:
            R = -self.Phi

        state = (
            prev_E,
            g,
            g_s
        )

        record = (k, mu, E, C, P)
        self._add_record(record)
        return state, (k, P, Rho), R, self.TimeSlot

    def get_num_states(self):
        return 11

    def get_num_actions(self):
        return len(self.actions_space)
