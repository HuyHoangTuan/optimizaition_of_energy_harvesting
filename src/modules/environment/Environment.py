import math

import numpy as np
from utils import LogUtils, RandomUtils

class Environment:
    def __init__(
            self,
            NumPU = 2,
            P_max = 1,
            Xi_s = 0.1,
            Xi_pr = [0.1, 0.1],
            Xi_ps = [0.1, 0.1],
            Xi_p = [0.1, 0.1],
            Xi_sp = [0.1, 0.1],
            I = [0.5, 0.5],
            Lambda = 0.1,
            N = 20,
            # Gamma = 0.99,
            # Alpha = 0.003,
            Eta = 0.9,
            T_s = 1,
            N_0 = 1,
            E_max = 0.2,
            Phi = 1,
            Rho = 0.4,
            A = 10,
            C_max = 0.5,
            Episode = 1400,
            Dynamic_Rho = False,
            reward_function_id = 0
    ):
        # SU-Tx ~ Agent
        # reward ~ instantaneous achievable rate
        # k = 0 -> transmit message
        # k = 1 -> harvest energy
        # s.t.
        # Agent always harvest energy even when transmitting message
        # Thus, Rho represents the ratio time that Agent can harvest energy in the time slot
        # mu = 1 - Rho represents the ratio time that Agent can transmit message in the time slot

        LogUtils.info('ENV', 'Started init Environment')

        self.Reward_Function_ID = reward_function_id
        self.NumPU = NumPU
        self.P_max = P_max
        self.I = I
        self.Lambda = Lambda
        self.N = N
        # self.Gamma = Gamma
        # self.Alpha = Alpha
        self.Eta = Eta
        self.T_s = T_s
        self.N_0 = N_0
        self.E_max = E_max
        self.Phi = Phi
        self.Rho = Rho
        self.Is_Dynamic_Rho = Dynamic_Rho
        self.A = A
        self.C_max = C_max
        self.Episode = Episode

        # init

        # g_s: [ [Episode][Time_Slot] ]
        # The channel gain power between SU-TX and SU-RX
        self.g_s = []
        for i in range(self.Episode):
            self.g_s.append(RandomUtils.rayleigh(Xi_s, self.N))

        # g_pr: [ [Episode][PU][Time_Slot] ]
        # The channel power gain between PU and SU-RX
        self.g_pr = []
        for i in range(self.Episode):
            self.g_pr.append([])
            for j in range(NumPU):
                self.g_pr[i].append(RandomUtils.rayleigh(Xi_pr[j], self.N))

        # g_sp: [ [Episode][PU][Time_Slot] ]
        # The channel power gain between SU-TX and PU-RX
        self.g_sp = []
        for i in range(self.Episode):
            self.g_sp.append([])
            for j in range(NumPU):
                self.g_sp[i].append(RandomUtils.rayleigh(Xi_sp[j], self.N))

        # g_ps: [ [Episode][PU][Time_slot] ]
        # The channel power gain between PU and SU-TX
        self.g_ps = []
        for i in range(self.Episode):
            self.g_ps.append([])
            for j in range(NumPU):
                self.g_ps[i].append(RandomUtils.rayleigh(Xi_ps[j], self.N))

        # P_P: [ [Episode][PU][Time_Slot] ]
        self._P_p = []
        for i in range(self.Episode):
            self._P_p.append([])
            for j in range(0, NumPU):
                self._P_p[i].append(RandomUtils.uniform(0, self.P_max, self.N))

        # E_ambient: [ [Episode][Time_Slot] ]
        self._E_ambient = []
        for i in range(self.Episode):
            self._E_ambient.append(RandomUtils.uniform(0, self.E_max, self.N))

        # action spaces
        # Huy's  edition
        k = [0]
        delta_P = 1.0 / 32.0
        P = [i * delta_P for i in range(1, int(0.5 / delta_P) * 2 + 1)]
        delta_Rho = 1.0 / 8.0
        Rho = [i * delta_Rho for i in range(0, int(1.0 / delta_Rho))]
        actions_space = np.array(np.meshgrid(k, P, Rho)).T.reshape(-1, 3)
        actions_space = np.append(actions_space, np.array([[1, 1.0, 0]]), axis = 0)
        self.actions_space = [tuple(x) for x in actions_space]
        #
        # Cong's edition
        # k_value = [0]
        # Rho_value = [s * 0.1 for s in range(0, 10, 1)]
        # P_value = [s * 0.01 for s in range(1, 53, 1)]
        # actions_space = np.array(np.meshgrid(k_value, P_value, Rho_value)).T.reshape(-1, 3)
        # actions_space = np.append(actions_space, np.array([[1, 1.0, 0]]), axis = 0)
        # self.actions_space = [tuple(x) for x in actions_space]

        RandomUtils.shuffle(self.actions_space)
        LogUtils.info('ENV', f'ACTIONS_SPACE: {self.get_num_actions()}')

        # state: (v, E, C, p_p, g_s, g_sp, g_pr, g_ps)
        self._Default_State = (
            0,
            0,
            0,
            *tuple([0 for i in range(NumPU)]),
            0,
            *tuple([0 for i in range(NumPU)]),
            *tuple([0 for i in range(NumPU)]),
            *tuple([0 for i in range(NumPU)])
        )
        # records: (k, mu, E, C, P, g_s)
        self._Default_Record = (0, 0, 0, 0, 0, 0)
        self.records = []

        # time slot
        self._Time_Slot = 0

        self.reset()

        LogUtils.info('ENV', 'Finished init Environment')

    def _get_k(self, action):
        k, _, _ = self.actions_space[action]
        return k

    def _get_Rho(self, action):
        _, _, Rho = self.actions_space[action]
        if self.Is_Dynamic_Rho is False:
            return self.Rho
        return Rho

    def _get_P(self, action):
        _, P, _ = self.actions_space[action]
        return P

    def _get_v(self, Time_Slot):
        if Time_Slot <= self.A:
            return 1  # PU_1
        else:
            return 0  # PU_2

    def _calc_Interference(self, P_p, g_pr, Rho):
        Interference = 0
        for i in range(self.NumPU):
            P_p_dbw = 10 * math.log(P_p[i] * 1000, 10)
            P_p_dbw = 10 ** (P_p_dbw / 10)
            Interference += (1 - Rho) * self.T_s * P_p_dbw * g_pr[i]

        return Interference

    def _calc_E_TS(self, P_p, G_ps, Rho):
        if P_p >= self.Lambda:
            return Rho * self.T_s * P_p * self.Eta * G_ps
        else:
            return 0

    def _Is_Interference(self, P, g_sp):
        for i in range(self.NumPU):
            if P * g_sp[i] > self.I:
                return False

        return True

    def _get_record(self, time_slot):
        # record: (k, mu, E, C, P)

        if time_slot <= 0 or time_slot > len(self.records):
            return self._Default_Record
        else:
            return self.records[time_slot - 1]

    def _add_record(self, record):
        self.records.append(record)

    def _convert_2_dbW(self, PW):
        P_dbw = 10 * math.log(PW * 1000, 10)
        P_dbw = 10 ** (P_dbw / 10)
        return P_dbw

    def reset(self):
        self.records = []
        self._Time_Slot = 0

        return self._Default_State, None

    def step(self, action, episode):
        # LogUtils.info('Environment', f'action: {action}, time_slot: {self.TimeSlot}')
        # return: state, action, reward, time_slot
        prev_k, prev_mu, prev_E, prev_C, prev_P, prev_G_s = self._get_record(self._Time_Slot)

        self._Time_Slot += 1
        v = self._get_v(self._Time_Slot)
        k = self._get_k(action)
        P = self._get_P(action)
        Rho = self._get_Rho(action)
        mu = 1 - Rho

        G_s = self.g_s[episode][self._Time_Slot - 1]
        P_p = (np.array(self._P_p[episode])[:, self._Time_Slot - 1]).tolist()
        G_pr = (np.array(self.g_pr[episode])[:, self._Time_Slot - 1]).tolist()
        G_sp = (np.array(self.g_sp[episode])[:, self._Time_Slot - 1]).tolist()
        G_ps = (np.array(self.g_ps[episode])[:, self._Time_Slot - 1]).tolist()

        # todo: maybe need to multiply Rho into E_ambient, because 1-Rho is the ratio time that agent
        # transmit power, thus, rho ratio time that agent harvest energy
        E_ambient = self._E_ambient[episode][self._Time_Slot - 1]
        E_TS = self._calc_E_TS(P_p[v], G_ps[v], Rho)
        E = E_TS + E_ambient

        C = max(0.0, min(prev_C + prev_E - (1.0 - prev_k) * prev_mu * prev_P * self.T_s, self.C_max))

        R = 0
        R_type = 0
        if self.Reward_Function_ID == 0:
            R_type = 0
            R = -self.Phi
            if k == 0 and mu * P * self.T_s <= C:
                if P * G_sp[v] <= self.I[v]:
                    P_dbw = self._convert_2_dbW(P)
                    P_p_dbw = self._convert_2_dbW(P_p[v])

                    if v == 1:
                        R_type = 1
                        R = mu * self.T_s * math.log2(1 + (P_dbw * G_s) / (self.N_0 + P_p_dbw * G_pr[v]))
                    else:
                        R_type = 1
                        R = mu * self.T_s * math.log2(1 + (P_dbw * G_s) / self.N_0)
            else:
                if k == 1 and mu * P * self.T_s > C:
                    R_type = 2
                    R = 0
        elif self.Reward_Function_ID == 1:
            P_dbw = self._convert_2_dbW(P)
            P_p_dbw = self._convert_2_dbW(P_p[v])
            if k == 0 and mu * P * self.T_s <= C:
                if P * G_sp[v] <= self.I[v]:
                    R_type = 1
                    Infer = mu * self.T_s * (P_p_dbw * G_pr[v])
                    R = mu * self.T_s * math.log2(1 + (P_dbw * G_s) / (self.N_0 + Infer))
            elif k == 1:
                R_type = 2
                R = 0
            else:
                R_type = 3
                if mu * P * self.T_s > C:
                    R += - mu * self.T_s * math.log2(1 + (mu * P * self.T_s - C) * G_s / self.N_0)
                if P * G_sp[v] > self.I[v]:
                    R += - mu * self.T_s * math.log2(1 + (P * G_sp[v] - self.I[v]) / self.N_0)

        state = (
            v,
            prev_E,
            C,
            *tuple(P_p),
            G_s,
            *tuple(G_sp),
            *tuple(G_pr),
            *tuple(G_ps)
        )

        action = (
            k,
            P,
            Rho
        )

        reward = (
            R,
            R_type
        )

        # print(f'k = {k}, mu = {mu}, E = {E}, C = {C}, P = {P}, g_s = {g_s}')
        record = (k, mu, E, C, P, G_s)
        self._add_record(record)
        return state, action, reward, self._Time_Slot

    def get_num_states(self):
        return len(self._Default_State)

    def get_num_actions(self):
        return len(self.actions_space)
