import numpy as np
import random
import math
class Environment:
  def __init__(self, P_max = 1 , Xi = 0.1 , Lambda = 0.1 , N = 20 , I = 0.5 , Gamma = 0.99 , \
               Alpha = 0.003 , Eta = 0.9 , T_s = 1 , N_0 = 1 \
               , E_max = 0.2 ,\
               Phi = 1 , Rho = 0.4 , A = 10):
    #self.actions_space = [(0 , s * 0.05) for s in range(1 , 11 , 1)] + [(1,s * 0.05) for s in range(1 , 11 , 1)]
    self.actions_space = [(0, s * 0.025) for s in range(1, 25, 1)] + [(1, s * 0.025) for s in range(1, 25, 1)]
    self.P_max = P_max
    self.Xi = Xi
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
    self.C_max = 0.5
    self.TimeSlot = -1

    self.arr_P_p1 = np.loadtxt('P_p1.txt' , delimiter = ' ')
    self.arr_P_p2 = np.loadtxt('P_p2.txt' , delimiter = ' ')
    self.arr_C_0 = np.loadtxt('C_0.txt' , delimiter = ' ')
    self.arr_E_ambient = np.loadtxt('E_ambient.txt' , delimiter = ' ')
    self.arr_g_s = np.loadtxt('g_s.txt' , delimiter = ' ')
    self.arr_g_p1r = np.loadtxt('g_p1r.txt' , delimiter = ' ')
    self.arr_g_p1s = np.loadtxt('g_p1s.txt' , delimiter = ' ')
    self.arr_g_p2s = np.loadtxt('g_p2s.txt' , delimiter = ' ')
    self.arr_g_sp1 = np.loadtxt('g_sp1.txt' , delimiter = ' ')
    self.arr_g_sp2 = np.loadtxt('g_sp2.txt' , delimiter = ' ')
    self.arr_g_p1 = np.loadtxt('g_p1.txt' , delimiter = ' ')
    self.arr_g_p2 = np.loadtxt('g_p2.txt' , delimiter = ' ')

  def harvest_energy(self, v):
    E_ambient = self.arr_E_ambient[self.TimeSlot]

    P_p1 = self.arr_P_p1[self.TimeSlot]
    P_p2 = self.arr_P_p2[self.TimeSlot]

    Rho = self.Rho
    if(v == 1 and P_p1 < self.Lambda) or (v == 0 and P_p2 < self.Lambda):
      Rho = 0

    E_TS = Rho * self.T_s * self.Eta
    E_TS *= P_p1 * self.g_p1s if v == 1 else P_p2 * self.g_p2s

    self.Mu = 1 - Rho
    E_h = E_TS + E_ambient

    return E_h

  def num_action_space(self):
    return len(self.actions_space)

  def num_state_space(self):
    return 11
  def get_g(self):
    self.g_s = self.arr_g_s[self.TimeSlot]
    self.g_p1r = self.arr_g_p1r[self.TimeSlot]
    self.g_p1s = self.arr_g_p1s[self.TimeSlot]
    self.g_p2s = self.arr_g_p2s[self.TimeSlot]
    self.g_sp1 = self.arr_g_sp1[self.TimeSlot]
    self.g_sp2 = self.arr_g_sp2[self.TimeSlot]
    self.g_p1 = self.arr_g_p1[self.TimeSlot]
    self.g_p2 = self.arr_g_p2[self.TimeSlot]

  def reset(self):
    self.TimeSlot += 1
    v = 1 if self.TimeSlot % self.N < self.A else 0
    # v = 1 cho pu1 v = 0 cho pu2
    self.E_prev = 0
    self.C = self.arr_C_0[int(self.TimeSlot / self.N)]
    self.get_g()

    return (v , self.E_prev , self.C , self.g_s , self.g_p1r ,self.g_p1s , self.g_p2s , self.g_sp1 , self.g_sp2 , self.g_p1 , self.g_p2 )

  def action_sampling(self):
    return random.sample([s for s in range(len(self.actions_space))], 1)[0]

  def map_action(self,num):
    return self.actions_space[num]
  def step(self, action):
    k, P = self.map_action(action)
    #tinh nang luong thu duoc buoc truoc va nang luong hien tai
    v = 1 if self.TimeSlot % self.N < self.A else 0
    E_h = self.harvest_energy(v)
    self.E_prev = E_h

    self.C = min(self.C + k * E_h - (1 - k) * self.Mu * P * self.T_s, self.C_max)
    self.C = max(self.C , 0)

    #tinh phan thuong tu hanh dong
    P_p1 = self.arr_P_p1[self.TimeSlot]
    P_dbW = 10 * math.log10(P)
    P_p1_dbW = 10 * math.log10(P_p1)
    R_1 = self.Mu * self.T_s * np.log2(1 + P_dbW * self.g_s / (self.N_0 + P_p1_dbW * self.g_p1r)) if v == 1 else 0
    R_2 = self.Mu * self.T_s * np.log2(1 + P_dbW * self.g_s / self.N_0) if v == 0 else 0
    if (k == 0 and v == 1 and P * self.T_s <= self.C and P * self.g_sp1 <= self.I):
      Reward = R_1
    elif (k == 0 and v == 0 and P * self.T_s <= self.C and P * self.g_sp2 <= self.I):
      Reward = R_2
    # elif (self.Mu < 1 and k == 1 and v == 1 and P * self.T_s <= self.C and P * self.g_sp1 <= self.I):
    #   Reward = R_1
    # elif (self.Mu < 1 and k == 1 and v == 0 and P * self.T_s <= self.C and P * self.g_sp2 <= self.I):
    #   Reward = R_2
    elif (k == 1 and P * self.T_s > self.C):
      Reward = 0
    else:
      Reward = -self.Phi
    #tinh trang thai hien tai
    self.TimeSlot += 1
    v = 1 if self.TimeSlot % self.N < self.A else 0
    self.get_g()

    Done = False
    if (self.TimeSlot % self.N == 0 and self.TimeSlot != 0):
      self.TimeSlot -= 1
      Done = True

    State = (v , self.E_prev , self.C , self.g_s , self.g_p1r ,self.g_p1s , self.g_p2s , self.g_sp1 , self.g_sp2 , self.g_p1 , self.g_p2)

    return (State , Reward , Done)
if __name__ == '__main__':
  env = Environment()
  State = env.reset()
  print(env.actions_space)
  #print(env.num_action_space())
