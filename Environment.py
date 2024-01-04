import numpy as np
import random
import math
class Environment:
  def __init__(self, P_max = 1 , Xi = 0.1 , Lambda = 0.1 , N = 20 , I = 0.5 , Gamma = 0.99 , \
               Alpha = 0.003 , Eta = 0.9 , T_s = 1 , N_0 = 1 \
               , E_max = 0.2 ,\
               Phi = 1 , Rho = 0.4 , A = 10):
    #self.actions_space = [(0 , s * 0.05) for s in range(1 , 11 , 1)] + [(1,s * 0.05) for s in range(1 , 11 , 1)]
    k_value = [0]
    Rho_value = [s * 0.1 for s in range(0,10 , 1)]
    P_value = [s * 0.01 for s in range(1,53 , 1)]
    actions_space = np.array(np.meshgrid(k_value ,Rho_value , P_value)).T.reshape(-1 , 3)
    actions_space = np.append(actions_space , np.array([[1 , 1.0 , 0]]) , axis = 0)
    self.actions_space = [tuple(x) for x in actions_space]
    #self.actions_space = [(0, s * 0.01) for s in range(1, 53, 1)] + [(1, s * 0.01) for s in range(1, 53, 1)]
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
    self.arr_g_p2r = np.loadtxt('g_p2r.txt', delimiter=' ')
    self.arr_g_p2s = np.loadtxt('g_p2s.txt' , delimiter = ' ')
    self.arr_g_sp1 = np.loadtxt('g_sp1.txt' , delimiter = ' ')
    self.arr_g_sp2 = np.loadtxt('g_sp2.txt' , delimiter = ' ')
    self.arr_g_p1 = np.loadtxt('g_p1.txt' , delimiter = ' ')
    self.arr_g_p2 = np.loadtxt('g_p2.txt' , delimiter = ' ')

  def harvest_energy(self , Rho):
    E_ambient = self.arr_E_ambient[self.TimeSlot]

    P_p1 = self.arr_P_p1[self.TimeSlot]
    P_p2 = self.arr_P_p2[self.TimeSlot]

    # if(P_p1 < self.Lambda and P_p2 < self.Lambda):
    #   Rho = 0

    E_TS = Rho * self.T_s * self.Eta * P_p1 * self.g_p1s
    E_TS += Rho * self.T_s * self.Eta * P_p2 * self.g_p2s

    self.Mu = 1 - Rho
    E_h = E_TS + E_ambient

    return E_h

  def map_to_exponential_function(self , x):
    # f(min) = 1/20 va f(max) = 1
    max_val = self.I + self.I + self.C_max

    b = 1/35
    a = max_val / math.log(1 / b)
    return b * math.exp(x / a)

  def num_action_space(self):
    return len(self.actions_space)

  def num_state_space(self):
    #return 9
    return 11
  def get_g(self):
    self.g_s = self.arr_g_s[self.TimeSlot]
    self.g_p1r = self.arr_g_p1r[self.TimeSlot]
    self.g_p2r = self.arr_g_p2r[self.TimeSlot]
    self.g_p1s = self.arr_g_p1s[self.TimeSlot]
    self.g_p2s = self.arr_g_p2s[self.TimeSlot]
    self.g_sp1 = self.arr_g_sp1[self.TimeSlot]
    self.g_sp2 = self.arr_g_sp2[self.TimeSlot]
    self.g_p1 = self.arr_g_p1[self.TimeSlot]
    self.g_p2 = self.arr_g_p2[self.TimeSlot]

  def reset(self):
    self.TimeSlot += 1
    # v = 1 cho pu1 v = 0 cho pu2
    self.E_prev = 0
    self.C = self.arr_C_0[int(self.TimeSlot / self.N)]
    self.get_g()

    P_p1 = self.arr_P_p1[self.TimeSlot]
    P_p2 = self.arr_P_p2[self.TimeSlot]
    #return (self.E_prev , self.C, 10 , 10 , 10 , 10 ,10 ,10 ,10)
    #return (self.E_prev , self.C, P_p1 , P_p2)
    return (self.E_prev , self.C,P_p1 , P_p2 ,  10 , 10 , 10 , 10 ,10 ,10 ,10)

  def action_sampling(self):
    return random.sample([s for s in range(len(self.actions_space))], 1)[0]

  def map_action(self,num):
    return self.actions_space[num]
  def step(self, action):
    k,Rho , P = self.map_action(action)
    #tinh nang luong thu duoc buoc truoc va nang luong hien tai
    E_h = self.harvest_energy(Rho)
    self.E_prev = E_h

    self.C = min(self.C + E_h - (1 - k) * self.Mu * P * self.T_s, self.C_max)
    self.C = max(self.C , 0)
    Rate = 0
    Reward = 0
    Foul = []
    Foul_cnt = 0

    # tinh phan thuong tu hanh dong hien tai
    P_p1 = self.arr_P_p1[self.TimeSlot]
    P_p2 = self.arr_P_p2[self.TimeSlot]
    if k == 0:
      P_dbm = 10 * math.log10(P) + 30

      P_p1_dbm = 10 * math.log10(P_p1) + 30
      P_p2_dbm = 10 * math.log10(P_p2) + 30

      P_10 = 10 ** (P_dbm / 10)
      P_p1_10 = 10 ** (P_p1_dbm / 10)
      P_p2_10 = 10 ** (P_p2_dbm / 10)


      Infer = (1 - Rho) * self.T_s * (P_p1_10 * self.g_p1r + P_p2_10 * self.g_p2r)
      R = self.Mu * self.T_s * math.log2(1 + P_10 * self.g_s / (self.N_0 + Infer))
      if (k == 0  and self.Mu * P * self.T_s <= self.C and P * self.g_sp1 <= self.I and P * self.g_sp2 <= self.I):
        d = (self.C - self.Mu * P * self.T_s) + (self.I - P * self.g_sp1) + (self.I - P * self.g_sp2)
        d = self.map_to_exponential_function(d)
        Rate = R
        Reward = R / d
      elif (k == 1):
        Reward = 0
      else:
        if(self.Mu * P * self.T_s > self.C):
          Foul.append(1)
          Foul_cnt += 1
          Reward += - self.Mu * self.T_s * math.log2(1 + (self.Mu * P * self.T_s - self.C) * self.g_s / (self.N_0 ))
        if(P * self.g_sp1 > self.I):
          Foul.append(2)
          Foul_cnt += 1
          Reward += - self.Mu * self.T_s * math.log2(1 + (P * self.g_sp1 - self.I) / (self.N_0 ))
          #Reward += - (P * self.g_sp1 - self.I)
        if(P * self.g_sp2 > self.I):
          Foul.append(3)
          Foul_cnt += 1
          Reward += - self.Mu * self.T_s * math.log2(1 + (P * self.g_sp2 - self.I) / (self.N_0 ))
          #Reward += - (P * self.g_sp2 - self.I)

    if not Foul:
      Foul.append(0)

    #time slot tiep theo
    self.TimeSlot += 1
    Done = False
    if (self.TimeSlot % self.N == 0 and self.TimeSlot != 0):
      self.TimeSlot -= 1
      Done = True

    #--- P1 va P2 cua timeslot tiep theo
    P_p1 = self.arr_P_p1[self.TimeSlot]
    P_p2 = self.arr_P_p2[self.TimeSlot]


    #State = (self.E_prev , self.C , self.g_s , self.g_sp1 , self.g_sp2 , self.g_p1s , self.g_p2s , self.g_p1r , self.g_p2r , Foul)
    #State = (self.E_prev , self.C , P_p1 , P_p2 , Foul)
    State = (self.E_prev, self.C,P_p1 , P_p2 , self.g_s, self.g_sp1, self.g_sp2, self.g_p1s, self.g_p2s, self.g_p1r, self.g_p2r, Foul ,Foul_cnt , Rate)
    self.get_g()

    return (State , Reward , Done)
if __name__ == '__main__':
  env = Environment()
  State = env.reset()
  print(env.num_state_space())
  #print(env.actions_space)
  #print(env.num_action_space())
