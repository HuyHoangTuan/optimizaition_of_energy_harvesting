import numpy as np
import matplotlib.pyplot as plt

Episodes = 1400 + 1
N = 20
const_C_max = 0.5
const_P_max = 1
const_A = 10
const_Xi = 0.1
const_E_max = 0.2
rng = np.random.default_rng()

#----------------------------------------------------
def g_generator(Xi = const_Xi):
  g_s = rng.exponential(1 / Xi , N * Episodes)
  g_p1r = rng.exponential(1 / Xi , N * Episodes)
  g_p1s = rng.exponential(1 / Xi , N * Episodes)
  g_p2r = rng.exponential(1 / Xi , N * Episodes)
  g_p2s = rng.exponential(1 / Xi , N * Episodes)
  g_sp1 = rng.exponential(1 / Xi , N * Episodes)
  g_sp2 = rng.exponential(1 / Xi , N * Episodes)
  g_p1 = rng.exponential(1 / Xi , N * Episodes)
  g_p2 = rng.exponential(1 / Xi , N * Episodes)

  np.savetxt('g_s.txt' , g_s ,delimiter = ' ')
  np.savetxt('g_p1r.txt' , g_p1r ,delimiter = ' ')
  np.savetxt('g_p1s.txt' , g_p1s ,delimiter = ' ')
  np.savetxt('g_p2r.txt', g_p2r, delimiter=' ')
  np.savetxt('g_p2s.txt' , g_p2s ,delimiter = ' ')
  np.savetxt('g_sp1.txt' , g_sp1 ,delimiter = ' ')
  np.savetxt('g_sp2.txt' , g_sp2 ,delimiter = ' ')
  np.savetxt('g_p1.txt' , g_p1 ,delimiter = ' ')
  np.savetxt('g_p2.txt' , g_p2 ,delimiter = ' ')

def P_and_C_and_E_ambient_generator(P_max = const_P_max , C_max = const_C_max , A = const_A , E_max = const_E_max):
  P_p1 = rng.uniform(0 , P_max , N * Episodes)
  P_p2 = rng.uniform(0 , P_max , N * Episodes)
  E_ambient = rng.uniform(0 , E_max , N * Episodes)
  C_0 = rng.uniform(0 , C_max , Episodes)

  # idx = np.array([s for s in range(N * Episodes)])
  #
  # idx_zero_p1 = np.where(idx % N >= A)
  # idx_zero_p2 = np.where(idx % N < A)
  #
  # P_p1[idx_zero_p1] = 0
  # P_p2[idx_zero_p2] = 0

  np.savetxt('P_p1.txt' , P_p1 , delimiter = ' ')
  np.savetxt('P_p2.txt' , P_p2 , delimiter = ' ')
  np.savetxt('E_ambient.txt' , E_ambient , delimiter = ' ')
  np.savetxt('C_0.txt' , C_0 , delimiter = ' ')


#-------------------------------
P_and_C_and_E_ambient_generator()
g_generator()

# data = np.loadtxt('E_ambient.txt' , delimiter = ' ')
# print(data)
# print(data.mean(0))
#
# plt.hist(data)
# plt.show()
