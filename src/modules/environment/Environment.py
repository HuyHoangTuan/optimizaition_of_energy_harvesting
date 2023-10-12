import numpy as np

class Environment:
    def __init__(
            self,
            P_max=1, Xi=0.1, Lambda=0.1, N=20, I=0.5, Gamma=0.99,
            Alpha=0.003, Eta=0.9, T_s=1, N_0=1,
            E_max=0.2,
            Phi=1, Rho=0.4, A=10
    ):
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
