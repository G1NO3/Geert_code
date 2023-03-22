import numpy as np
import seaborn as sns
from typing import Dict
import matplotlib.pyplot as plt


class Network():
    def __init__(self):
        N = 3000
        P = 4
        self.N = N
        self.P = P
        N_G = 400
        N_I = 800
        self.N_I = N_I
        self.N_G = N_G
        
        self.dt = 0.01

        self.tau_E = 1
        self.tau_I = 5
        self.tau_G = 5

        # self.D_th = 100
        # self.T_th = 2
        # self.th_0 = np.zeros(N)
        # self.th = np.zeros(N)

        X = np.zeros((P,N))
        self.X = X
        for i in range(4):
            if (N//4)*(i+1)+100 < N: 
                X[i, (N//4)*i:(N//4)*(i+1)+100] = np.ones(N//4+100)
            else:
                X[i, (N//4)*i:(N//4)*(i+1)] = np.ones(N//4)

        # Initialize W matrix
        self.W_EE = self.init_WEE(N,X)
        self.W_IE = self.init_WIE(N,X)
        self.W_EI = self.W_IE.T
        self.W_EG = np.ones((N,N_G))
        self.W_IG = np.ones((N_I,N_G))
        self.W_GE = self.W_EG.T
        self.W_GI = self.W_IG.T


        self.U_E = np.zeros(N)
        self.U_I = np.zeros(N_I)
        self.r_E = np.zeros(N)
        self.r_I = np.zeros(N_I)

        self.U_G = np.zeros(N_G)
        self.r_G = np.zeros(N_G)

        self.m_history = np.zeros(4)

    def make_tree(self,n):
        root = 0
        return root


    def init_WEE(self,N,X,hetero_learn=False):

        W = np.matmul(X.T,X)
        if hetero_learn:
            H = np.ones((N,N)) * 0.001
            H[0:N//2,0:N//2] += 0.0005
            H[N//2:N,N//2:N] += 0.005
            for i in range(4):
                H[(N//4)*i:(N//4)*(i+1),(N//4)*i:(N//4)*(i+1)] += 0.05
            W += H
        return W
#################
    def init_WIE(self,N,X):
        N_I = self.N_I
        W = np.zeros((N_I,N))
        W[0:N_I//2, 0:N//2+100] = np.ones((N_I//2, N//2+100))
        W[N_I//2:N_I, N//2:N] = np.ones((N_I//2, N//2))
        return W
###################  
    # calculate global_inhibition
    @property
    def glob_inh(self):
        return self.kappa*(np.power(np.sum(self.r_E),2)+np.power(np.sum(self.r_I),2))

    @property
    def intensity_I(self):
        return np.sum(self.r_I)/self.N_I
    
    @property
    def intensity_G(self):
        return np.sum(self.r_G)/self.N_G


    def update_threshold(self):
        delta_th = -(self.th - self.th_0 - self.r_E*self.D_th) / self.T_th
        self.th += delta_th

    def activation(self,x):
        return 1/(1+np.exp(-x))
        # return 2/np.pi * np.arctan(8 * np.pi * x)

    def update(self):
        # delta_E = (-self.U_E + np.matmul(self.r_E, self.W_EE) - np.matmul(self.r_I, self.W_IE) - self.glob_inh + self.current - self.th)/self.tau_E

        delta_E = (-self.U_E + np.matmul(self.r_E, self.W_EE) - np.matmul(self.r_I, self.W_IE) - np.matmul(self.r_G, self.W_GE) + self.current) / self.tau_E
        delta_I = (-self.U_I + np.matmul(self.r_E, self.W_EI) - np.matmul(self.r_G, self.W_GI)) / self.tau_I
        delta_G = (-self.U_G + np.matmul(self.r_E, self.W_EG)) / self.tau_G

        self.U_E += delta_E * self.dt
        self.U_I += delta_I * self.dt
        self.U_G += delta_G * self.dt

        self.r_E = self.activation(self.U_E)
        self.r_I = self.activation(self.U_I)
        self.r_G = self.activation(self.U_G)

        # self.update_threshold()

    # calculate m
    def update_m(self):
        m = np.zeros(4)
        for i in range(4):
            m[i] = np.clip(np.sum(self.r_E[self.N//4*i:self.N//4*(i+1)])/(self.N//4),0,1)
        self.m_history = np.vstack((self.m_history, m))

    def run(self,iterations,verbose=False):
        self.I_history = []
        self.G_history = []
        for i in range(iterations):
            # exert a current
            if (i+1) < 50:
                self.current = 40*self.X[0] + 10 * self.X[1] + 10 * self.X[2] + 10 * self.X[3]
            else:
                self.current = 0
            if i % 1 == 0:
                if verbose:
                    print('-----------------episode:---------------------------',i)
                    print('E2E:',np.matmul(self.r_E, self.W_EE))
                    print('I2E',np.matmul(self.r_I, self.W_IE))
                    print('G2E',np.matmul(self.r_G, self.W_GE))
                    print('glob_inh',self.glob_inh)
                    print('r_E:',self.r_E)
                    print('UE:',self.U_E)
                    print('r_I:',self.r_I)
                    print('UI:',self.U_I)
                    print('r_G:',self.r_G)
                    print('UG:',self.U_G) 
            self.update()
            self.update_m()
            self.I_history.append(self.intensity_I)
            self.G_history.append(self.intensity_G)
        return self.m_history.T
    
if __name__ == '__main__':
    np.set_printoptions(threshold=5)
    model = Network()
    history = model.run(30)
    plt.figure(figsize=(12,4))
    plt.subplot(131)
    sns.heatmap(history)
    plt.subplot(132)
    plt.plot(np.arange(len(model.I_history)),model.I_history)
    plt.title('intensity of local_inhibitory neurons')
    plt.subplot(133)
    plt.plot(np.arange(len(model.G_history)),model.G_history)
    plt.title('intensity of global_inhibitory neurons')
    plt.show()




