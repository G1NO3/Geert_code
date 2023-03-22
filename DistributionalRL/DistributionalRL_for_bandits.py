import numpy as np
from typing import Dict
import matplotlib.pyplot as plt

class RewardEnv:
    def __init__(self,mu_list,sigma_list):
        self.mu_list = mu_list
        self.sigma_list = sigma_list
    
    def step(self,action):
        mu = self.mu_list[action]
        sigma = self.sigma_list[action]
        return np.random.randn() * (sigma**2) + mu

class SingleNeuron:
    def __init__(self,alpha_plus,alpha_minus,n_action=8):
        self.alpha_plus = alpha_plus
        self.alpha_minus = alpha_minus
        self.value = np.zeros(n_action)
        self.quantile = self.alpha_plus / (self.alpha_plus + self.alpha_minus)
    def update(self,action,reward):
        RPE = reward - self.value[action]
        if RPE > 0:
            self.value[action] += self.alpha_plus * RPE
        elif RPE < 0:
            self.value[action] += self.alpha_minus * RPE

class NeuronGroupAgent:
    def __init__(self,n_neurons,n_action,env,lr):
        alpha_plus_list = np.ones(n_neurons) * lr
        quantiles = np.linspace(0.007,0.999,n_neurons) 
        alpha_minus_list = (1/quantiles - 1) * alpha_plus_list
        self.neurons = []
        for i in range(n_neurons):
            self.neurons.append(SingleNeuron(alpha_plus_list[i], alpha_minus_list[i],n_action))
        self.n_neurons = n_neurons
        self.n_action = n_action
        self.env = env

    def update_all(self,action,reward):
        for i in range(self.n_neurons):
            self.neurons[i].update(action,reward)

    def train(self,iterations):
        for i in range(iterations):
            action = np.random.randint(self.n_action)
            reward = self.env.step(action)
            self.update_all(action,reward)
    
    def reconstruct(self):
        for bandit in range(self.n_action):
            quantiles=[]
            expectations=[]
            for neuron in self.neurons:
                quantiles.append(neuron.quantile)
                expectations.append(neuron.value[bandit])
            probability = np.diff(quantiles)/np.diff(expectations)
            plt.plot(expectations[:-1],probability)
        plt.legend(np.arange(self.n_action))
        plt.title('Reconstructed')

plt.subplot(211)
x = np.linspace(-2,10,100)
y1 = 1/np.sqrt(2*np.pi)*np.exp(-(x-0)**2/2/(1**2))
y2 = 1/np.sqrt(2*np.pi*1.5**2)*np.exp(-(x-3)**2/2/(1.5**2))
y3 = 1/np.sqrt(2*np.pi*2**2)*np.exp(-(x-6)**2/2/(2**2))
plt.plot(x,y1)
plt.plot(x,y2)
plt.plot(x,y3)
plt.legend(['mu=0,sigma=1','mu=3,sigma=1.5','mu=6,sigma=2'])
plt.title('Original')
env = RewardEnv(
    mu_list = [0,3,6],
    sigma_list = [1,1.5,2]
)
Agent = NeuronGroupAgent(
    n_neurons = 20,
    n_action = 3,
    env = env,
    lr = 1e-2
)
plt.subplot(212)
Agent.train(7000)
Agent.reconstruct()
plt.show()