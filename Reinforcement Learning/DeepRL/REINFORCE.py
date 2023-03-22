import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import tqdm
import gym
import random
import math
import numpy as np
import os
from typing import Dict, Tuple, List
from matplotlib import pyplot as plt

class Network(nn.Module):
    def  __init__(self, in_dim, out_dim):
        super().__init__()
        self.network=nn.Sequential(
            nn.Linear(in_dim,64),
            nn.ReLU(),
            nn.Linear(64,out_dim),
            nn.Softmax()
        )
    def forward(self,state):
        state = torch.tensor(state)
        return self.network(state)


class PolicyGradient:
    def __init__(self,args:Dict):
        for k,v in args.items():
            setattr(self,k,v)

    def policy(self,state):
        action_dis = self.network.forward(torch.tensor(state)).detach().numpy()
        return np.random.choice(np.arange(self.env.action_space.n),size=1,p=action_dis)[0]

    def update_model(self,buffer_state,buffer_action,buffer_reward):
        T = len(buffer_reward)
        G_list = np.zeros_like(buffer_reward)
        running_r=0

        for t in reversed(range(T)):
            running_r = running_r * self.gamma + buffer_reward[t]
            G_list[t] = running_r

        action_dis = self.network(torch.tensor(buffer_state))
        CE_Loss = torch.sum(-torch.log(action_dis) * F.one_hot(torch.tensor(buffer_action).to(torch.int64),self.env.action_space.n),dim=0)

        loss = torch.sum(CE_Loss * torch.tensor(G_list).reshape(-1,1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().numpy()

    def train(self,iteration):
        loss_history = []
        reward_history = []
        episode_reward = 0
        for i in tqdm.tqdm(range(iteration)):
            buffer_state = []
            buffer_action = []
            buffer_reward = []
            done = False
            state = self.env.reset()[0]
            while not done:
                action = self.policy(state)
                next_state, reward, done, truncated, info = self.env.step(action)
                if done:
                    reward = -1
                if (i+1) % 50 == 0:
                    episode_reward += reward
                buffer_state.append(state)
                buffer_action.append(action)
                buffer_reward.append(reward)
                state = next_state
            loss = self.update_model(np.array(buffer_state),np.array(buffer_action),buffer_reward)
            if (i+1) % 50 == 0:
                loss_history.append((i, loss))
                reward_history.append((i,episode_reward))
                episode_reward = 0
        return loss_history,reward_history



if __name__ == '__main__':
    env = gym.make('CartPole-v1',render_mode='rgb_array')
    network = Network(in_dim=env.observation_space.shape[0],out_dim=env.action_space.n)
    trainer = PolicyGradient({
        'gamma':0.9,
        'env':env,
        'network':network,
        'optimizer':optim.Adam(network.parameters(),lr=1e-3,weight_decay=1e-4)

    })
    loss_history,reward_history = trainer.train(1000)
    plt.figure(figsize=(8,4))
    plt.subplot(121)
    x,y = zip(*loss_history)
    plt.plot(x,y)
    plt.legend(['loss'])
    plt.subplot(122)
    x,y = zip(*reward_history)
    plt.plot(x,y)
    plt.legend(['reward'])
    plt.show()
    
    episode_reward = []
    for i in range(10):
        total_reward=0
        done = False
        state = env.reset()[0]
        while not done:
            action = trainer.policy(state)
            next_state, reward, done, truncated, info = env.step(action)
            total_reward += reward
            state = next_state
        episode_reward.append(total_reward)
    print(np.mean(episode_reward))
        
