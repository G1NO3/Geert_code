import torch
import torch.nn as nn
import torch.optim as optim
###
import torch.nn.functional as F
###尽量避免使用F的任何函数，可能无用，因为他是外部的
import tqdm
import gym
import numpy as np
from typing import Dict
from matplotlib import pyplot as plt

class PolicyNetwork(nn.Module):
    def  __init__(self, in_dim, out_dim):
        super().__init__()
        self.network=nn.Sequential(
            nn.Linear(in_dim,20),
            nn.ReLU(),
            nn.Linear(20,out_dim),
            nn.Softmax()
        )
    def forward(self,state):
        state = torch.tensor(state)
        return self.network(state)

class ValueNetwork(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_dim, 20),
            nn.ReLU(),
            nn.Linear(20, out_dim),
        )
    def forward(self,state):
        state = torch.tensor(state)
        return self.network(state)

class PolicyGradient:
    def __init__(self,args:Dict):
        for k,v in args.items():
            setattr(self,k,v)
        self.buffer = []
        self.buffer_point = 0
        self.buffer_pre_length = 0

    def policy_selection(self,state):
        action_dis = self.policy_network.forward(torch.tensor(state)).detach().numpy()
        return np.random.choice(np.arange(self.env.action_space.n),size=1,p=action_dis)[0]

    def update_policy(self, state, action, td_error):

############Important
        action_dis = self.policy_network(torch.tensor(state))
        gradient = torch.sum(torch.log(action_dis) * F.one_hot(torch.tensor(action).to(torch.int64),self.env.action_space.n),dim=0)
############

        policy_loss = torch.sum( -td_error * gradient)

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()


        return policy_loss.detach().numpy()

    def update_value(self, state, reward, next_state):
        next_value = torch.tensor(reward) + self.gamma * self.value_network(next_state)
        value = self.value_network(state)
        value_loss = self.vl_fn(next_value, value)
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()


        with torch.no_grad():
            td_error = reward + self.gamma * self.value_network(next_state) - self.value_network(state)

        return td_error


    def train(self,iteration):
        policy_loss_history = []
        reward_history = []
        value_loss_history = []
        episode_reward = 0
        for i in tqdm.tqdm(range(iteration)):
            done = False
            state = self.env.reset()[0]
            while not done:
                action = self.policy_selection(state)
                next_state, reward, done, truncated, info = self.env.step(action)
                if done:
                    reward = -5
                if (i+1) % 50 == 0:
                    episode_reward += reward
                # self.add_to_buffer((state,action,reward,next_state))
                # td_error = self.update_value(state, reward, next_state)
                td_error = self.update_value(state, reward, next_state)
                policy_loss = self.update_policy(state, action, td_error)

                state = next_state

            if (i+1) % 50 == 0:
                policy_loss_history.append((i, policy_loss))
                value_loss_history.append((i, td_error))
                reward_history.append((i,episode_reward))

                episode_reward = 0
        return policy_loss_history, value_loss_history, reward_history



if __name__ == '__main__':
    env = gym.make('CartPole-v1',render_mode='rgb_array')
    policy_network = PolicyNetwork(in_dim=env.observation_space.shape[0],out_dim=env.action_space.n)
    value_network = ValueNetwork(in_dim=env.observation_space.shape[0],out_dim=1)
    trainer = PolicyGradient({
        'gamma':0.92,
        'env':env,
        'policy_network':policy_network,
        'policy_optimizer':optim.Adam(policy_network.parameters(),lr=1e-3),
        'value_network':value_network,
        'value_optimizer':optim.Adam(value_network.parameters(),lr=1e-3),
        'vl_fn':nn.MSELoss(),
        'batch_size':16,
        'buffer_size':10000
    })
    policy_loss_history, value_loss_history, reward_history = trainer.train(500)
    plt.figure(figsize=(12,4))
    plt.subplot(131)
    x,y = zip(*policy_loss_history)
    plt.plot(x,y)
    plt.title('Policyloss')

    plt.subplot(132)
    x,y = zip(*value_loss_history)
    plt.plot(x,y)
    plt.title('Valueloss')

    plt.subplot(133)
    x,y = zip(*reward_history)
    plt.plot(x,y)
    plt.title('Reward')
    plt.show()
    
    episode_reward = []
    for i in range(10):
        total_reward=0
        done = False
        state = env.reset()[0]
        while not done:
            action = trainer.policy_selection(state)
            next_state, reward, done, truncated, info = env.step(action)
            total_reward += reward
            state = next_state
        episode_reward.append(total_reward)
    print(np.mean(episode_reward))
        
