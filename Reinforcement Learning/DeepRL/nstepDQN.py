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
class DQN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.network = nn.Sequential(
            nn.Linear(in_dim, (in_dim+out_dim), bias=True),
            nn.ReLU(),
            nn.Linear((in_dim+out_dim), (in_dim+out_dim), bias=True),
            nn.ReLU(),
            nn.Linear((in_dim+out_dim), out_dim, bias=True)
        )
        self.network.apply(self.init)
        # 一个state输出两个action的value
    def init(self, m:nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        
    def forward(self, state_batch:torch.Tensor) -> torch.Tensor:
        return self.network(state_batch)

class DQNTrainer:
    DataType = Tuple[np.ndarray, int, float, np.ndarray]
    def __init__(self, args:Dict):
        for k,v in args.items():
            setattr(self, k, v)
        
        self.epsilon = self.epsilon_lower
        self.buffer = list()
        self.buffer_pointer = 0
        self.network.to(self.device)
        self.episode = []
    
    def add_to_buffer(self, data:DataType):
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(data)
        else:
            self.buffer[self.buffer_pointer] = data
        self.buffer_pointer += 1
        self.buffer_pointer %= self.buffer_size
    
    def sample_batch(self) -> List[DataType]:
        return random.sample(self.buffer, self.batch_size)
    
    def epsilon_greedy(self, state:np.ndarray):
        if random.random() < self.epsilon:
            return random.randint(0, self.env.action_space.n-1)
        return self.greedy(state)

    def greedy(self, state):
        return self.network(torch.tensor(state, device=self.device)).detach().cpu().numpy().argmax()
    
    def update_model(self, total_step):
        if total_step % self.update_freq != 0 or len(self.buffer) < self.batch_size:
            return
        batch = self.sample_batch()
        state_batch, action_batch, reward_batch, new_state_batch = zip(*batch)
        # max 指的是DNN的输出即V(s'),target_value = R+gamma*maxQ(s',a') QLearning

        target_values = self.gamma**self.n * self.network(torch.tensor(np.array(new_state_batch), device=self.device)).detach().max(dim=1).values\
            + torch.tensor(reward_batch, device=self.device)
        # 用discount_reward+gamma**n *Q(s',a')更新V(s)
        # values = Q(s,a)
        # n-step QLearning
        values = (self.network(torch.tensor(np.array(state_batch), device=self.device))
            .gather(dim=1, index=torch.tensor(action_batch, device=self.device).unsqueeze(dim=1))).squeeze()
        self.optimizer.zero_grad()
        loss = F.mse_loss(target_values, values)#loss method
        loss.backward()
        self.optimizer.step()
            
    def epsilon_decay(self, total_step):
        self.epsilon = self.epsilon_lower + (self.epsilon_upper-self.epsilon_lower) * math.exp(-total_step/self.epsilon_decay_freq)

    def train(self):
        total_step = 0
        for i in tqdm.trange(self.start_iter, self.iter):
            state = self.env.reset()[0]
            done = False
            action = self.epsilon_greedy(state)
            self.episode.clear()
            tau = 0
            t = 0
            T = np.inf
            while tau + 1 < T:
                if t < T:
                    new_state, reward, done, truncated, info = self.env.step(action)
                    self.episode.append((state,action,reward))
                    if done:
                        T = t+1
                        reward = self.end_reward
                total_step += 1
                self.epsilon_decay(total_step)
                t += 1
                tau = t-self.n
                discount_reward = 0.0
                if tau >= 0:
                    updating_state, updating_action = self.episode[tau][0:2]
                    n_iter = min(T-tau,self.n)
                    for j in range(n_iter):
                        discount_reward = self.gamma * discount_reward + self.episode[-j][-1]
                    self.add_to_buffer((updating_state, updating_action, discount_reward, new_state))
                    self.update_model(total_step)
                
                state = new_state
                
        self.save_model(i)
    
    def save_model(self, i):
        if i % self.save_freq == 0:
            torch.save(self.network.state_dict(), os.path.join(self.save_path, f'{i}.pkl'))

if __name__ == '__main__':
    env_name = 'CartPole-v1'
    save_path = 'models'
    
    env = gym.make(env_name,render_mode='rgb_array')
    network = DQN(in_dim=env.observation_space.shape[0], out_dim=env.action_space.n)
    latest_checkpoint = 0
#     if save_path not in os.listdir():
#         os.mkdir(save_path)
#     elif len(os.listdir(save_path)) != 0:
#         latest_checkpoint = max([int(file_name.split('.')[0]) for file_name in os.listdir(save_path)])
#         print(f'{latest_checkpoint}.pkl loaded')
#         network.load_state_dict(torch.load(os.path.join(save_path, f'{latest_checkpoint}.pkl')))
        
    trainer = DQNTrainer({
        'env':env,
        'env_name':env_name,
        'render':False,
        'end_reward':-1,
        'network':network,
        'start_iter':latest_checkpoint,
        'iter':latest_checkpoint+1000,
        'gamma':0.8,
        'batch_size':32,
        'buffer_size':1000,
        'update_freq':1,
        'epsilon_lower':0.03,
        'epsilon_upper':0.9,
        'epsilon_decay_freq':200,
        'device':'cpu',
        'optimizer':optim.Adam(network.parameters(),lr=1e-3),
        'save_path':save_path,
        'save_freq':50,
        'n':3
    })
    
    trainer.train()

    N = 10
    rewards=[]
    for i in range(N):
        state = env.reset()[0]
        done = False
        episode_reward = 0
        while not done:
            action = trainer.greedy(state)
            state, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            env.render()
        rewards.append(episode_reward)
    print(np.mean(rewards))
    env.close()

N = 10
rewards=[]
env = gym.make(env_name,render_mode='rgb_array')
for i in range(N):
    state = env.reset()[0]
    done = False
    episode_reward = 0
    while not done:
        action = trainer.greedy(state)
        state, reward, done, truncated, info = env.step(action)
        episode_reward += reward
        env.render()
    rewards.append(episode_reward)
print(np.mean(rewards))

env = gym.make(env_name,render_mode='human')
state = env.reset()[0]
done = False
episode_reward = 0
while not done:
    action = trainer.greedy(state)
    state, reward, done, truncated, info = env.step(action)
    episode_reward += reward
    env.render()
print(episode_reward)
env.close()