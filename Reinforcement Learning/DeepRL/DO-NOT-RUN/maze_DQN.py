import numpy as np
import gym
from typing import Tuple, Dict
import random
import torch
from torch import nn
from tqdm import tqdm

class DQN(nn.Module):
    def __init__(self,n_input,n_hidden,n_output):
        super().__init__()
        self.network = nn.Sequential(
                        nn.Linear(n_input,n_hidden),
                        nn.ReLU(),
                        nn.Linear(n_hidden,n_output),
        )
        self.network.apply(self.init)

    def init(self,m:nn.Linear):
        if isinstance(m,nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias,0)
    
    def forward(self,x):
        return self.network(x)
    
        
class MazeEnv(gym.Env):
    WALL = 1
    EMPTY = 0
    ACTIONS = {'x':[-1,1,0,0], 'y':[0,0,1,-1]}
    StateType = Tuple[int, int]
    
    def __init__(self, config:Dict):
        for k, v in config.items():
            setattr(self, k, v)
        
        self.max_y, self.max_x = self.maze.shape
        assert(self.maze[self.start_y, self.start_x] == MazeEnv.EMPTY)
        assert(self.maze[self.target_y, self.target_x] == MazeEnv.EMPTY)
        
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.MultiDiscrete([self.max_x, self.max_y])
    
    def is_valid_state(self, x, y):
        return  0 <= x < self.max_x and 0 <= y < self.max_y and self.maze[y, x] == MazeEnv.EMPTY
    
    def step(self, action:int) -> Tuple[StateType, float, bool, None]:
        new_x, new_y = self.x + MazeEnv.ACTIONS['x'][action], self.y + MazeEnv.ACTIONS['y'][action]
        
        if not self.is_valid_state(new_x, new_y):
            new_x, new_y = self.x, self.y
        self.x, self.y = new_x, new_y
        
        reach_target = ((self.x, self.y) == (self.target_x, self.target_y))
        reward = 1.0 if reach_target else 0.0
        
        return (self.x, self.y), reward, reach_target, None
    
    def reset(self):
        self.x = self.start_x
        self.y = self.start_y
        return (self.x, self.y)
    
    def render(self):
        for i in range(self.max_y):
            for j in range(self.max_x):
                if (j, i) == (self.x, self.y):
                    print('*', end='')
                elif (j, i) == (self.target_x, self.target_y):
                    print('G', end='')
                else:
                    print(self.maze[i, j], end='')
            print()
        print()
    
    def close(self):
        pass

def print_maze_policy(maze:np.ndarray, policy):
    max_y, max_x = maze.shape
    for i in range(max_y):
        for j in range(max_x):
            if maze[i, j] == MazeEnv.EMPTY:
                print('<>v^'[policy((j, i))], end='')
            else:
                print(maze[i, j], end='')
        print()
    print()

def random_maze_policy(state:MazeEnv.StateType):
    return random.randint(0, 3)

class DQNTrainer:
    def __init__(self,args:Dict):
        for k,v in args.items():
            setattr(self,k,v)
        self.buffer = []
        self.buffer_point = 0
        self.epsilon = self.epsilon_upper
        self.network.to(self.device)
        
    def epsilon_decay(self,total_step):
        self.epsilon = self.epsilon_lower + (self.epsilon_upper - self.epsilon_lower) * np.exp( - total_step / self.decay_constant) 

    def add_to_buffer(self,data):
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(data)
        else:
            self.buffer[self.buffer_point] = data
        self.buffer_point += 1
        self.buffer_point %= self.buffer_size

    
    def buffer_sample(self):
        return random.sample(self.buffer,self.batch_size)
    
    def update_model(self):
        if len(self.buffer) < self.batch_size:
            return
        batch = self.buffer_sample()

        state_batch, action_batch, reward_batch, new_state_batch = zip(*batch)
        target_values = torch.tensor(reward_batch, device=self.device)\
                + self.gamma * self.network(torch.tensor(new_state_batch,device=self.device).to(torch.float32)).detach().max(axis=1).values
        
        values = self.network(torch.tensor(state_batch,device=self.device).to(torch.float32))\
                .gather(dim=1,index=torch.tensor(action_batch,device=self.device).unsqueeze(dim=1)).squeeze()

        loss = self.loss_fn(target_values,values)
        self.optimizer.zero_grad()
#         print(loss)
        loss.backward()
        self.optimizer.step()
    
    def greedy(self,state):
        return self.network(torch.tensor(state,device=self.device).to(torch.float32)).detach().cpu().numpy().argmax()

    def epsilon_greedy(self,state):
        if random.random() < self.epsilon:
            return np.random.randint(0,self.env.action_space.n)
        return self.greedy(state)

    def train(self):
        total_step = 0
        for i in tqdm(np.arange(self.iteration_step)):
            state = env.reset()
            done = False
            while not done:
                action = self.epsilon_greedy(state)
                new_state, reward, done, info = self.env.step(action)
#                 print(action)
#                 print(state)
                self.epsilon_decay(total_step)
                self.add_to_buffer((state,action,reward,new_state))
                if i % self.update_interval == 0:
                    self.update_model()
                
                state = new_state
                total_step += 1

if __name__ == '__main__':
    maze = np.array([
        [0,0,0,0,0,0,0,1,0],
        [0,0,1,0,0,0,0,1,0],
        [0,0,1,0,0,0,0,1,0],
        [0,0,1,0,0,0,0,0,0],
        [0,0,0,0,0,1,0,0,0],
        [0,0,0,0,0,0,0,0,0]
    ])
    
    env = MazeEnv({
        'maze':maze,
        'start_x':0, 'start_y':2,
        'target_x':8, 'target_y':0
    })
    network = DQN(n_input=len(maze.shape),n_hidden=10,n_output=env.action_space.n)
    trainer = DQNTrainer({
        'env':env,
        'buffer_size':1000,
        'batch_size':32,
        'iteration_step':1000,
        'epsilon_upper':0.5,
        'epsilon_lower':0.03,
        'decay_constant':200,
        'update_interval':5,
        'gamma':0.95,
        'network':network,
        'optimizer':torch.optim.Adam(network.parameters(),lr=1e-3),
        'loss_fn':nn.MSELoss(),
        'device':'cuda:0'
        
    })
    state = env.reset()
    trainer.train()
