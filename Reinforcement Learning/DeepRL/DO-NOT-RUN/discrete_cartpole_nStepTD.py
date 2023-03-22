import gym
import numpy as np
import tqdm
import random
import math
from typing import Tuple, Dict, Any
import os
import matplotlib.pyplot as plt
class DiscreteCartPoleEnv(gym.Env):
    def __init__(self, intervals=16, rm='rgb_array'):
        self._env = gym.make('CartPole-v1',render_mode=rm)
        self.action_space = self._env.action_space
        self.intervals = intervals
        # Set the range of every dimension to [0,15]
        self.observation_space = gym.spaces.MultiDiscrete([intervals]*4)
        # Project the x to observation_space [0,15]
        self._to_discrete = lambda x, a, b: int(min(max(0, (x-a)*self.intervals/(b-a)), self.intervals))
        
    def render(self):
        self._env.render()
    
    def reset(self):
        return self._discretize(self._env.reset()[0])

    # Discretize the continuous variables
    def _discretize(self, state:np.array)->Tuple:
        cart_pos, cart_v, pole_angle, pole_v = state
        cart_pos = self._to_discrete(cart_pos, -2.4, 2.4)
        cart_v = self._to_discrete(cart_v, -3.0, 3.0)
        pole_angle = self._to_discrete(pole_angle, -0.5, 0.5)
        pole_v = self._to_discrete(pole_v, -2.0, 2.0)
        return (cart_pos, cart_v, pole_angle, pole_v)
    
    def step(self, action:int)->Tuple[Tuple, float, bool, Any]:
        state, reward, done, truncated, info = self._env.step(action)
        state = self._discretize(state)
        return state, reward, done, info

class QLearner:
    def __init__(self, config:Dict):
        for k, v in config.items():
            setattr(self, k, v)
        self.epsilon = self.epsilon_lower# possibly be defined above
        self.lr = self.lr_upper
        self.buffer = list()
        self.buffer_pointer = 0

    
    def add_to_buffer(self, data):
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(data)
        else:
            self.buffer[self.buffer_pointer] = data
        self.buffer_pointer += 1
        self.buffer_pointer %= self.buffer_size
    
    def sample_batch(self):
        return random.sample(self.buffer, self.batch_size)
    
    def greedy(self, state:Tuple)->int:
        return self.q[state].argmax()

    def epsilon_greedy(self, state:Tuple)->int:
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        return self.greedy(state)
    
    def epsilon_decay(self, total_step):
        self.epsilon = self.epsilon_lower + (self.epsilon_upper - self.epsilon_lower) * math.exp(-total_step / self.epsilon_decay_freq)
    
    def lr_decay(self, total_step):
        self.lr = self.lr_lower + (self.lr_upper - self.lr_lower) * math.exp(-total_step / self.lr_decay_freq)
    
    def update_q(self, total_step):
        if total_step % self.update_freq != 0 or len(self.buffer) < self.batch_size:
            return
        batch = self.sample_batch()# Randomly select the Q(s,a) batches to update
        for state, action, reward, new_state in batch:
            self.q[state][action] += self.lr * (self.gamma * self.q[new_state].max() + reward - self.q[state][action])
    
    def train(self):
        total_step = 0
        for i in tqdm.trange(self.start_iter, self.iter):
            state = self.env.reset()
            done = False
            while not done:
                total_step += 1
                action = self.epsilon_greedy(state)
                self.epsilon_decay(total_step)
                new_state, reward, done, _ = self.env.step(action)
                if self.render:# as a verbose button
                    self.env.render()
                if done:
                    reward = self.end_reward
                self.add_to_buffer((state, action, reward, new_state))
                self.update_q(total_step)
                self.lr_decay(total_step)
                self.save_model(i)
                state = new_state
    
    def save_model(self, i):
        if i % self.save_freq == 0:
            np.save(os.path.join(self.save_path, f'{i}.npy'), self.q)
        
class nStepTD:
    def __init__(self, config:Dict):
        for k, v in config.items():
            setattr(self, k, v)
        self._q = np.zeros(shape=(intervals+1,)*env.observation_space.shape[0]+(env.action_space.n,))
        self.episode = []
        if self.decay:
            self.epsilon = self.epsilon_lower
            self.alpha = self.alpha_upper
        else:
            self.epsilon = self.epsilon_lower
            self.alpha = self.alpha_lower
    
    def epsilon_greedy(self,state):
        if random.random() < self.epsilon:
            return env.action_space.sample()
        return self.greedy(state)
    
    def epsilon_decay(self, total_step):
        self.epsilon = self.epsilon_lower + (self.epsilon_upper - self.epsilon_lower) * math.exp(-total_step / self.epsilon_decay_freq)
    
    def alpha_decay(self, total_step):
        self.lr = self.alpha_lower + (self.alpha_upper - self.alpha_lower) * math.exp(-total_step / self.alpha_decay_freq)

    def greedy(self,state:Tuple)->int :
        return self._q[state].argmax()

    def train(self):
        total_step = 0
        for i in tqdm.tqdm(range(self.iteration)):
            self.episode.clear()
            state = self.env.reset()
            tau = 0
            t = 0
            T = np.inf
            while tau + 1 < T :
                if t < T:
                    action = self.epsilon_greedy(state)
                    new_state, reward, done, info = self.env.step(action)
                    self.episode.append((state, action, reward))
                    if done:
                        T = t+1
                    if self.decay:
                        self.epsilon_decay(total_step)
                        self.alpha_decay(total_step)
                        total_step += 1
                t += 1
                tau = t - self.n
                discount_reward = 0.0
                state = new_state
                if tau >= 0:
                    updating_state, updating_action = self.episode[tau][0:2]
                    n_iter = min(T-tau, self.n)
                    for j in range(n_iter):
                        discount_reward = self.gamma * discount_reward + self.episode[-j][-1]
                    if t < T:
                        max_action = self.greedy(new_state)
                        discount_reward += self._q[new_state][max_action]*(self.gamma**self.n)
                    self._q[updating_state][updating_action] += self.alpha * (discount_reward - self._q[updating_state][updating_action])





             
if __name__ == '__main__':
    env_name = 'DiscreteCartPole'
    save_path = 'q_tables'
    intervals = 16
    # env = DiscreteCartPoleEnv(intervals)
    # produce a high_dimensional table the shape of which is [9,9,9,9,2]
    # q_table = np.zeros(shape=(intervals+1,)*env.observation_space.shape[0]+(env.action_space.n,))
    
    # latest_checkpoint = 0
    
    # if save_path not in os.listdir():
    #     os.mkdir(save_path)
    # elif len(os.listdir(save_path)) != 0:
    #     latest_checkpoint = max([int(file_name.split('.')[0]) for file_name in os.listdir(save_path)])
    #     print(f'{latest_checkpoint}.npy loaded')
    #     q_table = np.load(os.path.join(save_path, f'{latest_checkpoint}.npy'))# loaded the latest q_table
            
    # trainer = QLearner({
    #     'env':env,
    #     'env_name':env_name,
    #     'render':False,
    #     'end_reward':-1,
    #     'q':q_table,
    #     'start_iter':latest_checkpoint,
    #     'iter':latest_checkpoint+1000,
    #     'batch_size':128,
    #     'buffer_size':10000,
    #     'gamma':0.9,
    #     'update_freq':1,
    #     'epsilon_lower':0.05,
    #     'epsilon_upper':0.8,
    #     'epsilon_decay_freq':200,
    #     'lr_lower':0.05,
    #     'lr_upper':0.5,
    #     'lr_decay_freq':200,
    #     'save_path':save_path,
    #     'save_freq':50
    # })



    N=8
    Iteration_steps = [1000, 5000, 10000]
    Intervals = [8, 12, 16]
    for interval in Intervals:
        env = DiscreteCartPoleEnv(interval)
        for it in Iteration_steps:
            state = env.reset()
            rewards=[]
            dif_rew = [0]
            for i in range(1,N):
                trainer_n = nStepTD({
                    'env':env,
                    'env_name':env_name,
                    'gamma':0.9,
                    'epsilon_lower':0.05,
                    'epsilon_upper':0.8,
                    'epsilon_decay_freq':200,
                    'alpha_lower':0.05,
                    'alpha_upper':0.5,
                    'alpha_decay_freq':200,
                    'iteration':it,
                    'n':i,
                    'decay':False
                })
                trainer_n.train()

                for train_count in range(10):
                    state = env.reset()
                    episode_reward = 0
                    done = False
                    while not done:
                        action = trainer_n.greedy(state)
                        state, reward, done, _ = env.step(action)
                        episode_reward += reward
                        env.render()
                    rewards.append(episode_reward)
                dif_rew.append(np.mean(rewards))

            plt.plot(np.arange(1,N),dif_rew[1:])
        plt.legend(Iteration_steps)
        env.close()
        if trainer_n.decay:
            plt.title(f'Mean rewards as a function of the number of back-updating steps n,\n intervals={interval} with decaying alpha and epsilon')
        else:
            plt.title(f'Mean rewards as a function of the number of back-updating steps n,\n intervals={interval} without decaying alpha and epsilon')
        plt.xlabel('n')
        plt.ylabel('Mean rewards over 10 trials')
        if trainer_n.decay:
            plt.savefig(f'train_result,intev={interval},decaying.jpg')
        else:
            plt.savefig(f'train_result,intev={interval},without decaying.jpg')
    plt.show()
    