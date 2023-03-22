import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from tqdm import tqdm
from VAE import *

PRE_SIZE = TOTAL_SIZE
general_data = torch.tensor(np.load('44_ReduceTo_5_VAE.npy')).float()
class TrainSet(Dataset):
    def __init__(self):
        self.pre_data = general_data[:TRAIN_SIZE]
        self.post_data = general_data[PRE_SIZE:PRE_SIZE+TRAIN_SIZE]

    def __getitem__(self, index):
        return self.pre_data[index], self.post_data[index]

    def __len__(self):
        return len(self.pre_data)

class TestSet(Dataset):
    def __init__(self):
        self.pre_data = general_data[TRAIN_SIZE:PRE_SIZE]
        self.post_data = general_data[PRE_SIZE+TRAIN_SIZE:]
    
    def __getitem__(self, index):
        return self.pre_data[index], self.post_data[index]

    def __len__(self):
        return len(self.pre_data)

class DNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.BatchNorm1d(HIDDEN_UNIT),
            nn.Linear(HIDDEN_UNIT,64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64,HIDDEN_UNIT)
        )
    def forward(self, x):
        return self.network(x)


if __name__ == '__main__':

    
    trainset = TrainSet()
    testset = TestSet()

    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=False)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

    model = DNN()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    name_model = 'mapping_2.pt'
    # 6--10->32->10 
    print('-----Train-----')
    model.train()
    for epoch in range(10):
        total_loss = 0.0
        for pre, post in tqdm(trainloader):

            post_pred = model(pre)
            loss = loss_fn(post_pred, post)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss
        avg_loss = total_loss/(2*TRAIN_SIZE/BATCH_SIZE)
        print(avg_loss.item())
        print(pre[np.random.randint(0,32)])
        print(post_pred[np.random.randint(0,32)])
        print(post[np.random.randint(0,32)])
    torch.save(model.state_dict(),name_model)

    print('-----Test-----')
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for pre, post in tqdm(testloader):
            post_pred = model(pre)
            loss = loss_fn(post_pred, post)
            total_loss += loss
    avg_loss = total_loss/(2*TEST_SIZE/BATCH_SIZE)
    print(avg_loss.item())    
