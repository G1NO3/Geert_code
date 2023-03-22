import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from tqdm import tqdm
from VAE import *

trainset = NeuronTrainSet()
testset = NeuronTestSet()

trainloader = DataLoader(trainset, batch_size = BATCH_SIZE, shuffle = False)
testloader = DataLoader(testset, batch_size = BATCH_SIZE, shuffle = False)

model = VAE(N_INPUT)
optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)

print('-----Train-----')
model.train()
for epoch in range(5):
    total_loss = 0.0
    for x in tqdm(trainloader):
        
        mu, lnsigma, encoded, decoded = model(x)
        loss = loss_fn(mu, lnsigma, decoded, x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss
    avg_loss = total_loss/(TRAIN_SIZE*2/BATCH_SIZE)
    print(avg_loss.item())

torch.save(model.state_dict(), 'prepost_VAE.pt')

print('-----Test-----')
model.eval()
total_loss = 0.0
with torch.no_grad():
    for x in tqdm(testloader):
        mu, lnsigma, encoded, decoded = model(x)
        loss = test_loss(decoded, x)
        total_loss += loss
    avg_loss = total_loss/(TEST_SIZE*2/BATCH_SIZE)
    print(avg_loss.item())



