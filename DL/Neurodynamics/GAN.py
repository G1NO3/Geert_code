import numpy as np 
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
import torchvision
from tqdm import tqdm
import matplotlib.pyplot as plt

HIDDEN_UNITS = 50
batch_size_train = 64
batch_size_test = 32
device = 'cuda:0'

class TrainSet(Dataset):
    def __init__(self,data):
        super().__init__()
        self.data = data
    def __getitem__(self,index):
        return self.data[index]
    def __len__(self):
        return len(self.data)

class TestSet(Dataset):
    def __init__(self,data):
        super().__init__()
        self.data = data
    def __getitem__(self,index):
        return self.data[index]
    def __len__(self):
        return len(self.data)

class Generator(nn.Module):
    def __init__(self,n_input=HIDDEN_UNITS):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(n_input, 64),
            nn.ReLU(),
            nn.Linear(64,128),
            nn.ReLU(),
            nn.Linear(128,256),
            nn.ReLU(),
        )
        self.generate = nn.Sequential(
            nn.ConvTranspose2d(1,3,3),
            nn.ReLU(),
            nn.ConvTranspose2d(3,5,5),
            nn.ReLU(),
            nn.ConvTranspose2d(5,3,5),
            nn.ReLU(),
            nn.ConvTranspose2d(3,1,3),
            nn.Tanh(),
        )

    def forward(self,x):
        flatten = self.linear(x)
        out = self.generate(flatten.reshape(-1,1,16,16))
        return out


class Discriminator(nn.Module):
    def __init__(self,n_input=28):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1,3,3),
            nn.ReLU(),
            nn.Conv2d(3,5,3),
            nn.ReLU(),
            nn.Conv2d(5,3,5),
            nn.ReLU(),
            nn.Conv2d(3,1,5),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,1),
        )
    def forward(self,x):
        return self.network(x)


class Generator(nn.Module):
    def __init__(self,n_input=HIDDEN_UNITS):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_input, 64),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(64,0.8),
            nn.Linear(64,128),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(128,0.8),
            nn.Linear(128,256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256,0.8),
            nn.Linear(256,512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512,0.8),
            nn.Linear(512,784),
            nn.Tanh()
        )
    def forward(self,x):
        return self.network(x).reshape(1,-1,28,28)

class Discriminator(nn.Module):
    def __init__(self,n_input=28):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(start_dim=2),
            nn.Linear(784,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,1),
        )
    def forward(self,x):
        return self.network(x)



train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data/', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.5,), (0.5,))
                               ])),
    batch_size=batch_size_train, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.5,), (0.5,))
                               ])),
    batch_size=batch_size_test, shuffle=True)


D = Discriminator().to(device)
G = Generator().to(device)
Doptimizer = torch.optim.RMSprop(D.parameters(),lr=1e-3)
Goptimizer = torch.optim.RMSprop(G.parameters(),lr=1e-3)
# loss_fn = nn.BCELoss()


G.load_state_dict(torch.load('G.pt'))
D.load_state_dict(torch.load('D.pt'))


D.train()
G.train()
print('-----Train')
for epoch in range(20):
    total_Dloss = 0
    total_Gloss = 0
    for i, (data, target) in tqdm(enumerate(train_loader)):
        z = torch.tensor(np.random.normal(0,1,(batch_size_train,HIDDEN_UNITS))).float().to(device)
        x_prime = G(z)

        true_score = D(data.to(device))
        false_score = D(x_prime.detach())
        D_loss = - torch.mean(true_score) + torch.mean(false_score)
        Doptimizer.zero_grad()
        D_loss.backward()
        Doptimizer.step()

        for p in D.parameters():
            p.data.clamp_(-0.01,0.01)
        false_score = D(x_prime)
        G_loss = - torch.mean(false_score)
        Goptimizer.zero_grad()
        G_loss.backward()
        Goptimizer.step()


        total_Dloss += D_loss
        total_Gloss += G_loss
    avg_G_loss = total_Gloss/(len(train_loader))
    avg_D_loss = total_Dloss/(len(train_loader))
    print('G_loss:',avg_G_loss.item())
    print('D_loss:',avg_D_loss.item())
torch.save(G.state_dict(),'G.pt')
torch.save(D.state_dict(),'D.pt')



print('-----Test')
G.eval()
D.eval()
total_Dloss = 0
total_Gloss = 0
with torch.no_grad():
    for data, target in tqdm(test_loader):
        z = torch.randn(batch_size_train, HIDDEN_UNITS).to(device)
        x_prime = G(z)
        true_score = D(data.to(device))
        false_score = D(x_prime)
        D_loss = - true_score.mean() + false_score.mean()
        G_loss = - false_score.mean()
        total_Dloss += D_loss
        total_Gloss += G_loss
    avg_G_loss = total_Gloss/(len(test_loader))
    avg_D_loss = total_Dloss/(len(test_loader))
    print('G_loss:',avg_G_loss.item())
    print('D_loss:',avg_D_loss.item())

for i in range(6):
    z = torch.randn((1,HIDDEN_UNITS)).to(device)
    img = G(z).detach().cpu().numpy().squeeze()
    plt.subplot(2,3,i+1)
    plt.imshow(img)
plt.savefig('fig')
plt.show()




