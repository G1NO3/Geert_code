import torch
from torch import nn
from torch.utils.data import Dataset
import numpy as np

TRAIN_SIZE = 100000
TEST_SIZE = 20000
TOTAL_SIZE = TRAIN_SIZE+TEST_SIZE
BATCH_SIZE = 32
N_INPUT = 44
HIDDEN_UNIT = 10
train_index = np.arange(TRAIN_SIZE + TEST_SIZE)
np.random.shuffle(train_index)

NAME_preset = 'data/neuronset_pre_44.pth'
NAME_postset = 'data/neuronset_post_44.pth'

class NeuronTrainSet(Dataset):
    def __init__(self):
        self.data1 = torch.load(NAME_preset)[train_index[:TRAIN_SIZE]]
        self.data2 = torch.load(NAME_postset)[train_index[:TRAIN_SIZE]]
        self.data = torch.cat((self.data1,self.data2))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

class NeuronTestSet(Dataset):
    def __init__(self):
        self.data1 = torch.load(NAME_preset)[train_index[TRAIN_SIZE:]]
        self.data2 = torch.load(NAME_postset)[train_index[TRAIN_SIZE:]]
        self.data = torch.cat((self.data1,self.data2))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

class VAE(nn.Module):
    def __init__(self, n_input=N_INPUT, hidden_unit=HIDDEN_UNIT):
        super().__init__()
        self.hidden_unit = hidden_unit
        self.encoder = nn.Sequential(
            nn.Linear(n_input,64),
            nn.ReLU(),
            nn.BatchNorm1d(64)
        )
        self.mu_net = nn.Sequential(
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,hidden_unit),
        )
        self.lnsigma_net = nn.Sequential(
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,hidden_unit),
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_unit,32),
            nn.ReLU(),
            nn.Linear(32,32),
            nn.ReLU(),
            nn.Linear(32,n_input),
            nn.ReLU()
        )

    def forward(self,x):
        encoded = self.encoder(x)
        mu = self.mu_net(encoded)
        lnsigma = self.lnsigma_net(encoded)
        z = torch.tensor(np.random.normal(0,1,size=self.hidden_unit)).float() * torch.exp(lnsigma) + mu

        decoded = self.decoder(z)
        return mu, lnsigma, z, decoded

    def decoding(self,z):
        with torch.no_grad():
            return self.decoder(z)

def loss_fn(mu, lnsigma, decoded, data):
    # softmax = nn.Softmax()
    # p = softmax(data)
    # q = softmax(decoded)
    # reconstructed_error = torch.sum ( - p * torch.log(q) )
    criterion = nn.MSELoss()
    reconstructed_error = criterion(decoded, data)
    KL_div = torch.sum(mu.pow(2) + torch.exp(lnsigma) - lnsigma - 1)/5000
    return reconstructed_error + KL_div

def test_loss(decoded, data):
    # softmax = nn.Softmax()
    # p = softmax(data)
    # q = softmax(decoded)
    # reconstructed_error = torch.sum ( - p * torch.log(q) )
    criterion = nn.MSELoss()
    reconstructed_error = criterion(decoded, data)
    return reconstructed_error

class SimpleAutoencoder(nn.Module):
    def __init__(self,n_input=N_INPUT, hidden_unit=HIDDEN_UNIT):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_input,128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64,hidden_unit),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_unit,64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64,128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,n_input),
            nn.ReLU()
        )

    def forward(self,x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
    
    def decoding(self,z):
        with torch.no_grad():
            return self.decoder(z)

class NeuronSet(Dataset):
    def __init__(self):
        self.data_pre = torch.load(NAME_preset)
        self.data_post = torch.load(NAME_postset)
        self.data = torch.cat((self.data_pre,self.data_post),axis=0)
        self.data = torch.tensor(self.data).float()

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)