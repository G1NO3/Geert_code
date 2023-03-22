from VAE import *
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

neuronset = NeuronSet()
neuronloader = DataLoader(neuronset, batch_size = 32, shuffle = False)

print('-----Dimension Reducing-----')
model=VAE(N_INPUT)
model.load_state_dict(torch.load('prepost_VAE.pt'))
model.eval()
encoded_list = []
with torch.no_grad():
    for data in tqdm(neuronloader):
        mu, lnsigma, encoded, decoded = model(data)
        for vector in encoded:
            encoded_list.append(np.array(vector))


encoded_arr = np.array(encoded_list)
print(encoded_arr[np.random.randint(0,2*(TRAIN_SIZE+TEST_SIZE),5)])
print(encoded_arr.shape)
np.save('44_ReduceTo_5_VAE.npy', encoded_arr)