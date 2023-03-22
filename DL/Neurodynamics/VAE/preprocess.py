import numpy as np
import torch

f1 = open('X.txt')
f2 = open('Y.txt')
NAME_preset = 'data/neuronset_pre_44.pth'
NAME_postset = 'data/neuronset_post_44.pth'
N_INPUT = 44

datalist = []
while True:
    a = list(map(int, f1.readline().split()))[:N_INPUT]
    if a :
        datalist.append(a)
    else:
        f1.close()
        break
x = np.array(datalist)
print(x[np.random.randint(0,80000,5)])
print(x.shape)
torch.save(torch.tensor(np.array(datalist)).float(), NAME_preset)


datalist = []
while True:
    a = list(map(int, f2.readline().split()))[:N_INPUT]
    if a :
        datalist.append(a)
    else:
        f2.close()
        break
x = np.array(datalist)
print(x.shape)
print(x[np.random.randint(0,80000,5)])
torch.save(torch.tensor(np.array(datalist)).float(), NAME_postset)