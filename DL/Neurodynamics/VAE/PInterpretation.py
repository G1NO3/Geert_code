from VAE import *
import torch
import numpy as np
import Ponicare_mapping

model = Ponicare_mapping.DNN()
model.load_state_dict(torch.load('mapping_2.pt'))
index = np.random.randint(0,TEST_SIZE,10)
trainset = Ponicare_mapping.TrainSet()
testset = Ponicare_mapping.TestSet()

model.eval()
with torch.no_grad():
    for i in index:
        print(i)
        pre, post = testset[i:i+1]
        post_pred = model(pre)
        loss = test_loss(post_pred, post)
        print('pre:',pre)
        print('post:',post)
        print('post_pred:',post_pred)
        print('loss:',loss)
        print()

z=torch.tensor(np.random.normal(0,3,HIDDEN_UNIT)).float().reshape(1,-1)
x_p=model.decoding(z)
print(x_p)
