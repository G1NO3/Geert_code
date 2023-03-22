from VAE import *
import torch
import numpy as np
import Ponicare_mapping
model = VAE(N_INPUT)
model.load_state_dict(torch.load('prepost_VAE.pt'))
index = np.random.randint(0,TEST_SIZE,5)

trainset = NeuronTrainSet()
testset = NeuronTestSet()

# model = Ponicare_mapping.DNN()
# model.load_state_dict(torch.load('mapping_2.pt'))
# index = np.random.randint(0,TRAIN_SIZE,10)
# trainset = Ponicare_mapping.TrainSet()
# testset = Ponicare_mapping.TestSet()

model.eval()
with torch.no_grad():
    for i in index:
        print(i)
        data = testset[i:i+1]

        mu, lnsigma, encoded, decoded = model(data)
        loss = test_loss(decoded, data)
        KL = loss_fn(mu, lnsigma, decoded, data) - loss
        print('data:')
        print(data)
        print('mu:',mu)
        print('lnsigma:',lnsigma)
        print('z:',encoded)
        print('decoded:')
        print(decoded)
        print('loss:')
        print(loss)
        print('KL:')
        print(KL)
        print()

# model.eval()
# with torch.no_grad():
#     for i in index:
#         print(i)
#         pre,post = testset[i:i+1]

#         post_pred = model(pre)
#         loss = test_loss(post_pred, post)
#         print('pre:')
#         print(pre)
#         print('post:',post)
#         print('post_pred:',post_pred)
#         print('loss:')
#         print(loss)

#         print()

# z=torch.tensor(np.random.normal(0,3,HIDDEN_UNIT)).float().reshape(1,-1)
# x_p=model.decoding(z)
# print(x_p)

