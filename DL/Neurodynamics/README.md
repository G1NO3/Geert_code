This is a part I'm working on recently. I have merged a result published on ICLR to my autoencoder for feature extraction and the result is quite satisfying.  
(you can see their codes at thuml/Time-Series-Library/models/TimesNet.py)  
And I am also trying to use StackGAN to hierarchically render the delicate dynamics of the neurons.  
  
In this part I want to describe my own style of coding. First I will try some simple models and write them in a single file(see GAN.py).  
Then when code gets bigger, I will separate it into several files(see TimesAE and VAE). Basically there are four, one for preprocessing, one for model definition and the other two are for training and interpretation of the results.  
This is for DL. For RL, the files are respectively for environment, model, actor, learner and training(see DeepRL) .
