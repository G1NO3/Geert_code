import torch
from torch import nn

class Resblock(nn.Module):
    def __init__(self,dim=64,kernel_size=3,stride=3,padding=1):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(dim,dim,kernel_size=kernel_size,stride=stride,padding=padding),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim,dim,kernel_size=kernel_size,stride=stride,padding=padding),
            nn.BatchNorm2d(dim)
        )
        self.ReLU_layer = nn.ReLU()

    def forward(self, x):
        out = self.layer(x)
        out = out + x
        out = self.ReLU_layer(out)
        return out 

class CNNModel(nn.Module):

    def make_block(self,num):
        block_list = []
        for i in range(num):
            block_list.append(Resblock(64))
        return nn.Sequential(*block_list)

    def __init__(self):
        nn.Module.__init__(self)
        self._tower = nn.Sequential(
            nn.Conv2d(135, 128, 3, 1, 1, bias = False),
            nn.Conv2d(128, 64, 3, 1, 1, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        self.blocks = self.make_block(9)
        self._logits = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 9, 256),
            nn.ReLU(True),
            nn.Linear(256, 235)
        )
        self._value_branch = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 9, 256),
            nn.ReLU(True),
            nn.Linear(256, 1)
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, input_dict):
        obs = input_dict["observation"].float()
        hidden = self._tower(obs)
        hidden = self.blocks(hidden)
        logits = self._logits(hidden)
        mask = input_dict["action_mask"].float()
        inf_mask = torch.clamp(torch.log(mask), -1e38, 1e38)
        masked_logits = logits + inf_mask
        value = self._value_branch(hidden)
        return masked_logits, value