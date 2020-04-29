import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc = nn.Sequential(nn.Linear(3, 10),
                                nn.SELU(),
                                nn.Linear(10, 20),
                                nn.SELU(),
                                nn.Linear(20, 20),
                                nn.SELU(),
                                nn.Linear(20, 1))

    def forward(self, x):

        return self.fc(x)