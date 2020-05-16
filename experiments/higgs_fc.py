import sys
sys.path.append('../')
from argparse import ArgumentParser
import layers
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
from torchvision import datasets, transforms
import uuid
import pickle
import os
os.environ['CUDA_VISIBLE_DEVICES']='3'


torch.cuda.set_device(3)
use_cuda = torch.cuda.is_available()

df = pd.read_csv('~/data/HIGGS.csv',  header=None)
X_train = torch.Tensor(df.loc[:len(df)-500000, np.arange(1, 29)].values)
mean = X_train.mean(0)
std = X_train.std(0)
X_train = (X_train - mean) / std
y_train = torch.Tensor(df.loc[:len(df)-500000, 0].values).type(torch.long)
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)

X_test = torch.Tensor(df.loc[len(df)-500000:, np.arange(1, 29)].values)
X_test = (X_test - mean) / std
y_test = torch.Tensor(df.loc[len(df)-500000:, 0].values).type(torch.long)
test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=20000, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=20000)


def train(args, model, device, train_loader, optimizer, epoch, alpha=1.):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        q = len(train_loader) * F.nll_loss(output, target, reduction='sum')
        w = alpha * model.get_kl()
        loss = q + w
        loss.backward()
        optimizer.step()
        if batch_idx % 1000 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.6f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return 100. * correct / len(test_loader.dataset)


from collections import OrderedDict
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc = nn.Sequential(OrderedDict([
            ('f6', Layer(28, 28)),
            ('relu6', nn.ReLU()),
            ('f7', layers.FCDeterministic(28, 10)),
            ('sig7', nn.LogSoftmax(dim=-1))
        ]))

    def forward(self, img):
        img = img.view(img.shape[0], -1)
        output = self.fc(img)
        return output

    def get_kl(self):
        kl = 0
        for child in [self.fc.f6]:
            kl += child.get_kl()

        return kl


def main():
    device = torch.device("cuda:3" if use_cuda else "cpu")
    model = Net().to(device)
    optimizer = optim.Adam(model.parameters())

    logs = []
    for epoch in range(EPOCH_NUMBER):
        train(None, model, device, train_loader, optimizer, epoch, alpha=ALPHA)
        model.fc.f6.fully_toeplitz = False
        nft_acc = test(None, model, device, test_loader)
        model.fc.f6.fully_toeplitz = True
        ft_acc = test(None, model, device, test_loader)

        if LAYER == 'vanilla':
            drop_fraq = 1.
            dropout_mean = 42.
        elif LAYER == 'fc':
            drop_fraq = 1.
            dropout_mean = 42.
        elif LAYER == 'bernoulli':
            drop_fraq = droped_fraq = (model.fc.f6.p.data.cpu().numpy().flatten() > DROP_THRESHOLD).sum() / len(
                model.fc.f6.p.data.cpu().numpy().flatten())
            dropout_mean = model.fc.f6.p.data.cpu().numpy().flatten().mean()
        elif LAYER == 'gaussian':
            drop_fraq = (model.fc.f6.logalpha.data.cpu().numpy().flatten() > DROP_THRESHOLD).sum() / len(
                model.fc.f6.logalpha.data.cpu().numpy().flatten())
            dropout_mean = model.fc.f6.logalpha.data.cpu().numpy().flatten().mean()
        else: raise NotImplementedError

        logs += [[nft_acc, ft_acc, drop_fraq, dropout_mean]]
        print(logs)
        with open(f'logs/{EXPERIMENT_NAME}.pickle', 'wb') as handle:
            pickle.dump(logs, handle)

if __name__ == '__main__':
    parser = ArgumentParser(description='Process some integers.')
    parser.add_argument('--layer', type=str, required=True, help='vanilla | bernoulli | gaussian')
    parser.add_argument('--epochs', type=int, required=True, help='epoch number')
    parser.add_argument('--alpha', type=float, required=True, help='kl-term weight')
    kwargs = vars(parser.parse_args())
    LAYER, EPOCH_NUMBER, ALPHA = kwargs['layer'], kwargs['epochs'], kwargs['alpha']
    if LAYER == 'vanilla':
        Layer = layers.FCToeplitz
        Layer.get_kl = lambda self: 0.
    elif LAYER == 'fc':
        Layer = layers.FCDeterministic
        Layer.get_kl = lambda self: 0.
    elif LAYER == 'bernoulli':
        Layer = layers.FCToeplitzBernoulli
        DROP_THRESHOLD = 0.95
    elif LAYER == 'gaussian':
        Layer = layers.FCToeplitzGaussain
        DROP_THRESHOLD = 3
    else:
        raise NotImplementedError

    EXPERIMENT_NAME = f'higgs_{LAYER}_{EPOCH_NUMBER}_{ALPHA}_{uuid.uuid4().hex}'

    main()
