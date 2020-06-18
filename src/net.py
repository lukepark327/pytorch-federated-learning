"""Ref
https://github.com/bamos/densenet.pytorch
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.models as models
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader

import math


class Bottleneck(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(Bottleneck, self).__init__()
        interChannels = 4 * growthRate
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(interChannels)
        self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat((x, out), 1)
        return out


class SingleLayer(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = torch.cat((x, out), 1)
        return out


class Transition(nn.Module):
    def __init__(self, nChannels, nOutChannels):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1,
                               bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):
    def __init__(self, growthRate, depth, reduction, nClasses, bottleneck):
        super(DenseNet, self).__init__()

        nDenseBlocks = (depth - 4) // 3
        if bottleneck:
            nDenseBlocks //= 2

        nChannels = 2 * growthRate
        self.conv1 = nn.Conv2d(3, nChannels, kernel_size=3, padding=1,
                               bias=False)
        self.dense1 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks * growthRate
        nOutChannels = int(math.floor(nChannels * reduction))
        self.trans1 = Transition(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense2 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks * growthRate
        nOutChannels = int(math.floor(nChannels * reduction))
        self.trans2 = Transition(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense3 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks * growthRate

        self.bn1 = nn.BatchNorm2d(nChannels)
        self.fc = nn.Linear(nChannels, nClasses)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_dense(self, nChannels, growthRate, nDenseBlocks, bottleneck):
        layers = []
        for i in range(int(nDenseBlocks)):
            if bottleneck:
                layers.append(Bottleneck(nChannels, growthRate))
            else:
                layers.append(SingleLayer(nChannels, growthRate))
            nChannels += growthRate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.dense3(out)
        out = torch.squeeze(F.avg_pool2d(F.relu(self.bn1(out)), 8))
        out = F.log_softmax(self.fc(out), dim=1)
        return out


def train(args, epoch, net, trainLoader, optimizer, logger=None, show=False):
    if (not show) and (logger is None):
        return

    net.train()  # tells net to do training

    nProcessed = 0
    nTrain = len(trainLoader.dataset)

    for batch_idx, (data, target) in enumerate(trainLoader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()

        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = net(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        nProcessed += len(data)
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        incorrect = pred.ne(target.data).cpu().sum()
        err = 100. * incorrect / len(data)
        partialEpoch = epoch + batch_idx / len(trainLoader) - 1

        if show:
            print('Train Epoch: {:.2f} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tError: {:.6f}'.format(
                partialEpoch, nProcessed, nTrain, 100. * batch_idx / len(trainLoader),
                loss.item(), err))

        if logger is not None:
            logger.write('{},{},{}\n'.format(partialEpoch, loss.item(), err))
            logger.flush()


def test(args, epoch, net, testLoader, optimizer, logger=None, show=True):
    if (not show) and (logger is None):
        return

    net.eval()  # tells net to do evaluating

    test_loss = 0
    incorrect = 0

    for data, target in testLoader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()

        with torch.no_grad():
            # data, target = Variable(data), Variable(target)
            output = net(data)
            test_loss += F.nll_loss(output, target).item()
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            incorrect += pred.ne(target.data).cpu().sum()

    test_loss = test_loss
    test_loss /= len(testLoader)  # loss function already averages over batch size
    nTotal = len(testLoader.dataset)
    err = 100. * incorrect / nTotal

    if show:
        print('\nTest set: Average loss: {:.4f}, Error: {}/{} ({:.0f}%)\n'.format(
            test_loss, incorrect, nTotal, err))

    if logger is not None:
        logger.write('{},{},{}\n'.format(epoch, test_loss, err))
        logger.flush()


def adjust_opt(optAlg, optimizer, epoch):
    if optAlg == 'sgd':
        if epoch < 150:
            lr = 1e-1
        elif epoch == 150:
            lr = 1e-2
        elif epoch == 225:
            lr = 1e-3
        else:
            return

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


if __name__ == '__main__':
    import argparse
    import setproctitle
    import os
    import shutil

    """argparse"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSz', type=int, default=128)
    parser.add_argument('--nEpochs', type=int, default=300)
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--path')
    parser.add_argument('--no-load', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--opt', type=str, default='sgd',
                        choices=('sgd', 'adam', 'rmsprop'))
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.path = args.path or 'data/base'
    setproctitle.setproctitle(args.path)

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # if os.path.exists(args.path):
    #     shutil.rmtree(args.path)
    os.makedirs(args.path, exist_ok=True)

    """normalization
    # TODO: get Mean and Std
    # Ref: https://github.com/bamos/densenet.pytorch
    """
    normMean = [0.49139968, 0.48215827, 0.44653124]
    normStd = [0.24703233, 0.24348505, 0.26158768]
    normTransform = transforms.Normalize(normMean, normStd)

    trainTransform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normTransform
    ])
    testTransform = transforms.Compose([
        transforms.ToTensor(),
        normTransform
    ])

    """data
    # TODO: set num_workers
    """
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    trainLoader = DataLoader(
        dset.CIFAR10(root='cifar', train=True, download=True, transform=trainTransform),
        batch_size=args.batchSz, shuffle=True, **kwargs)
    testLoader = DataLoader(
        dset.CIFAR10(root='cifar', train=False, download=True, transform=testTransform),
        batch_size=args.batchSz, shuffle=False, **kwargs)

    """net
    # TODO: remove batch normalization (and residual connection ?)
    """
    net = DenseNet(growthRate=12, depth=100, reduction=0.5, bottleneck=True, nClasses=10)
    print('>>> Number of params: {}'.format(
        sum([p.data.nelement() for p in net.parameters()])))

    if args.cuda:

        if torch.cuda.device_count() > 1:
            """DataParallel
            # TODO: setting output_device
            # torch.cuda.device_count()
            """
            net = nn.DataParallel(net)

        net = net.cuda()

    if args.opt == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=1e-1, momentum=0.9)  # , weight_decay=1e-4)
    elif args.opt == 'adam':
        optimizer = optim.Adam(net.parameters())  # , weight_decay=1e-4)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(net.parameters())  # , weight_decay=1e-4)

    # load
    if not args.no_load:
        path_and_file = os.path.join(args.path, 'latest.pth')

        if os.path.isfile(path_and_file):
            print(">>> Load weights:", path_and_file)
            net = torch.load(path_and_file)
        else:
            print(">>> No pre-trained weights")

    # log files
    trainF = open(os.path.join(args.path, 'train.csv'), 'w')
    testF = open(os.path.join(args.path, 'test.csv'), 'w')

    """train and test"""
    for epoch in range(1, args.nEpochs + 1):

        adjust_opt(args.opt, optimizer, epoch)

        train(args, epoch, net, trainLoader, optimizer, show=True, logger=trainF)
        test(args, epoch, net, testLoader, optimizer, show=True, logger=testF)

        # save
        torch.save(net, os.path.join(args.path, 'latest.pth'))

    trainF.close()
    testF.close()
