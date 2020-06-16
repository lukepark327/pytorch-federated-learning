"""Ref
# Ref: https://github.com/bamos/densenet.pytorch
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

from torch.utils.data import DataLoader, random_split

import os
import numpy as np


class Client:
    def __init__(self,
                 args,
                 _id,
                 net,
                 trainset=None, testset=None):

        assert(_id != None)
        self._id = _id

        self.path = args.path or ('clients/' + str(self._id))
        os.makedirs(self.path, exist_ok=True)

        """data
        # TODO: set num_workers
        # TODO: per-client-normalization (not global)
        """
        kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}
        self.trainset = trainset
        if self.trainset is not None:
            # dset.CIFAR10(root='cifar', train=True, download=True, transform=trainTransform)
            self.trainLoader = DataLoader(self.trainset,
                                          batch_size=args.batchSz, shuffle=True, **kwargs)
        self.testset = testset
        if self.testset is not None:
            # dset.CIFAR10(root='cifar', train=False, download=True, transform=testTransform)
            self.testLoader = DataLoader(self.testset,
                                         batch_size=args.batchSz, shuffle=False, **kwargs)

        """net
        # TBA
        """
        # DenseNet(growthRate=12, depth=100, reduction=0.5, bottleneck=True, nClasses=10)
        self.net = net

        self.cuda = args.cuda
        if self.cuda:
            if torch.cuda.device_count() > 1:
                """DataParallel
                # TODO: setting output_device
                # torch.cuda.device_count()
                """
                self.net = nn.DataParallel(self.net)
            # else:  # one GPU
            self.net = self.net.cuda()  # use cuda

        if args.opt == 'sgd':
            self.optimizer = optim.SGD(net.parameters(), lr=1e-1, momentum=0.9, weight_decay=1e-4)
        elif args.opt == 'adam':
            self.optimizer = optim.Adam(net.parameters(), weight_decay=1e-4)
        elif args.opt == 'rmsprop':
            self.optimizer = optim.RMSprop(net.parameters(), weight_decay=1e-4)

    """ML
    # TBA
    """

    def save(self, path=None, name='latest.pth'):
        path = path or self.path
        loca = os.path.join(path, name)

        torch.save(self.net, loca)

    def load(self, path=None, name='latest.pth'):
        path = path or self.path
        loca = os.path.join(path, name)

        if os.path.isfile(loca):
            # print(">>> Load weights:", loca)
            self.net = torch.load(loca)
        # else:
            # print(">>> No pre-trained weights")

    def set_dataset(self, trainset=None, testset=None):
        assert((trainset or testset) != None)

        # TODO: set num_workers
        # TODO: per-client-normalization (not global)
        kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}
        self.trainset = trainset
        if self.trainset is not None:
            # dset.CIFAR10(root='cifar', train=True, download=True, transform=trainTransform)
            self.trainloader = DataLoader(self.trainset,
                                          batch_size=args.batchSz, shuffle=True, **kwargs)
        self.testset = testset
        if self.testset is not None:
            # dset.CIFAR10(root='cifar', train=False, download=True, transform=testTransform),
            self.testloader = DataLoader(self.testset,
                                         batch_size=args.batchSz, shuffle=False, **kwargs)

    def train(self,
              epoch,
              show=True, logger=None):

        assert((not show) or (logger is None))

        self.net.train()

        nProcessed = 0
        nTrain = len(self.trainLoader.dataset)

        for batch_idx, (data, target) in enumerate(self.trainLoader):

            if self.cuda:
                data, target = data.cuda(), target.cuda()

            data, target = Variable(data), Variable(target)
            self.optimizer.zero_grad()
            output = self.net(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            self.optimizer.step()

            nProcessed += len(data)
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            incorrect = pred.ne(target.data).cpu().sum()
            err = 100. * incorrect / len(data)
            partialEpoch = epoch + batch_idx / len(self.trainLoader) - 1

            if show:
                print('Train Epoch: {:.2f} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tError: {:.6f}'.format(
                    partialEpoch, nProcessed, nTrain, 100. * batch_idx / len(self.trainLoader),
                    loss.item(), err))

            if logger is not None:
                logger.write('{},{},{}\n'.format(
                    partialEpoch, loss.item(), err))
                logger.flush()

            # break  # TODO: TMP

    def test(self,
             epoch,
             show=True, logger=None):

        assert((not show) or (logger is None))

        net.eval()  # tells net to do evaluating

        test_loss = 0
        incorrect = 0

        for data, target in self.testLoader:

            if self.cuda:
                data, target = data.cuda(), target.cuda()

            with torch.no_grad():
                # data, target = Variable(data), Variable(target)
                output = self.net(data)
                test_loss += F.nll_loss(output, target).item()
                pred = output.data.max(1)[1]  # get the index of the max log-probability
                incorrect += pred.ne(target.data).cpu().sum()

        test_loss = test_loss
        test_loss /= len(self.testLoader)  # loss function already averages over batch size
        nTotal = len(self.testLoader.dataset)
        err = 100. * incorrect / nTotal

        if show:
            print('\nTest set: Average loss: {:.4f}, Error: {}/{} ({:.0f}%)\n'.format(
                test_loss, incorrect, nTotal, err))

        if logger is not None:
            logger.write('{},{},{}\n'.format(
                epoch, test_loss, err))
            logger.flush()

    def _get_params(self):
        params = self.net.named_parameters()
        dict_params = dict(params)

        return dict_params

    def _set_params(self, new_params: dict):
        pass  # TODO

    def get_weights(self):
        dict_params = self._get_params()
        dict_grad = dict()

        for name, param in dict_params.items():
            dict_grad[name] = param.data

        return dict_grad

    def set_weights(self, new_weights: dict):
        net_state_dict = self.net.state_dict()
        params = self.net.named_parameters()
        dict_params = dict(params)

        for name, new_weight in new_weights.items():
            if name in dict_params:
                dict_params[name].data.copy_(new_weight.data)

        net_state_dict.update(dict_params)
        self.net.load_state_dict(net_state_dict)

    def get_grad(self):
        dict_params = self._get_params()
        dict_grad = dict()

        for name, param in dict_params.items():
            if param.requires_grad:
                dict_grad[name] = param.grad

        return dict_grad

    def set_grad(self, new_grads: dict):
        # TODO: Applying grad to weights via GD
        pass

    # def average_weights(self):
    #     pass

    # def set_average_weights(self, paramses: list, repus: list):  # TODO: norm.
    #     model_dict = self.net.state_dict()

    #     my_params = self.net.named_parameters()
    #     dict_my_params = dict(my_params)

    #     # set zeros
    #     for name in paramses[0].keys():
    #         if name in dict_my_params:
    #             dict_my_params[name].data.zero_()

    #     for i, repu in enumerate(repus):
    #         params = paramses[i]

    #         for name, param in params.items():
    #             if name in dict_my_params:
    #                 dict_my_params[name].data.add_(repu * param.data)

    #     model_dict.update(dict_my_params)
    #     self.net.load_state_dict(model_dict)

    """DAG
    # TODO
    """

    def select_node(self):
        pass

    def test_node(self):
        pass

    def create_node(self):
        pass

    """Viz
    # TODO: tensorboard
    """


if __name__ == "__main__":
    import argparse
    from net import DenseNet

    """argparse"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSz', type=int, default=128)
    parser.add_argument('--nEpochs', type=int, default=300)
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--path')
    parser.add_argument('--seed', type=int, default=950327)
    parser.add_argument('--opt', type=str, default='sgd',
                        choices=('sgd', 'adam', 'rmsprop'))
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    """ML"""
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    """Data
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

    trainset = dset.CIFAR10(root='cifar', train=True, download=True, transform=trainTransform)
    testset = dset.CIFAR10(root='cifar', train=False, download=True, transform=testTransform)

    # Random split
    splited_trainset = random_split(trainset, [15000, 25000, 10000])
    splited_testset = random_split(testset, [2000, 6000, 2000])

    """FL
    # TBA
    """
    net = DenseNet(growthRate=12, depth=100, reduction=0.5, bottleneck=True, nClasses=10)
    # print('>>> Number of params: {}'.format(
    #     sum([p.data.nelement() for p in net.parameters()])))

    '''
    if args.cuda:

        if torch.cuda.device_count() > 1:
            """DataParallel
            # TODO: setting output_device
            # torch.cuda.device_count()
            """
            net = nn.DataParallel(net)

        net = net.cuda()

    if args.opt == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=1e-1, momentum=0.9, weight_decay=1e-4)
    elif args.opt == 'adam':
        optimizer = optim.Adam(net.parameters(), weight_decay=1e-4)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(net.parameters(), weight_decay=1e-4)

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
    '''

    '''
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
    '''

    # clients[1].set_weights(clients[0].get_weights())
    # clients[2].set_weights(clients[0].get_weights())

    # # clients[0].eval(r=0)
    # # clients[1].eval(r=0)
    # # clients[2].eval(r=0)

    # clients[0].set_average_weights(
    #     [clients[1].get_weights(), clients[2].get_weights()],
    #     [0.9, 0.1])

    # # clients[0].eval(r=1)
    # # clients[1].eval(r=1)
    # # clients[2].eval(r=1)

    clients = []
    for i in range(3):
        clients.append(Client(
            args=args,
            _id=i,
            net=net,
            trainset=splited_trainset[i],
            testset=splited_testset[i]))
    # print(clients[0].net.state_dict().keys())

    """Test"""
    print(clients[0].get_weights()['fc.bias'])
    print(clients[0].get_grad()['fc.bias'])

    clients[0].train(epoch=1)

    print(clients[0].get_weights()['fc.bias'])
    print(clients[0].get_grad()['fc.bias'])
