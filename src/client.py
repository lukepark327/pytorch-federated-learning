import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from torch.utils.data import DataLoader

import os
import numpy as np


class Client:
    _id = 0

    def __init__(self,
                 args,
                 net, trainset=None, testset=None,
                 _id=None, log=False):

        # id
        if _id != None:
            self._id = _id
        else:
            self._id = Client._id
            Client._id += 1

        self.path = (args.path or 'clients') + '/' + str(self._id)
        os.makedirs(self.path, exist_ok=True)

        # logger
        if log:
            self.trainF = open(os.path.join(self.path, 'train.csv'), 'w')
            self.testF = open(os.path.join(self.path, 'test.csv'), 'w')
        else:
            self.trainF, self.testF = None, None

        """data
        # TODO: set num_workers
        # TODO: per-client-normalization (not global)
        """
        self.cuda = args.cuda
        self.batch_size = args.batchSz

        kwargs = {'num_workers': 1, 'pin_memory': True} if self.cuda else {}
        self.trainset = trainset
        if self.trainset is not None:
            # dset.CIFAR10(root='cifar', train=True, download=True, transform=trainTransform)
            self.trainLoader = DataLoader(self.trainset,
                                          batch_size=self.batch_size, shuffle=True, **kwargs)
        self.testset = testset
        if self.testset is not None:
            # dset.CIFAR10(root='cifar', train=False, download=True, transform=testTransform)
            self.testLoader = DataLoader(self.testset,
                                         batch_size=self.batch_size, shuffle=False, **kwargs)

        """net
        # TBA
        """
        # DenseNet(growthRate=12, depth=100, reduction=0.5, bottleneck=True, nClasses=10)
        self.net = net

        if self.cuda:
            if torch.cuda.device_count() > 1:
                """DataParallel
                # TODO: setting output_device
                # torch.cuda.device_count()
                """
                self.net = nn.DataParallel(self.net)
            # else:  # one GPU
            self.net = self.net.cuda()  # use cuda

        self.opt = args.opt
        if self.opt == 'sgd':
            self.optimizer = optim.SGD(net.parameters(), lr=1e-1, momentum=0.9)  # , weight_decay=1e-4)
        elif self.opt == 'adam':
            self.optimizer = optim.Adam(net.parameters())  # , weight_decay=1e-4)
        elif self.opt == 'rmsprop':
            self.optimizer = optim.RMSprop(net.parameters())  # , weight_decay=1e-4)

        """Metadata
        # TODO: save latest acc. to reduce computation
        """
        self.acc = None

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

    def set_dataset(self, trainset=None, testset=None, batch_size=None):
        assert((trainset or testset) != None)
        batch_size = self.batch_size or batch_size

        # TODO: set num_workers
        # TODO: per-client-normalization (not global)
        kwargs = {'num_workers': 1, 'pin_memory': True} if self.cuda else {}
        self.trainset = trainset
        if self.trainset is not None:
            # dset.CIFAR10(root='cifar', train=True, download=True, transform=trainTransform)
            self.trainLoader = DataLoader(self.trainset,
                                          batch_size=batch_size, shuffle=True, **kwargs)
        self.testset = testset
        if self.testset is not None:
            # dset.CIFAR10(root='cifar', train=False, download=True, transform=testTransform),
            self.testLoader = DataLoader(self.testset,
                                         batch_size=batch_size, shuffle=False, **kwargs)

    def train(self, epoch, show=True, log=True):
        # assert((not show) or (self.trainF is None))

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

            if (self.trainF is not None) and log:
                self.trainF.write('{},{},{}\n'.format(
                    partialEpoch, loss.item(), err))
                self.trainF.flush()

    def test(self, epoch, show=True, log=True):
        # assert((not show) or (self.testF is None))

        self.net.eval()  # tells net to do evaluating

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

        if (self.testF is not None) and log:
            self.testF.write('{},{},{}\n'.format(
                epoch, test_loss, err))
            self.testF.flush()

        self.acc = 100. - err.item()

        return err.item()

    def adjust_opt(self, epoch):
        if self.opt == 'sgd':
            if epoch < 150:
                lr = 1e-1
            elif epoch == 150:
                lr = 1e-2
            elif epoch == 225:
                lr = 1e-3
            else:
                return

            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

    def _get_params(self):
        params = self.net.named_parameters()
        dict_params = dict(params)

        return dict_params

    def _set_params(self, new_params: dict):
        pass  # TODO

    def get_weights(self):
        dict_params = self._get_params()
        dict_weights = dict()

        for name, param in dict_params.items():
            dict_weights[name] = param.data

        return dict_weights

    def set_weights(self, new_weights: dict):
        net_state_dict = self.net.state_dict()
        dict_params = self._get_params()

        for name, new_weight in new_weights.items():
            if name in dict_params:
                dict_params[name].data.copy_(new_weight.data)

        net_state_dict.update(dict_params)
        self.net.load_state_dict(net_state_dict)

    def get_average_weights(self, weightses: list, repus: list):
        dict_avg_weights = dict()

        for i, repu in enumerate(repus):
            weights = weightses[i]

            for name, weight in weights.items():
                if name not in dict_avg_weights:
                    dict_avg_weights[name] = torch.zeros_like(weight)

                dict_avg_weights[name].data.add_(repu * weight.data)

        return dict_avg_weights

    def set_average_weights(self, weightses: list, repus: list):  # TODO: norm.
        self.set_weights(self.get_average_weights(weightses, repus))

    # # TODO: gradient. Is it really needeed?
    # # See https://github.com/AshwinRJ/Federated-Learning-PyTorch

    # def _get_grad(self):
    #     dict_params = self._get_params()
    #     dict_grad = dict()

    #     for name, param in dict_params.items():
    #         if param.requires_grad:
    #             dict_grad[name] = param.grad

    #     return dict_grad

    # def _set_grad(self, new_grads: dict):
    #     pass  # TODO: Applying grad to weights via GD

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

    import torchvision.datasets as dset
    import torchvision.transforms as transforms
    from torch.utils.data import random_split

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

    # set seed
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    """Data
    # TODO: get Mean and Std per client
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
    def _dense_net():
        return DenseNet(growthRate=12, depth=100, reduction=0.5, bottleneck=True, nClasses=10)
        # print('>>> Number of params: {}'.format(
        #     sum([p.data.nelement() for p in net.parameters()])))

    clients = []
    for i in range(3):
        clients.append(Client(
            args=args,
            net=_dense_net(),
            trainset=splited_trainset[i],
            testset=splited_testset[i],
            log=True))

    # Test
    wanna_see = 'module.fc.weight'

    print('Init')
    print(clients[0].get_weights()[wanna_see].data[0][0])
    print(clients[1].get_weights()[wanna_see].data[0][0])
    print(clients[2].get_weights()[wanna_see].data[0][0])

    clients[1].set_weights(clients[0].get_weights())
    clients[2].set_weights(clients[0].get_weights())

    print('Set')
    print(clients[0].get_weights()[wanna_see].data[0][0])
    print(clients[1].get_weights()[wanna_see].data[0][0])
    print(clients[2].get_weights()[wanna_see].data[0][0])

    # train
    clients[1].train(epoch=1, show=False)
    clients[2].train(epoch=1, show=False)

    print('After training')
    print(clients[0].get_weights()[wanna_see].data[0][0])
    print(clients[1].get_weights()[wanna_see].data[0][0])
    print(clients[2].get_weights()[wanna_see].data[0][0])

    # avg
    clients[0].set_average_weights(
        [clients[1].get_weights(), clients[2].get_weights()],
        [0.9, 0.1])

    print('After averaging')
    print(clients[0].get_weights()[wanna_see].data[0][0])
    print(clients[1].get_weights()[wanna_see].data[0][0])
    print(clients[2].get_weights()[wanna_see].data[0][0])
