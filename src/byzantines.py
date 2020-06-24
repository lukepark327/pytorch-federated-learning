"""
# Inherit Client.py
# TODO: variety Byz.s
"""
import torch

from client import Client


class Byzantine_Omniscience(Client):
    """
    # make sum of vectors to zero
    # Expected ?
    # TBA
    """
    pass


class Byzantine_Random(Client):
    def train(self,
              epoch, show=True, log=True):

        # random weights
        rand_weights = dict()

        weights_dict = self.get_weights()

        for name, value in weights_dict.items():
            rand_weights[name] = torch.rand_like(value)

        self.set_weights(rand_weights)


if __name__ == "__main__":
    import argparse

    import torchvision.datasets as dset
    import torchvision.transforms as transforms

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

    def _dense_net():
        return DenseNet(growthRate=12, depth=100, reduction=0.5, bottleneck=True, nClasses=10)
        # print('>>> Number of params: {}'.format(
        #     sum([p.data.nelement() for p in net.parameters()])))

    # client = Client(
    #     args=args,
    #     net=_dense_net(),
    #     trainset=trainset,
    #     testset=testset,
    #     log=False)

    client = Byzantine_Random(
        args=args,
        net=_dense_net(),
        trainset=trainset,
        testset=testset,
        log=False)

    # Test
    wanna_see = 'module.fc.weight'

    print('Init')
    print(client.get_weights()[wanna_see].data[0][0])

    print('Rand 1')
    client.train()
    print(client.get_weights()[wanna_see].data[0][0])

    print('Rand 2')
    client.train()
    print(client.get_weights()[wanna_see].data[0][0])

    print(client._id)
