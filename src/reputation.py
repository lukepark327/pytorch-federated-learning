import time
import math

# from tqdm import tqdm


def by_random():
    pass


def by_accuracy(
        proposals: list, count: int, test_client,
        epoch, show=False, log=False,
        timing=False, optimal_stopping=False):

    n = len(proposals)
    assert(n >= count)

    bests, idx_bests, elapsed = [], [], None
    accs = []

    if timing:
        start = time.time()

    if optimal_stopping and (n >= 3):
        """optimal stopping mode
        # TODO: not a best, but t% satisfaction (10, 20, ...)
        # Ref. this: https://horizon.kias.re.kr/6053/
        """
        passing_number = int(n / math.e)
        passings, watches = [], []
        cutline = 0.

        for i, proposal in enumerate(proposals):  # enumerate(tqdm(proposals)):
            test_client.set_weights(proposal.get_weights())
            res = 100. - test_client.test(epoch, show=show, log=log)
            accs.append(res)
            idx_bests.append(i)
            if cutline < res:
                cutline = res
                if i >= passing_number:
                    break
    else:
        """normal mode
        # TBA
        """
        for i, proposal in enumerate(proposals):  # tqdm(proposals):
            test_client.set_weights(proposal.get_weights())
            accs.append(
                100. - test_client.test(epoch, show=show, log=log))
            idx_bests.append(i)

    # print(accs)
    bests = accs[:]
    bests, idx_bests = (list(t)[:count] for t in zip(*sorted(zip(bests, idx_bests), reverse=True)))

    # elapsed time
    if timing:
        elapsed = time.time() - start
        # print(elapsed)

    return bests, idx_bests, elapsed


def by_Frobenius(
        proposals: list, count: int, test_client,
        epoch, show=False, log=False,
        timing=False, optimal_stopping=False):

    pass


def by_GNN():
    pass  # TODO


def by_population():
    pass  # TODO: NAS, ES


if __name__ == "__main__":
    import argparse

    import torch
    import torchvision.datasets as dset
    import torchvision.transforms as transforms
    from torch.utils.data import random_split

    from net import DenseNet
    from client import Client

    """argparse"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSz', type=int, default=128)
    parser.add_argument('--nEpochs', type=int, default=300)
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--load', action='store_true')
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
    splited_trainset = random_split(trainset, [int(len(trainset) / 10) for _ in range(10)])
    splited_testset = random_split(testset, [int(len(testset) / 10) for _ in range(10)])

    """FL
    # TBA
    """
    def _dense_net():
        return DenseNet(growthRate=12, depth=100, reduction=0.5, bottleneck=True, nClasses=10)
        # print('>>> Number of params: {}'.format(
        #     sum([p.data.nelement() for p in net.parameters()])))

    clients = []
    for i in range(10):
        clients.append(Client(
            args=args,
            net=_dense_net(),
            trainset=splited_trainset[i],
            testset=splited_testset[i],
            log=True and (not args.load)))

    # Test
    # clients[0].set_weights(clients[0].get_weights())
    clients[1].set_weights(clients[0].get_weights())
    clients[2].set_weights(clients[0].get_weights())

    tmp_client = Client(  # for eval. the others' net / et al.
        args=args,
        net=_dense_net(),
        trainset=None,
        testset=None,
        log=False,
        _id=-1)

    # train
    # clients[0].train(epoch=1, show=False)

    if args.load:
        for i in range(10):
            clients[i].load()
    else:
        for c in range(10):
            for i in range(1, 2):
                clients[c].train(epoch=i, show=True)
            clients[c].save()

    tmp_client.set_dataset(trainset=None, testset=clients[0].testset)
    bests, idx_bests, elapsed = by_accuracy(
        proposals=clients, count=2, test_client=tmp_client,
        epoch=1, show=False, log=False,
        timing=True, optimal_stopping=True
    )

    print(bests, idx_bests, elapsed)
