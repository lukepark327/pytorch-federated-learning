import time
import math
import random

# from tqdm import tqdm

import torch


def by_random():
    pass


def suffle(A):
    return (list(t) for t in zip(*(random.sample([i for i in (enumerate(A))], len(A)))))


def by_accuracy(
        proposals: list, count: int, test_client,
        epoch, show=False, log=False,
        timing=False, optimal_stopping=False):

    if timing:
        start = time.time()

    n = len(proposals)
    assert(n >= count)

    bests, idx_bests, elapsed = [], [], None
    accs = []

    if optimal_stopping and (n >= 3):
        """optimal stopping mode
        # TODO: Randomize input list (proposals)
        # TODO: not a best, but t% satisfaction (10, 20, ...)
        # Ref. this: https://horizon.kias.re.kr/6053/
        """
        passing_number = int(n / math.e)
        cutline = 0.

        idx_suffled, suffled = suffle(proposals)

        for i, proposal in enumerate(suffled):
            test_client.set_weights(proposal.get_weights())
            res = 100. - test_client.test(epoch, show=show, log=log)
            accs.append(res)
            idx_bests.append(idx_suffled[i])
            if cutline < res:
                cutline = res
                if (i >= passing_number) and (i + 1 >= count):
                    break
    else:
        """normal mode
        # TBA
        """
        for i, proposal in enumerate(proposals):  # tqdm(proposals):
            test_client.set_weights(proposal.get_weights())
            res = 100. - test_client.test(epoch, show=show, log=log)
            accs.append(res)
            idx_bests.append(i)

    # print(accs)
    bests = accs[:]
    bests, idx_bests = (list(t)[:count] for t in zip(*sorted(zip(bests, idx_bests), reverse=True)))

    # elapsed time
    if timing:
        elapsed = time.time() - start
        # print(elapsed)

    return bests, idx_bests, elapsed


def filterwise_normalization(weights: dict):
    theta = Frobenius(weights)

    res = dict()
    for name, value in weights.items():
        d = Frobenius({name: value})
        res[name] = value.div(d).mul(theta)

    return res


def Frobenius(weights: dict, base_weights: dict = None):
    total = 0.
    for name, value in weights.items():
        if base_weights is not None:
            elem = value.sub(base_weights[name])
        else:
            elem = value.clone().detach()

        elem.mul_(elem)
        total += torch.sum(elem).item()

    return math.sqrt(total)


def by_Frobenius(
        proposals: list, count: int, base_client,
        FN=False,  # TODO: acc=False, test_client=None,
        timing=False, optimal_stopping=False):

    if timing:
        start = time.time()

    n = len(proposals)
    assert(n >= count)

    bests, idx_bests, elapsed = [], [], None
    distances = []

    if optimal_stopping and (n >= 3):
        """optimal stopping mode
        # TODO: Her own weights' Frobenius Norm is 0
        # so they are always best.
        """
        passing_number = int(n / math.e)
        cutline = 0.

        idx_suffled, suffled = suffle(proposals)
        cached = None

        for i, proposal in enumerate(suffled):  # enumerate(tqdm(proposals)):
            if FN:
                if cached is None:
                    cached = filterwise_normalization(base_client.get_weights())

                res = -1 * Frobenius(
                    filterwise_normalization(proposal.get_weights()),
                    base_weights=cached)
            else:
                res = -1 * Frobenius(
                    proposal.get_weights(), base_weights=base_client.get_weights())

            if i == 0:
                cutline = res

            distances.append(res)
            idx_bests.append(idx_suffled[i])

            if cutline < res:
                cutline = res
                if i >= passing_number and (i + 1 >= count):
                    break
    else:
        """normal mode
        # TBA
        """
        cached = None

        for i, proposal in enumerate(proposals):
            if FN:
                if cached is None:
                    cached = filterwise_normalization(base_client.get_weights())

                res = -1 * Frobenius(
                    filterwise_normalization(proposal.get_weights()),
                    base_weights=cached)
            else:
                res = -1 * Frobenius(
                    proposal.get_weights(), base_weights=base_client.get_weights())

            distances.append(res)
            idx_bests.append(i)

    # print(distances)
    bests = distances[:]
    bests, idx_bests = (list(t)[:count] for t in zip(*sorted(zip(bests, idx_bests), reverse=True)))

    # elapsed time
    if timing:
        elapsed = time.time() - start
        # print(elapsed)

    return bests, idx_bests, elapsed


def by_GNN():
    pass  # TODO


def by_population():
    pass  # TODO: NAS, ES


if __name__ == "__main__":
    import argparse

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

    # by_accuracy
    for c in range(10):
        print("\nClient", c)
        tmp_client.set_dataset(trainset=None, testset=clients[c].testset)

        # by accuracy
        bests, idx_bests, elapsed = by_accuracy(
            proposals=clients, count=5, test_client=tmp_client,
            epoch=1, show=False, log=False,
            timing=True, optimal_stopping=False)
        print("Acc\t:", idx_bests, elapsed)

        # by accuracy with optimal stopping
        bests, idx_bests, elapsed = by_accuracy(
            proposals=clients, count=5, test_client=tmp_client,
            epoch=1, show=False, log=False,
            timing=True, optimal_stopping=True)
        print("Acc(OS)\t:", idx_bests, elapsed)

        # by Frobenius L2 norm
        bests, idx_bests, elapsed = by_Frobenius(
            proposals=clients, count=5, base_client=clients[c],
            FN=False,  # acc=False, test_client=None,
            timing=True, optimal_stopping=False
        )
        print("F\t:", idx_bests, elapsed)

        # by Frobenius L2 norm with filter-wised normalization
        bests, idx_bests, elapsed = by_Frobenius(
            proposals=clients, count=5, base_client=clients[c],
            FN=True,  # acc=False, test_client=None,
            timing=True, optimal_stopping=False
        )
        print("F(N)\t:", idx_bests, elapsed)

        # by Frobenius L2 norm with filter-wised normalization and optimal stopping
        bests, idx_bests, elapsed = by_Frobenius(
            proposals=clients, count=5, base_client=clients[c],
            FN=True,  # acc=False, test_client=None,
            timing=True, optimal_stopping=True
        )
        print("F(N&OS)\t:", idx_bests, elapsed)
