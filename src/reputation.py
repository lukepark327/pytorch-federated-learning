import time
import math

from tqdm import tqdm


def by_accuracy(
        proposals: list, count: int, test_client,
        epoch, show=False, log=False,
        time=False, optimal_stopping=False):

    assert(len(proposals) >= count)

    bests, idx_bests, elapsed = [], [], None

    if time:
        start = time.time()

    """optimal stopping mode
    # TODO: more than 1
    # TODO: not a best, but t% satisfaction (10, 20, ...)
    # Ref. this: https://horizon.kias.re.kr/6053/
    """
    if optimal_stopping:
        n = len(proposals)
        assert(n >= 3)  # TODO: remove restriction

        passing_number = int(n / math.e)
        passings, watches = [], []
        cutline, idx_cutline = 0., 0

        for i, proposal in enumerate(tqdm(proposals)):
            if i < passing_number:  # passing
                test_client.set_weights(proposal.get_weights())
                res = 100. - test_client.test(epoch, show=show, log=log)
                if cutline < res:
                    cutline, idx_cutline = res, i
            else:  # wathing
                test_client.set_weights(proposal.get_weights())
                res = 100. - test_client.test(epoch, show=show, log=log)
                if cutline < res:
                    cutline, idx_cutline = res, i
                    break

        bests, idx_bests = [cutline], [idx_cutline]

    """normal mode
    # TBA
    """
    else:
        accs = []
        for proposal in tqdm(proposals):
            test_client.set_weights(proposal.get_weights())
            accs.append(
                100. - test_client.test(epoch, show=show, log=log))

        bests = sorted(accs)[:count]
        for best in bests:
            idx_bests.append(accs.index(best))

    # elapsed time
    if time:
        elapsed = time.time() - start
        # print(elapsed)

    return bests, idx_bests, elapsed


def by_Frobenius(
        proposals: list, count: int, net,
        time=False, optimal_stopping=False):

    pass


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
    for _ in range(2):
        clients[1].train(epoch=1, show=False)
    for _ in range(4):
        clients[2].train(epoch=1, show=False)

    bests, idx_bests, elapsed = by_accuracy(
        proposals=clients, count=1, test_client=tmp_client,
        epoch=1, show=False, log=False,
        time=True, optimal_stopping=False
    )

    print(bests, idx_cutline, elapsed)
