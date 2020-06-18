import argparse
import random
from copy import deepcopy

import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import random_split

from tqdm import tqdm

from net import DenseNet
from client import Client
from dag import Node
import reputation


if __name__ == "__main__":
    """TODO
    # TODO: global test set
    """

    """argparse"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--nNodes', type=int, default=100)
    parser.add_argument('--batchSz', type=int, default=128)
    parser.add_argument('--nEpochs', type=int, default=300)
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--path')
    # parser.add_argument('--load', action='store_true')  # TODO
    parser.add_argument('--seed', type=int, default=1)
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
    splited_trainset = random_split(trainset, [int(len(trainset) / args.nNodes) for _ in range(args.nNodes)])
    splited_testset = random_split(testset, [int(len(testset) / args.nNodes) for _ in range(args.nNodes)])

    """Set nodes
    # TBA
    """
    def _dense_net():
        return DenseNet(growthRate=12, depth=100, reduction=0.5, bottleneck=True, nClasses=10)
        # print('>>> Number of params: {}'.format(
        #     sum([p.data.nelement() for p in net.parameters()])))

    clients = []
    for i in range(args.nNodes):
        clients.append(Client(
            args=args,
            net=_dense_net(),
            trainset=splited_trainset[i],
            testset=splited_testset[i],
            log=True))

    tmp_client = Client(  # for eval. the others' net / et al.
        args=args,
        net=_dense_net(),
        trainset=None,
        testset=None,
        log=False,
        _id=-1)

    """Set DAG
    # TODO: DAG connection
    """
    genesis = Node(
        weights=tmp_client.get_weights(),
        _id=-1)

    nodes = []
    nodes.append(genesis)

    """Run simulator
    # TODO: logging time (train, test)
    """
    latest_nodes = deepcopy(nodes)  # in DAG

    for epoch in range(1, args.nEpochs + 1):
        print(">>> Round %5d" % (epoch))

        # select activated clients
        n_activated = random.randint(1, args.nNodes)
        activateds = random.sample([t for t in range(args.nNodes)], n_activated)

        current_nodes = []
        current_accs = []

        for a in tqdm(activateds):
            client = clients[a]

            client.adjust_opt(epoch)

            """References
            # TBA
            """
            # My acc
            my_acc = 100. - client.test(epoch, show=False, log=True)

            # The others' acc
            # TODO: parameterize
            # TODO: ETA
            tmp_client.set_dataset(trainset=None, testset=client.testset)
            if len(latest_nodes) < 2:  # 1
                bests, idx_bests, _ = reputation.by_accuracy(
                    proposals=latest_nodes, count=1, test_client=tmp_client,
                    epoch=epoch, show=False, log=False,
                    timing=False, optimal_stopping=False)
            else:
                bests, idx_bests, _ = reputation.by_accuracy(
                    proposals=latest_nodes, count=2, test_client=tmp_client,
                    epoch=epoch, show=False, log=False,
                    timing=False, optimal_stopping=False)

            best_nodes = [latest_nodes[idx_best] for idx_best in idx_bests]

            # TODO: parameterize
            if (len(bests) < 2) or (bests[0] < my_acc):
                weightses = [client.get_weights(), best_nodes[0].get_weights()]
                repus_sum = my_acc + bests[0]
                repus = [my_acc / repus_sum, bests[0] / repus_sum]
            elif bests[1] < my_acc:
                weightses = [best_nodes[0].get_weights(), client.get_weights()]
                repus_sum = bests[0] + my_acc
                repus = [bests[0] / repus_sum, my_acc / repus_sum]
            else:
                weightses = [best_nodes[0].get_weights(), best_nodes[1].get_weights()]
                repus = [bests[0] / sum(bests), bests[1] / sum(bests)]

            """FL
            # own weights + the other's weights
            """
            client.set_average_weights(weightses, repus)

            # train
            client.train(epoch, show=False, log=True)

            # for logging
            current_accs.append(my_acc)

            # save weights
            client.save()

            """DAG
            # TODO
            """
            # create node
            new_node = Node(
                weights=client.get_weights())
            # nodes.append(new_node)
            current_nodes.append(new_node)

        """Log
        # TODO: save to file
        """
        print(">>> activated_clients:", activateds)
        print(">>> latest_nodes:", [d.get_id() for d in latest_nodes])
        print(">>> current_nodes:", [d.get_id() for d in current_nodes])
        print(">>> current_accs:", current_accs)
        print()

        latest_nodes = deepcopy(current_nodes)
