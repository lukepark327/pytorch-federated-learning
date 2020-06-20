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
from byzantines import Byzantine_Random
from dag import Node
import reputation


if __name__ == "__main__":
    """TODO
    # TODO: global test set
    """

    """argparse"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--nNodes', type=int, default=100)
    parser.add_argument('--nByzs', type=int, default=33)
    parser.add_argument('--batchSz', type=int, default=128)
    parser.add_argument('--nEpochs', type=int, default=300)
    parser.add_argument('--op-stop', action='store_true')
    parser.add_argument('--filter', action='store_true')
    parser.add_argument('--repute', type=str, default='acc',
                        choices=('acc', 'Frobenius', 'random', 'GNN'))
    parser.add_argument('--path')
    parser.add_argument('--no-cuda', action='store_true')
    # parser.add_argument('--load', action='store_true')  # TODO
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--opt', type=str, default='sgd',
                        choices=('sgd', 'adam', 'rmsprop'))
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    args.norm = args.nNodes - args.nByzs

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

    tmp_client = Client(  # for eval. the others' net / et al.
        args=args,
        net=_dense_net(),
        trainset=None,
        testset=None,
        log=False,
        _id=-1)

    clients = []
    for i in range(args.nNodes):
        if i < args.nByzs:  # Byzantine nodes
            client = Byzantine_Random(
                args=args,
                net=_dense_net(),
                trainset=splited_trainset[i],
                testset=splited_testset[i],
                log=True)
        else:  # Honest nodes
            client = Client(
                args=args,
                net=_dense_net(),
                trainset=splited_trainset[i],
                testset=splited_testset[i],
                log=True)
        client.set_weights(tmp_client.get_weights())  # same init. weights
        clients.append(client)

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
        # At least one honest node
        n_activated_byz = random.randint(0, args.nByzs)  # in Byz.
        n_activated_norm = random.randint(1, args.norm)  # in Norm.
        activateds = random.sample([t for t in range(args.nByzs)], n_activated_byz)
        activateds += random.sample([t + args.nByzs for t in range(args.norm)], n_activated_norm)

        current_nodes = []
        current_accs = []

        for a in tqdm(activateds):
            client = clients[a]

            if a < args.nByzs:  # Byzantine node
                pass  # skip averaging
            else:  # Normal node
                client.adjust_opt(epoch)

                """References
                # TBA
                """
                # My acc
                if client.acc is None:
                    my_acc = 100. - client.test(epoch, show=False, log=False)
                else:
                    my_acc = client.acc

                # The others' acc
                # TODO: parameterize
                # TODO: ETA
                tmp_client.set_dataset(trainset=None, testset=client.testset)

                if choicse == 'acc':
                    bests, idx_bests, _ = reputation.by_accuracy(
                        proposals=latest_nodes, count=min(len(latest_nodes), 2), test_client=tmp_client,
                        epoch=epoch, show=False, log=False,
                        timing=False, optimal_stopping=args.op_stop)
                elif choicse == 'Frobenius':
                    bests, idx_bests, _ = reputation.by_Frobenius(
                        proposals=latest_nodes, count=min(len(latest_nodes), 2), base_client=client, FN=args.filter,
                        return_acc=True, test_client=tmp_client, epoch=epoch, show=False, log=False,
                        timing=False, optimal_stopping=args.op_stop)
                elif choicse == 'random':
                    bests, idx_bests, _ = reputation.by_random(
                        proposals=latest_nodes, count=min(len(latest_nodes), 2),
                        return_acc=True, test_client=tmp_client, epoch=epoch, show=False, log=False,
                        timing=False)
                elif choicse == 'GNN':
                    pass  # TODO
                else:
                    raise()  # err

                best_nodes = [latest_nodes[idx_best] for idx_best in idx_bests]
                elected_nodes = []
                elected_repus = []

                # check self contain
                self_contain = (sum([b.creator == a for b in best_nodes]) != 0)

                # TODO: parameterize
                if (len(bests) < 2):  # 1
                    elected_nodes = [best_nodes[0], client]
                    elected_repus = [bests[0], my_acc]
                elif not self_contain:
                    if bests[1] > my_acc:
                        elected_nodes = [best_nodes[0], best_nodes[1]]
                        elected_repus = [bests[0], bests[1]]
                    else:
                        elected_nodes = [best_nodes[0], client]
                        elected_repus = [bests[0], my_acc]
                else:  # self-contained
                    # TODO: How to select the other honest node? (Mix)
                    # Current implementation:
                    # there exists the possibility of own + own (no change)
                    if bests[1] > my_acc:
                        elected_nodes = [best_nodes[0], best_nodes[1]]
                        elected_repus = [bests[0], bests[1]]
                    else:
                        elected_nodes = [best_nodes[0], client]
                        elected_repus = [bests[0], my_acc]

                """FL
                # own weights + the other's weights
                """
                weightses = [e.get_weights() for e in elected_nodes]
                repus_sum = sum(repus)
                repus = [e / repus_sum for e in elected_repus]

                client.set_average_weights(weightses, repus)

            # train
            client.train(epoch, show=False, log=True)

            # for logging
            after_avg_acc = 100. - client.test(epoch, show=False, log=True)
            current_accs.append(after_avg_acc)

            # save weights
            client.save()

            """DAG
            # TODO
            """
            # create node
            new_node = Node(
                weights=client.get_weights(),
                creator=a)
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
