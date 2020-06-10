import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import os.path
import os
import random
from copy import deepcopy

from dag import Node
from net import Net
from client import Client
import arguments


if __name__ == "__main__":

    """arguments"""
    args = arguments.parser()

    ns = args.nodes
    rs = args.rounds

    print("> Setting:", args)

    """Preprocess"""
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)

    """random split
    TODO: various distribution methods
    """
    splited_trainset = torch.utils.data.random_split(trainset, [int(len(trainset) / ns) for _ in range(ns)])
    splited_testset = torch.utils.data.random_split(testset, [int(len(testset) / ns) for _ in range(ns)])

    # print(len(splited_trainset[0]), len(splited_trainset[1]), len(splited_trainset[2]))

    """set nodes"""
    clients = []
    for i in range(ns):
        clients.append(Client(
            trainset=splited_trainset[i],
            testset=splited_testset[i],
            net=Net(),
            _id=i))

    tmp_client = Client(  # tmp
        trainset=None,
        testset=splited_testset[i],
        net=Net(),
        _id=-1)

    """set DAG"""
    global_id = -1
    nodes = []

    genesis = Node(
        r=-1,
        w=clients[0].get_weights(),
        _id=global_id)
    nodes.append(genesis)
    global_id += 1

    """run simulator"""
    latest_nodes = deepcopy(nodes)  # in DAG

    for r in range(rs):
        print(">>> Round %5d" % (r))

        # select activated clients
        n_activated = random.randint(1, ns)  # TODO: range
        activateds = random.sample([t for t in range(ns)], n_activated)

        current_nodes = []
        current_accs = []

        for a in activateds:
            client = clients[a]

            """reference"""
            if len(latest_nodes) < 2:  # never be 0
                parent = latest_nodes + latest_nodes  # TODO: parameterize
                repus = [0.5, 0.5]
            else:
                # cal. edges
                accs = []
                for l in latest_nodes:
                    tmp_client.set_dataset(client.trainset, client.testset)
                    tmp_client.set_weights(l.get_weights())
                    accs.append(tmp_client.eval(r=-1))

                sorted_accs = sorted(accs)[:2]
                idx_first, idx_second = accs.index(sorted_accs[0]), accs.index(sorted_accs[1])

                parent = [latest_nodes[idx_first], latest_nodes[idx_second]]  # TODO: numpy
                repus = [sorted_accs[0] / sum(sorted_accs), sorted_accs[1] / sum(sorted_accs)]

            """train"""
            client.set_average_weights(
                [parent[0].get_weights(), parent[1].get_weights()],
                repus)

            client.train(r=r, epochs=1, log_flag=True)
            current_accs.append(client.eval(r=r, log_flag=True))

            """create Node"""
            new_node = Node(
                r=r,
                w=client.get_weights(),
                _id=global_id,
                parent=parent,
                edges=repus)  # TODO: sorted_accs
            nodes.append(new_node)
            global_id += 1

            current_nodes.append(new_node)

        """log"""
        print(">>> activated_nodes:", activateds)
        print(">>> latest_nodes:", [d.get_id() for d in latest_nodes])
        print(">>> current_nodes:", [d.get_id() for d in current_nodes])
        print(">>> current_accs:", current_accs)

        latest_nodes = deepcopy(current_nodes)
        

