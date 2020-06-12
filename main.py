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
import pickle
from copy import deepcopy

from dag import Node
from net import Net
from client import Client
import arguments
from plot import draw


if __name__ == "__main__":

    os.system("rm -rf clients")

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
    global_id = 0
    nodes = []

    genesis = Node(
        r=0,
        w=clients[0].get_weights(),
        _id=global_id)
    nodes.append(genesis)
    global_id += 1

    """list for DAG graph"""
    f = []
    t = []

    """run simulator"""
    # tip_nodes = deepcopy(nodes)  # in DAG
    tip_nodes = [0]

    for r in range(1, rs + 1):
        print(">>> Round %5d" % (r))

        # select activated clients
        n_activated = random.randint(1, ns)  # TODO: range
        activateds = random.sample([t for t in range(ns)], n_activated)

        current_nodes = []
        current_accs = []
        old_parents = set()

        for a in activateds:
            client = clients[a]

            """reference"""
            #print("tip_nodes: ", tip_nodes)
            if len(tip_nodes) < 2:  # never be 0
                parents = [nodes[tip_nodes[0]], nodes[tip_nodes[0]]]  # TODO: parameterize
                repus = [0.5, 0.5]
                old_parents.add(tip_nodes[0])
                f.append(tip_nodes[0])
                t.append(global_id)
            else:
                # cal. edges
                accs = []
                for l in tip_nodes:
                    tip = nodes[l]
                    tmp_client.set_dataset(client.trainset, client.testset)
                    tmp_client.set_weights(tip.get_weights())
                    accs.append([tip.get_id(), tmp_client.eval(r=-1)])

                sorted_accs = sorted(accs, key=lambda l:l[1], reverse=True)
                #print("sorted_accs: ", sorted_accs)
                tip1, tip2 = sorted_accs[0][0], sorted_accs[1][0]
                #print("tips: ", tip1, tip2)

                parents = [nodes[tip1], nodes[tip2]]  # TODO: numpy
                acc_sum = sorted_accs[0][1] + sorted_accs[1][1]
                repus = [sorted_accs[0][1] / acc_sum, sorted_accs[1][1] / acc_sum]
                parent_ids = [p.get_id() for p in parents]
                old_parents.update(parent_ids)
                f.extend(parent_ids)
                t.extend([global_id, global_id])

            """train"""
            client.set_average_weights(
                [parents[0].get_weights(), parents[1].get_weights()],
                repus)

            client.train(r=r, epochs=1, log_flag=True)
            current_accs.append(client.eval(r=r, log_flag=True))

            """create Node"""
            new_node = Node(
                r=r,
                w=client.get_weights(),
                _id=global_id,
                parent=parents,
                edges=repus)  # TODO: sorted_accs
            nodes.append(new_node)

            current_nodes.append(global_id)
            tip_nodes.append(global_id)

            global_id += 1

        for p in old_parents:
            #print(p)
            try:
                tip_nodes.remove(p)
            except ValueError:
                pass  # do nothing!
        tip_nodes.extend(current_nodes)
        tip_nodes = list(set(tip_nodes))

        # write from / to
        with open('dag.data', 'wb') as filehandle:
            # store the data as binary data stream
            pickle.dump({"f": f, "t": t}, filehandle)

        """log"""
        print(">>> activated_clients:", activateds)
        print(">>> current_nodes:", [d for d in current_nodes])
        print(">>> current_accs:", current_accs)
        print(">>> tip_nodes:", [t for t in tip_nodes])

        # tip_nodes = deepcopy(current_nodes)

    # draw(f, t)
