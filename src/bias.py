import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import random
import collections


from visual import heatmap


# Current implementation is nothing but `80:20` .
# TODO: using np.random.pareto()
def pareto(labels, target, size=1):
    res = []
    for _ in range(size):
        if random.random() < 0.8:
            res.append(target)
        else:
            res.append(np.random.choice(np.delete(labels, target)))
    return res


if __name__ == "__main__":

    """Preprocess"""
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    """biased split
    # TODO: distriting each dataset size biased
    # collections.Counter(pareto(np.arange(ns), 0, size=len(trainset)))
    """
    ns = 5  # number of clients
    cs = len(classes)

    # trainset
    targets = trainset.targets
    targets_collect = collections.Counter(targets)
    targets_dist = [targets_collect[j] for j in range(cs)]
    print(targets_dist)

    dist_map = []
    for i in range(cs):
        x = collections.Counter(
            pareto(np.arange(ns), i % cs, size=int(len(trainset) / ns)))
        dist_each = [x[j] for j in range(cs)]
        dist_map.append(dist_each)

    dist_map = np.array(dist_map)
    # heatmap(dist_map,
    #         log=True, annot=False,
    #         xlabel="classes", ylabel="nodes", title="dataset dist. (log)",
    #         save=False, show=False)

    print(dist_map)
    naive_dist = [dist_map[:, i].sum() for i in range(cs)]
    refined_dist = [naive_dist]

    # """random split
    # TODO: various distribution methods
    # """
    # splited_trainset = torch.utils.data.random_split(trainset, [int(len(trainset) / ns) for _ in range(ns)])
    # splited_testset = torch.utils.data.random_split(testset, [int(len(testset) / ns) for _ in range(ns)])

    # # print(len(splited_trainset[0]), len(splited_trainset[1]), len(splited_trainset[2]))
