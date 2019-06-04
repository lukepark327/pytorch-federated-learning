import numpy as np

import mechanisms
import visualization


def split_dataset(nodes, size, mecha_dist, mecha_bias, x_train, y_train, visual=False):
    """
    Return splited dataset

    :param int node: The number of nodes
    :param int size: The number of data label
    :param string mecha_dist: Describe mechanism among nodes
    :param string mecha_bias: Describe mechanism among one's data
    :param np.array x_train: Target data to split (problems)
    :param np.array y_train: Target data to split (solutions)
    :param boolean visual: Set True if you want visualization (default: False)
    :return: tuple of list of splited dataset (x_dist, y_dist)
    """
    dist = np.ones((nodes, size))

    # distribution among nodes
    dist_nodes = mechanisms.select(mecha_dist, nodes)

    # distribution among one's data
    for i in range(nodes):
        dist_data = mechanisms.select(mecha_bias, size)
        dist[i] = dist_nodes[i] * dist_data

    # print(sum(dist))
    # print(sum(sum(dist)))

    # visualization
    if visual:
        visualization.heatmap(dist)

    x_num = [int(i) for i in dist_nodes * len(x_train)]
    y_num = [int(i) for i in dist_nodes * len(y_train)]

    # for residues
    x_num[-1] += len(x_train) - sum(x_num)
    y_num[-1] += len(y_train) - sum(y_num)

    x_dist = list()
    front = back = 0
    for i in range(nodes):
        back += int(x_num[i])
        x_dist.append(x_train[front:back])
        front = back

    y_dist = list()
    front = back = 0
    for i in range(nodes):
        back += int(y_num[i])
        y_dist.append(y_train[front:back])
        front = back

    return x_dist, y_dist
