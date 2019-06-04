import arguments
from distribute import split_dataset
from node import Node
from model import Model
import policy

import tensorflow as tf
import numpy as np
import random
# from pprint import pprint


SEED = 950327
random.seed(SEED)
np.random.seed(SEED)


def locally_train(peers, epochs=2, verbose=0):
    # Training locally
    for i, peer in enumerate(peers):
        # print(">>> Locally train peer", i)
        peer.model.fit(peer.x_train, peer.y_train, epochs=epochs, verbose=verbose)
        peer.model.evaluate(peer.x_test, peer.y_test, verbose=verbose)


def locally_eval(peers, verbose=0):
    # Training locally
    for i, peer in enumerate(peers):
        # print(">>> Locally eval peer", i)
        peer.model.evaluate(peer.x_test, peer.y_test, verbose=verbose)


def get_preference(peers):
    res = list()
    for peer in peers:
        losses = [
            (peers[n].model.evaluate_by_other(peer.x_test, peer.y_test))[0]
            for n in peer.neighbors]
        res.append(
            policy.greedy(losses, len(peer.neighbors), reverse=False))  # return top peer's ID(index)
    return res


def updates(peers, top=3, comp=False):
    preferences = get_preference(peers)
    # pprint(preferences)

    for i, peer in enumerate(peers):
        preference = preferences[i][:top]

        peers_weights = [peers[p].model.weight for p in preference]
        peer_reputations = np.ones(len(preference))  # simple averaging

        if comp:
            # simple choice between local's own weights and averaging weights
            peer.update_weights_with_comparison(
                peers_weights, peer_reputations, peer.x_test, peer.y_test)
        else:
            peer.update_weights(peers_weights, peer_reputations)


def initial_weights():
    model = Model()
    return model.weight


def avg_and_sd(arr):
    nparr = np.array(arr)

    avg = np.average(nparr)
    sd = np.std(nparr)
    return avg, sd


if __name__ == "__main__":
    args = arguments.parser()
    n_nodes = args.nodes
    mecha_dist = args.dist
    mecha_bias = args.bias
    rounds = args.round
    top = args.top
    dataset_size = args.dataset

    size = 10

    # Load datasets
    mnist = tf.keras.datasets.mnist  # Load MNIST datasets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    print(">>> Done Loding Datasets")

    # Temp
    global_x_test = x_train[dataset_size:]
    global_y_test = y_train[dataset_size:]
    x_train = x_train[:dataset_size]
    y_train = y_train[:dataset_size]

    # for same initial weights
    w = initial_weights()

    # Big one
    cent = Model()
    cent.set_weights(w)
    cent.fit(x_train, y_train, epochs=5)
    cent.evaluate(global_x_test, global_y_test)
    print(">>> Done Training Big one NN")

    x_dist_train, y_dist_train = split_dataset(
        n_nodes, size,
        mecha_dist, mecha_bias,
        x_train, y_train,
        visual=False)
    x_dist_test, y_dist_test = split_dataset(
        n_nodes, size,
        mecha_dist, mecha_bias,
        x_test, y_test,
        visual=False)

    # Set peer nodes
    nodes = list()
    for i in range(n_nodes):
        node = Node()
        node.set_model(Model())  # set model
        node.set_data(
            x_dist_train[i], y_dist_train[i],
            x_dist_test[i], y_dist_test[i])  # set data
        node.neighbors = range(n_nodes)  # assume fully connected with myself
        node.model.set_weights(w)  # for same initial weights
        nodes.append(node)

    # Test
    for round in range(rounds):
        print(">>> Start round", round + 1, "/", rounds)
        # print(">>> acc before train", [node.model.acc for node in nodes])

        # acc of big one NN
        print(">>> acc of cent.", cent.acc)

        locally_train(nodes, epochs=5, verbose=0)  # local training

        # acc_before_gen is the accuracy before updating weights
        acc_before_gen = [node.model.acc for node in nodes]
        print(">>> acc before Gen.",
              avg_and_sd(acc_before_gen),
              acc_before_gen)

        # acc_before_gen_global is the accuracy tested with global testset before updating weights
        acc_before_gen_global = [node.model.evaluate_by_other(global_x_test, global_y_test)[1] for node in nodes]
        print(">>> acc before Gen. with Global",
              avg_and_sd(acc_before_gen_global),
              acc_before_gen_global)

        updates(nodes, top=top, comp=False)  # update; averaging weights
        locally_eval(nodes, verbose=0)  # local evalutation

        # acc_after_gen is the accuracy after updating weights
        acc_after_gen = [node.model.acc for node in nodes]
        print(">>> acc after Gen.",
              avg_and_sd(acc_after_gen),
              acc_after_gen)

        # acc_after_gen_global is the accuracy tested with global testset after updating weights
        acc_after_gen_global = [node.model.evaluate_by_other(global_x_test, global_y_test)[1] for node in nodes]
        print(">>> acc after Gen. with Global",
              avg_and_sd(acc_after_gen_global),
              acc_after_gen_global)

# nohup python main.py --dist "random" --bias "random" --nodes=100 --round=10000 --dataset=58000
