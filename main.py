from model import FLModel
from node import Node, split_dataset
from block import Blockchain, Block, printBlock
import arguments

import tensorflow as tf
import numpy as np


# python main.py --nodes=5 --round=1000 --globalSet=10000
if __name__ == "__main__":
    # parsing hyperparameters
    args = arguments.parser()
    num_nodes = args.nodes
    num_round = args.round
    num_global_testset = args.globalSet
    print(">>> Setting:", args)

    # Set Tensorflow GPU
    # tf.device('/device:GPU:0')

    # Load datasets
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # get global testset by train
    global_x_test = x_train[:num_global_testset]
    global_y_test = y_train[:num_global_testset]
    x_train = x_train[num_global_testset:]
    y_train = y_train[num_global_testset:]

    # set FL model
    mnist_model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    mnist_model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    flmodel = FLModel(mnist_model)

    """set blockchain"""
    init_weights = flmodel.get_weights()
    genesis = Block(
        0,
        "0" * 64,
        init_weights,
        (global_x_test, global_y_test)
    )
    flchain = Blockchain(genesis)  # set blockchain with genesis block

    """set nodes"""
    # split dataset
    my_x_train = split_dataset(x_train, num_nodes)
    my_y_train = split_dataset(y_train, num_nodes)
    my_x_test = split_dataset(x_test, num_nodes)
    my_y_test = split_dataset(y_test, num_nodes)

    # set nodes
    nodes = list()
    for i in range(num_nodes):
        nodes.append(
            Node(flmodel, (my_x_train[i], my_y_train[i]), (my_x_test[i], my_y_test[i])))

    # set Leader (Primary)  # pBTF
    Leader = Node(flmodel, (None, None), (global_x_test, global_y_test))

    """main"""
    for nextBlockNumber in range(1, num_round + 1):
        currentBlockNumber = nextBlockNumber - 1
        currentBlockWeight = flchain.blocks[currentBlockNumber].weights

        # update weights
        peer_weights = list()
        # TODO: set a different reputation per nodes
        peer_reputations = np.ones(num_nodes)
        for i, node in enumerate(nodes):
            node.flmodel.set_weights(currentBlockWeight)  # set weights
            node.flmodel.fit(node.x_train, node.y_train, epochs=1)  # training
            peer_weight = node.flmodel.get_weights()
            peer_weights.append(peer_weight)

            # eval. each node
            node.flmodel.evaluate(node.x_test, node.y_test)
            print("> node: %-5d" % i, end="\t")
            print("loss: %-8.4f" % node.flmodel.loss, end="\t")
            print("acc: %-8.4f" % node.flmodel.acc, end="\r")
            # time.sleep(0.3)

        Leader.raw_update_weights(peer_weights, peer_reputations)
        nextBlockWeight = Leader.flmodel.get_weights()

        # create next block
        new_block = Block(
            nextBlockNumber,
            flchain.getBlock(nextBlockNumber - 1).calBlockHash(),
            nextBlockWeight,
            (Leader.x_test, Leader.y_test)
        )
        flchain.append(new_block)  # append next block

        Leader.flmodel.evaluate(Leader.x_test, Leader.y_test)  # eval.

        # print
        print(" " * 64, end="\r")
        print("round: %-6d" % nextBlockNumber, end="\t")
        print("loss: %-8.4f" % Leader.flmodel.loss, end="\t")
        print("acc: %-8.4f" % Leader.flmodel.acc)
        printBlock(flchain.blocks[-1])
