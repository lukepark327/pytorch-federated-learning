from model import FLModel
import numpy as np


class Node:
    def __init__(self, flmodel, train: tuple, test: tuple):
        self.neighbors = None
        self.reputations = None

        self.flmodel = flmodel
        self.x_train = train[0]
        self.y_train = train[1]
        self.x_test = test[0]
        self.y_test = test[1]

    def set_neighbors(self):
        pass

    def update_reputations(self):
        pass

    def set_model(self, flmodel):
        self.flmodel = flmodel

    def set_data(self, train: tuple, test: tuple):
        self.x_train = train[0]
        self.y_train = train[1]
        self.x_test = test[0]
        self.y_test = test[1]

    def update_weights(self):
        pass

    def raw_update_weights(self, peer_weights, peer_reputations):
        """
        :param array peer_weights: neighbors' weights
        :param array peer_reputations: my reputations about neighbors
        """
        total_reputations = sum(peer_reputations)
        num_peer = len(peer_reputations)

        new_weights = list()
        for layer in peer_weights[0]:
            new_weights.append(np.zeros(layer.shape))

        for i, layer in enumerate(new_weights):
            for j in range(num_peer):
                layer += peer_weights[j][i] * peer_reputations[j] / total_reputations

        self.flmodel.set_weights(new_weights)


def split_dataset(dataset, num):
    num_dataset = len(dataset)
    unit = int(num_dataset / num)

    res = list()
    for i in range(num):
        if i == 0:
            res.append(dataset[:unit])
        elif i == num - 1:
            res.append(dataset[i * unit:])
        else:
            res.append(dataset[i * unit:(i + 1) * unit])

    return res


if __name__ == "__main__":
    import tensorflow as tf

    num_nodes = 5

    # load data
    mnist = tf.keras.datasets.fashion_mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # split dataset
    my_x_train = split_dataset(x_train, num_nodes)
    my_y_train = split_dataset(y_train, num_nodes)
    my_x_test = split_dataset(x_test, num_nodes)
    my_y_test = split_dataset(y_test, num_nodes)

    # set model
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

    # set nodes
    nodes = list()
    for i in range(num_nodes):
        nodes.append(
            Node(flmodel, (my_x_train[i], my_y_train[i]), (my_x_test[i], my_y_test[i])))
