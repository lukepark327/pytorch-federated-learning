from flmodel import Flmodel
from utils import MetaFunc
from utils import sha3_256


class Node:
    def __init__(
            self,
            id: str,  # unique ID (Address)
            flmodel: Flmodel = None,  # NN model
            x_train=None, y_train=None,
            x_test=None, y_test=None,
            reputation: dict = dict(),
            policy_update_reputation_name: str = None,
            policy_update_reputation_func: callable = None,
            policy_update_model_weights_name: str = None,
            policy_update_model_weights_func: callable = None,
            policy_replace_model_name: str = None,
            policy_replace_model_func: callable = None):
        self.id = id
        self.__flmodel = flmodel
        self.__x_train, self.__y_train = x_train, y_train
        self.__x_test, self.__y_test = x_test, y_test
        self.__reputation = reputation
        self.__policy_update_reputation = MetaFunc(
            policy_update_reputation_name,
            policy_update_reputation_func
        )
        self.__policy_update_model_weights = MetaFunc(
            policy_update_model_weights_name,
            policy_update_model_weights_func
        )
        self.__policy_replace_model = MetaFunc(
            policy_replace_model_name,
            policy_replace_model_func
        )

    def print(self):
        def cal_weights_hash(weights: list):
            weights_list = list()
            for weight in weights:
                b = weight.tobytes()
                weights_list.append(b)
            return sha3_256(weights_list)

        print("")
        print("id        :\t", self.id)
        print("weight    :\t", cal_weights_hash(self.get_model_weights()))
        print("train     :\t", self.get_train_size())
        print("test      :\t", self.get_test_size())
        print("reputation:\t", self.get_reputation())
        print("policies  :\t", "update_reputation: ",
              self.get_policy_update_reputation_name())
        print("policies  :\t", "update_model_weights: ",
              self.get_policy_update_model_weights_name())
        print("policies  :\t", "replace_model: ",
              self.get_policy_replace_model_name())

    """reputation"""

    def set_reputation(self, reputation: dict):
        self.__reputation = reputation  # (id: str => amount: float)

    def get_reputation(self):
        return self.__reputation

    """data"""

    def set_train_data(self, x_train, y_train):
        self.__x_train, self.__y_train = x_train, y_train

    def set_test_data(self, x_test, y_test):
        self.__x_test, self.__y_test = x_test, y_test

    def get_train_data(self):
        return self.__x_train, self.__y_train

    def get_test_data(self):
        return self.__x_test, self.__y_test

    def get_train_size(self):
        return len(self.__x_train) if self.__x_train is not None else 0

    def get_test_size(self):
        return len(self.__x_test) if self.__x_test is not None else 0

    """Flmodel"""

    def set_model(self, flmodel: Flmodel):
        self.__flmodel = flmodel

    def get_model(self):
        return self.__flmodel

    def fit_model(self, epochs=1, callbacks=[], verbose=0):
        self.__flmodel.fit(
            self.__x_train, self.__y_train,
            epochs=epochs, callbacks=callbacks, verbose=verbose)

    def evaluate_model(self, verbose=0):
        return self.__flmodel.evaluate(
            self.__x_test, self.__y_test, verbose=verbose)

    def get_model_metrics(self):
        return self.__flmodel.get_metrics()

    def get_model_weights(self):
        return self.__flmodel.get_weights()

    def set_model_weights(self, new_weights):
        self.__flmodel.set_weights(new_weights)

    def predict_model(self, x_input):
        return self.__flmodel.predict(x_input)

    """policies"""

    def update_reputation(self, *args):
        return self.__policy_update_reputation.func(*args)

    def update_model_weights(self, *args):
        return self.__policy_update_model_weights.func(*args)

    def replace_model(self, *args):
        return self.__policy_replace_model.func(*args)

    def get_policy_update_reputation_name(self):
        return self.__policy_update_reputation.name

    def get_policy_update_model_weights_name(self):
        return self.__policy_update_model_weights.name

    def get_policy_replace_model_name(self):
        return self.__policy_replace_model.name

    def set_policy_update_reputation(
            self,
            policy_update_reputation_name,
            policy_update_reputation_func):
        self.__policy_update_reputation = MetaFunc(
            policy_update_reputation_name,
            policy_update_reputation_func
        )

    def set_policy_update_model_weights(
            self,
            policy_update_model_weights_name,
            policy_update_model_weights_func):
        self.__policy_update_model_weights = MetaFunc(
            policy_update_model_weights_name,
            policy_update_model_weights_func
        )

    def set_policy_replace_model(
            self,
            policy_replace_model_name,
            policy_replace_model_func):
        self.__policy_replace_model = MetaFunc(
            policy_replace_model_name,
            policy_replace_model_func
        )


if __name__ == "__main__":
    import tensorflow as tf
    from utils import split_dataset
    import numpy as np
    import arguments
    import time

    def create_model():
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
        return Flmodel(mnist_model)

    def my_policy_update_model_weights(self: Node, peer_weights: dict):
        # get reputation
        reputation = self.get_reputation()
        if len(reputation) == 0:
            raise ValueError
        if len(reputation) != len(peer_weights):
            raise ValueError

        ids = list(reputation.keys())
        total_reputation = sum(reputation.values())

        # set zero-filled NN layers
        new_weights = list()
        for layer in peer_weights[ids[0]]:
            new_weights.append(np.zeros(layer.shape))

        # calculate new_weights
        for i, layer in enumerate(new_weights):
            for id in ids:
                layer += peer_weights[id][i] * \
                    reputation[id] / total_reputation

        # set new_weights
        self.set_model_weights(new_weights)

    def equally_fully_connected(my_id: str, ids: list):
        reputation = dict()
        for id in ids:
            if id == my_id:
                continue
            reputation[id] = 1.
        return reputation

    def avg_time(times):
        if len(times) == 0:
            return 0.0
        else:
            return sum(times) / len(times)

    """main"""

    # arguments
    args = arguments.parser()
    num_nodes = args.nodes
    num_round = args.round
    print("> Setting:", args)

    # load data
    mnist = tf.keras.datasets.fashion_mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # split dataset
    my_x_train = np.array_split(x_train, num_nodes)
    my_y_train = np.array_split(y_train, num_nodes)
    my_x_test = np.array_split(x_test, num_nodes)
    my_y_test = np.array_split(y_test, num_nodes)

    # set nodes
    ids = [str(i) for i in range(num_nodes)]
    nodes = list()
    for i, id in enumerate(ids):
        nodes.append(Node(
            id,
            create_model(),
            my_x_train[i], my_y_train[i],
            my_x_test[i], my_y_test[i],
            equally_fully_connected(id, ids),
            policy_update_model_weights_name="equal",
            policy_update_model_weights_func=my_policy_update_model_weights))

    # round
    elapsed_times = list()
    for r in range(num_round):
        start_time = time.time()

        print("Round", r)

        # train
        for node in nodes:
            node.fit_model(epochs=1)
            print("train :\tnode: %5s" % (node.id), end="\r")
        print(" " * 73, end="\r")

        # test
        losses = list()
        accuracies = list()
        for node in nodes:
            metrics = node.evaluate_model()
            # print("test  :\t", node.id, loss, metrics)
            losses.append(metrics[0])
            accuracies.append(metrics[1])
            print("own:\tnode: %5s\tloss: %7.4f\tacc : %7.4f," % (
                node.id, metrics[0], metrics[1]), end="\r")

        print(" " * 73, end="\r")
        print("own:\tmax_loss: %7.4f\tmin_loss: %7.4f\tavg_loss: %7.4f" % (
            max(losses), min(losses), sum(losses) / len(losses)))
        print("own:\tmax_acc : %7.4f\tmin_acc : %7.4f\tavg_acc : %7.4f" % (
            max(accuracies), min(accuracies), sum(accuracies) / len(accuracies)))

        # TODO: comparison

        # update weights
        for node in nodes:
            # get neighbors weights
            peer_weights = dict()
            for peer in nodes:
                if peer.id == node.id:
                    continue
                peer_weights[peer.id] = peer.get_model_weights()

            # TODO: duplicated weight update problem on the earliest nodes.
            node.update_model_weights(node, peer_weights)
            print("update:\tnode: %5s" % (node.id), end="\r")
        print(" " * 73, end="\r")

        # test
        losses = list()
        accuracies = list()
        for node in nodes:
            metrics = node.evaluate_model()
            # print("update:\t", node.id, loss, metrics)
            losses.append(metrics[0])
            accuracies.append(metrics[1])
            print("mix:\tnode: %5s\tloss: %7.4f\tacc : %7.4f," % (
                node.id, metrics[0], metrics[1]), end="\r")

        print(" " * 73, end="\r")
        print("mix:\tmax_loss: %7.4f\tmin_loss: %7.4f\tavg_loss: %7.4f" % (
            max(losses), min(losses), sum(losses) / len(losses)))
        print("mix:\tmax_acc : %7.4f\tmin_acc : %7.4f\tavg_acc : %7.4f" % (
            max(accuracies), min(accuracies), sum(accuracies) / len(accuracies)))

        end_time = time.time()
        elapsed_time = end_time - start_time
        elapsed_times.append(elapsed_time)
        print("elapsed time: %f\tETA: %f" %
              (elapsed_time, avg_time(elapsed_times)))
