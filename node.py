from model import Model

import numpy as np


class Node:
    def __init__(self):
        self.model = None

        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

        self.neighbors = None

    def set_model(self, model):
        self.model = model

    def set_data(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def update_weights(self, peer_weights, peer_reputations):
        """
        :param array peer_weights: neighbors' weights
        :param array peer_reputations: my reputations about neighbors
        """
        new_weights = [np.zeros(w.shape) for w in self.model.weight]
        total_size = np.sum(peer_reputations)

        for p in range(len(peer_weights)):
            for i in range(len(new_weights)):
                new_weights[i] += peer_weights[p][i] * peer_reputations[p] / total_size
        self.model.set_weights(new_weights)
    
    def update_weights_with_comparison(self, peer_weights, peer_reputations, x_test, y_test):
        """
        :param array peer_weights: neighbors' weights
        :param array peer_reputations: my reputations about neighbors
        """
        new_model = Model()
        new_weights = [np.zeros(w.shape) for w in self.model.weight]
        total_size = np.sum(peer_reputations)

        for p in range(len(peer_weights)):
            for i in range(len(new_weights)):
                new_weights[i] += peer_weights[p][i] * peer_reputations[p] / total_size
        
        new_model.set_weights(new_weights)

        # comparison
        my_acc = self.model.evaluate_by_other(x_test, y_test)[1]  # acc
        new_acc = new_model.evaluate_by_other(x_test, y_test)[1]  # acc
        if my_acc < new_acc:
            print(">>> ! Select new weights")
            self.model.set_weights(new_weights)


if __name__ == "__main__":
    pass
