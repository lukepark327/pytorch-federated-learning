"""
DAG (Directed Acyclic Graph)
"""


class Node:
    _id = 0

    def __init__(self,
                 weights,
                 #  parent: list = [],
                 #  edges: list = [],
                 _id=None):

        # id
        if _id != None:
            self._id = _id
        else:
            self._id = Node._id
            Node._id += 1

        # TODO: rounds

        self.weights = weights
        # self.parent = parent
        # self.edges = edges

    def get_id(self):
        return self._id

    def get_weights(self):
        return self.weights
