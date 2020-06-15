"""
DAG (Directed Acyclic Graph)
"""


class Node:
    def __init__(self,
                 r,  # TODO: case None
                 w,  # TODO: gradient
                 _id=None,  # TODO: case None
                 parent: list = [],
                 edges: list = []):

        assert(_id != None)
        self._id = _id

        self.r = r
        self.w = w

        self.parent = parent
        self.edges = edges

    def get_id(self):
        return self._id

    def get_weights(self):
        return self.w
