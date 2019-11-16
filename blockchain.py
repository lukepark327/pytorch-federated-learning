# The following code is the implementation of
# blockchain-like shared database for simulation.
# Not real blockchain structures and functions.
from flmodel import Flmodel


# DAG
class Reference:
    def __init__(self, transaction, metrics: list = list()):
        self.transaction = transaction
        self.metrics = metrics  # loss, acc, et al.


class Transaction:
    def __init__(
            self,
            id: str,
            prev_transactions: list = list(),
            prev_transactions_metrics: list = list(),
            flmodel: Flmodel = None):
        self.id = id
        self.flmodel = flmodel
        self.weights = 0  # weight using DAG's tx
        self.references = self.__set_references(
            prev_transactions, prev_transactions_metrics)
        self.__update_predecessors_weights()

    def __set_references(self, prev_transactions: list, prev_transactions_metrics: list):
        if len(prev_transactions) != len(prev_transactions_metrics):
            raise ValueError

        res = list()
        for prev_tx, prev_tx_metrics in zip(prev_transactions, prev_transactions_metrics):
            res.append(Reference(prev_tx, prev_tx_metrics))
        return res

    def get_all_predecessors(self):
        predecessors = set()
        for refer in self.references:
            predecessors.add(refer.transaction)
            grands = refer.transaction.get_all_predecessors()
            predecessors.union(grands)
        return predecessors

    # TODO: policy
    def __update_predecessors_weights(self):
        predecessors = self.get_all_predecessors()
        for predecessor in predecessors:
            predecessor.weights += 1


class Blockchain:
    def __init__(

    ):
        pass


if __name__ == "__main__":
    # Genesis Tx
    tx_0 = Transaction(str(0))
    print("0: ", tx_0.weights, [t.id for t in tx_0.get_all_predecessors()])
    print("=" * 8)

    # Tx_1
    tx_1 = Transaction(str(1), [tx_0], [0.4])
    print("0: ", tx_0.weights, [t.id for t in tx_0.get_all_predecessors()])
    print("1: ", tx_1.weights, [t.id for t in tx_1.get_all_predecessors()])
    print("=" * 8)

    # Tx_2
    tx_2 = Transaction(str(2), [tx_0, tx_1], [0.2, 0.0])
    print("0: ", tx_0.weights, [t.id for t in tx_0.get_all_predecessors()])
    print("1: ", tx_1.weights, [t.id for t in tx_1.get_all_predecessors()])
    print("2: ", tx_2.weights, [t.id for t in tx_2.get_all_predecessors()])
    print("=" * 8)

    # Tx_3
    tx_3 = Transaction(str(3), [tx_1, tx_2], [0.8, 0.1])
    print("0: ", tx_0.weights, [t.id for t in tx_0.get_all_predecessors()])
    print("1: ", tx_1.weights, [t.id for t in tx_1.get_all_predecessors()])
    print("2: ", tx_2.weights, [t.id for t in tx_2.get_all_predecessors()])
    print("3: ", tx_3.weights, [t.id for t in tx_3.get_all_predecessors()])
    print("=" * 8)
