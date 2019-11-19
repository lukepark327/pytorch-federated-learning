# The following code is the implementation of
# blockchain-like shared database for simulation.
# Not real blockchain structures and functions.
from flmodel import Flmodel
from utils import MetaFunc, cal_weights_hash


class Reference:
    def __init__(self, id: str, metrics: list = list()):
        self.id = id
        self.metrics = metrics  # loss, acc, et al.


class Transaction:
    def __init__(
            self,
            id: str,
            owner: str,
            timestamp: int,
            prev_transactions: list = list(),
            prev_transactions_metrics: list = list(),
            flmodel: Flmodel = None,
            weight: int = 1):
        self.id = id
        self.owner = owner
        self.timestamp = timestamp
        self.flmodel = flmodel
        self.weight = weight  # weight using DAG's tx
        self.references = self.__set_references(
            prev_transactions, prev_transactions_metrics)

    def print(self):
        print("")
        print("id        :\t", self.id)
        print("owner     :\t", self.owner)
        print("timestamp :\t", self.timestamp)
        print("flmodel   :\t", cal_weights_hash(self.flmodel.get_model_weights()) if self.flmodel is not None else "None")
        print("weight    :\t", self.weight)
        print("references:\t", self.get_references_ids())

    def __set_references(self, prev_transactions: list, prev_transactions_metrics: list):
        if len(prev_transactions) != len(prev_transactions_metrics):
            raise ValueError

        res = list()
        for prev_tx_id, prev_tx_metrics in zip(prev_transactions, prev_transactions_metrics):
            res.append(Reference(prev_tx_id, prev_tx_metrics))
        return res

    def get_references_ids(self):
        return [r.id for r in self.references]


class Blockchain:
    def __init__(
            self,
            genesis_transaction: Transaction,
            policy_update_txs_weight_name: str = None,
            policy_update_txs_weight_func: callable = None,
            policy_snapshot_name: str = None,
            policy_snapshot_func: callable = None,
            *args):
        self.genesis_transaction = genesis_transaction
        self.__policy_update_txs_weight = MetaFunc(
            policy_update_txs_weight_name,
            policy_update_txs_weight_func
        )
        self.__policy_snapshot = MetaFunc(
            policy_snapshot_name,
            policy_snapshot_func
        )
        self.transactions = dict()
        self.add_transaction(genesis_transaction)

    def print(self):
        print("")
        print("tx_num    :\t", len(self.transactions.keys()))
        print("tx_weights:\t", {
              tx.id: tx.weight for tx in self.transactions.values()})
        print("policies  :\t", "update_txs_weight: ",
              self.get_policy_update_txs_weight_name())
        print("policies  :\t", "snapshot: ",
              self.get_policy_snapshot_name())

    """transaction"""

    def get_transaction_by_id(self, id: str):
        return self.transactions[id]

    def get_latest_transaction_by_owner(self, owner: str):
        txs = self.get_all_transactions_by_owner(owner)
        times = [tx.timestamp for tx in txs]
        return txs[times.index(max(times))]

    def get_all_transactions_by_owner(self, owner: str):
        res = list()
        for tx in self.transactions.values():
            if tx.owner == owner:
                res.append(tx)
        return res

    def get_all_predecessors_by_id(self, id: str):
        references_id = self.get_transaction_by_id(id).get_references_ids()

        predecessors = set()
        for r_id in references_id:
            predecessors.add(r_id)
            grands = self.get_all_predecessors_by_id(r_id)
            predecessors = predecessors.union(grands)
        return predecessors

    def add_transaction(self, tx: Transaction, *args):
        """invalid tx cases"""
        # already exist
        if tx.id in self.transactions:
            raise ValueError
        # invalid references
        predecessors_ids = tx.get_references_ids()
        for p_id in predecessors_ids:
            if p_id not in self.transactions:
                raise ValueError
            # invalid timestamp
            predecessor = self.get_transaction_by_id(p_id)
            if predecessor.timestamp > tx.timestamp:
                raise ValueError

        # add transaction in blockchain
        self.transactions[tx.id] = tx

        # update predecessors' weight
        self.update_txs_weight(self, tx.id, *args)

        # TODO: snapshot if needed
        # self.snapshot()

    """policies"""

    def update_txs_weight(self, id: str, *args):
        return self.__policy_update_txs_weight.func(id, *args)

    def snapshot(self, *args):
        return self.__policy_snapshot.func(*args)

    def get_policy_update_txs_weight_name(self):
        return self.__policy_update_txs_weight.name

    def get_policy_snapshot_name(self):
        return self.__policy_snapshot.name

    def set_policy_update_txs_weight(
            self,
            policy_update_txs_weight_name,
            policy_update_txs_weight_func):
        self.__policy_update_txs_weight = MetaFunc(
            policy_update_txs_weight_name,
            policy_update_txs_weight_func
        )

    def set_policy_snapshot(
            self,
            policy_snapshot_name,
            policy_snapshot_func):
        self.__policy_snapshot = MetaFunc(
            policy_snapshot_name,
            policy_snapshot_func
        )


if __name__ == "__main__":
    from time import time, sleep

    def my_policy_update_txs_weight(self: Blockchain, id: str):
        amount = self.get_transaction_by_id(id).weight
        predecessors_ids = self.get_all_predecessors_by_id(id)
        for p_id in predecessors_ids:
            p_tx = self.get_transaction_by_id(p_id)
            p_tx.weight += amount

    genesis_transaction = Transaction("0", "Alice", int(time()))
    blockchain = Blockchain(
        genesis_transaction,
        policy_update_txs_weight_name="heaviest",
        policy_update_txs_weight_func=my_policy_update_txs_weight)
    blockchain.print()

    sleep(1.5)

    tx_1 = Transaction("1", "Bob", int(time()), ["0"], [0.8])
    blockchain.add_transaction(tx_1)
    blockchain.print()
    [p.print() for p in blockchain.get_all_transactions_by_owner("Bob")]

    sleep(1.5)

    tx_2 = Transaction("2", "Bob", int(time()), ["0", "1"], [0.9, 0.7])
    blockchain.add_transaction(tx_2)
    blockchain.print()
    [p.print() for p in blockchain.get_all_transactions_by_owner("Bob")]

    sleep(1.5)

    tx_3 = Transaction("3", "Eve", int(time()), ["2"], [0.3])
    blockchain.add_transaction(tx_3)
    blockchain.print()
    blockchain.get_latest_transaction_by_owner("Bob").print()
