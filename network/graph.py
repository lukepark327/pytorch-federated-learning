from copy import deepcopy

from .transaction import Transaction, TxTypeEnum
from ml.flmodel import FLModel

class TxGraph:
    """
    TxGraph represents the DAG(Directed Acyclic Graph) structure consists of Transaction instances
    """
    def __init__(self, genesis_tx=None, eval_set=tuple()):
        if genesis_tx is not None:
            self.genesis_tx = genesis_tx
        else:
            self.genesis_tx = self.generate_genesis()
        self.transactions = dict()
        self.add_transaction(self.genesis_tx)

        self.x_eval, self.y_eval = eval_set
        self.missing_transactions = set()
        self.model_evaluated_results = dict()

    @property
    def eval_set(self):
        return self.x_eval, self.y_eval

    @eval_set.setter
    def eval_set(self, eval_set):
        self.x_eval, self.y_eval = eval_set

    # Transaction related methods
    def generate_genesis(self):
        genesis = Transaction(
            tx_type=TxTypeEnum.NONE, 
            task_id='',
            model_id='', 
            timestamp=0, 
            refs=[])
        self.genesis_tx = genesis
    
    def has_transaction(self, tx):
        return tx.txid in self.transactions.keys()

    def add_transaction(self, tx):
        self.transactions[tx.txid] = tx
    
    def add_missing_transaction(self, txid):
        self.missing_transactions.add(txid)

    def evaluate_and_record_model(self, model: FLModel):
        if model.model_id in self.model_evaluated_results.keys():
            return
        self.model_evaluated_results[model.model_id] = model.evaluate(self.x_eval, self.y_eval)

    def get_evaluation_result(self, model_id):
        return self.model_evaluated_results[model_id]

    def get_transaction_by_id(self, txid):
        if txid in self.transactions.keys():
            return self.transactions[txid]
        else:
            return None

    def get_transaction_by_model_id(self, model_id):
        for tx in self.transactions.values():
            if tx.model_id == model_id:
                return tx
        return None

    def get_transaction_list_by_ids(self, txids):
        return [ self.get_transaction_by_id(txid) for txid in txids if txid is not None ]

    def get_all_predecessors_by_id(self, txid):
        tx = self.get_transaction_by_id(txid)
        predecessors = set()
        if tx is not None:
            for r_id in tx.ref_ids:
                predecessors.add(r_id)
                grands = self.get_all_predecessors_by_id(r_id)
                predecessors = predecessors.union(grands)
        else:
            self.add_missing_transaction(txid)
        return predecessors

    # TODO: do not use timestamp, make this function use tip selection algorithm
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
        
    # Policy related methods
    @property
    def tx_referenced_table(self):
        ref_table = dict()
        for tx in self.transactions:
            for r_id in tx.ref_ids:
                if r_id in ref_table:
                    ref_table[r_id] += 1
                else:
                    ref_table[r_id] = 1
        return ref_table

    @property
    def owner_referenced_table(self):
        ref_table = dict()
        for txid, count in self.tx_referenced_table:
            tx = self.get_transaction_by_id(txid)
            if tx is not None:
                if tx.owner in ref_table:
                    ref_table[tx.owner] += count
                else:
                    ref_table[tx.owner] = count
        return ref_table

    def __str__(self):
        return (
            "number of transactions: " + str(len(self.transactions.keys())) + \
            # "missing transactions: " + str(self.missing_transactions) + \
            "\nmodel evaluated: " + str(self.model_evaluated_results) 
        )
    