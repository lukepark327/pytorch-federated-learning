from random import random

from .graph import TxGraph
from .transaction import Transaction, TxTypeEnum, Reference
from ml.task import Task, compile_model
from policy.selection import Selection
from policy.updating import Updating
from policy.comparison import Comparison

class Node:
    def __init__(
        self, 
        nid, global_time, task_id,
        global_model_table, train_set, test_set, eval_rate,
        tx_graph: TxGraph, 
        selection: Selection, updating: Updating, comparison: Comparison,
        adjacent_list=list(), model_id=None
        ):
        self.nid = nid
        self.adjacent_list = adjacent_list
        self.time = global_time
        self.task_id = task_id
        self.model_table = global_model_table
        self.model_id = model_id
        self.__x_train, self.__y_train = train_set
        self.__x_test, self.__y_test = test_set
        self.eval_rate = eval_rate
        self.tx_graph = tx_graph
        self.selection = selection
        self.updating = updating
        self.comparison = comparison
        self.__test_cache = dict()
        self.__tx_sending_buffer = list()
        self.__tx_receiving_buffer = list()
        
    def will_get_transaction(self, tx: Transaction):
        self.__tx_receiving_buffer.append(tx)

    def will_send_transaction(self, tx: Transaction):
        self.__tx_sending_buffer.append(tx)

    def get_transaction(self, tx: Transaction):
        if self.tx_graph.has_transaction(tx) is True:
            return
        self.tx_graph.add_transaction(tx)
        self.will_send_transaction(tx)
        if random() < self.eval_rate:
            self.tx_graph.evaluate_and_record_model(
                model_id=tx.model_id,
                model=self.model_table[tx.model_id]
            )

    def get_transactions_from_buffer(self):
        for received_tx in self.__tx_receiving_buffer:
            self.get_transaction(received_tx)
        self.__tx_receiving_buffer = list()
    
    def make_new_transaction(self, tx_type, task_id, refs):
        tx = Transaction(tx_type, task_id, self.nid, self.time, refs)
        self.tx_graph.add_transaction(tx)
        return tx

    def send_new_transaction(self, tx):
        for node in self.adjacent_list:
            node.will_get_transaction(tx)
    
    def send_txs_in_buffer(self):
        for tx in self.__tx_sending_buffer:
            for node in self.adjacent_list:
                node.will_get_transaction(tx)
        self.__tx_sending_buffer = list()

    def select_transactions(self):
        self.selection.update(self.tx_graph)
        return self.selection.select(self.tx_graph)

    @property
    def current_model(self):
        if self.model_id == None:
            return None
        return self.model_table[self.model_id]
    
    @property
    def data_set(self):
        return (
            self.__x_train, self.__y_train, 
            self.__x_test, self.__y_test, 
            self.tx_graph.eval_set
        )
    
    @data_set.setter
    def data_set(self, train_set, test_set, eval_set):
        self.__x_train, self.__y_train = train_set
        self.__x_test, self.__y_test = test_set
        self.tx_graph.eval_set = eval_set
        # Refresh test cache
        self.__test_cache = dict()

    def test_and_cache(self, model_id, model):
        if model_id in self.__test_cache.keys():
            return self.__test_cache[model_id]
        res = model.evaluate(self.__x_test, self.__y_test)
        self.__test_cache[model_id] = res
        return res

    def open_task(self, task: Task):
        tx = self.make_new_transaction(
            TxTypeEnum.OPEN, 
            task.task_id, 
            [ Reference(self.tx_graph.genesis_tx.txid, [None, None]) ]
        )
        self.model_table[tx.model_id] = task.task_model
        self.send_new_transaction(tx)

    def init_local_train(self, task: Task):
        basic_model = task.create_base_model()
        basic_model.fit(self.__x_train, self.__y_train)
        self.model_id = 'local' + self.nid
        self.model_table[self.model_id] = basic_model
        self.test_and_cache(self.model_id, basic_model)
    
    def update(self, task: Task):
        selected_txs = self.select_transactions()
        if len(selected_txs) is 0:
            return
        selected_models = [ self.model_table[tx.model_id] for tx in selected_txs ]
        
        new_model = self.updating.update(selected_models, task)

        new_eval = new_model.evaluate(self.__x_test, self.__y_test)
        prev_eval = self.test_and_cache(self.model_id, self.current_model)
        if self.comparison.satisfied(prev_eval, new_eval):
            new_tx = self.make_new_transaction(
                TxTypeEnum.SOLVE,
                self.task_id, 
                [ Reference(tx.txid, self.tx_graph.get_evaluation_result(tx.model_id)) \
                    for tx in selected_txs ]
            )
            self.model_table[new_tx.model_id] = new_model
            self.__test_cache[new_tx.model_id] = new_eval
            self.model_id = new_tx.model_id
            self.send_new_transaction(new_tx)
    

    def __str__(self):
        return "\nnode id: " + self.nid + \
            "\nlength of data: " + str(len(self.__x_train)) + \
            "\nadjacent nodes: " + str([ 'node id: ' + n.nid for n in self.adjacent_list ]) + \
            "\ncurrent model id: " + self.model_id + \
            "\ntransaction info: " + str(self.tx_graph.get_transaction_by_id(self.model_id)) + \
            "\nevaluation rate: " + str(self.eval_rate) + \
            "\ncurr eval: " + str(self.__test_cache[self.model_id] )
