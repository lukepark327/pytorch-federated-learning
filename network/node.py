from random import random

from graph import TxGraph
from transaction import Transaction
from policy.selection import Selection
from policy.updating import Updating

class Node:
    def __init__(
        self, 
        nid, adjacent_list,
        global_model_table, model_id, train_set, test_set, eval_rate,
        tx_graph: TxGraph, selection: Selection, updating: Updating
        ):
        self.nid = nid
        self.adjacent_list = adjacent_list
        self.model_table = global_model_table
        self.__model_id = model_id
        self.__x_train, self.__y_train = train_set
        self.__x_test, self.__y_test = test_set
        self.eval_rate = eval_rate
        self.tx_graph = tx_graph
        self.selection = selection
        self.updating = updating
    
    def get_transaction(self, tx: Transaction):
        self.tx_graph.add_transaction(tx)

        if random() < self.eval_rate:
            self.tx_graph.evaluate_and_record_model(
                model_id=tx.model_id
                model=self.model_table[tx.model_id]
            )

    def send_new_transaction(self, tx):
        for node in self.adjacent_list:
            node.get_transaction(tx)
    
    def select_transactions(self, number_of_selection):
        self.selection.update(self.tx_graph)
        return self.selection.select(number_of_selection)

    @property
    def model_id(self):
        return self.model_table[self.__model_id]
    
    @model.setter
    def model_id(self, model_id):
        self.__model_id = model_id
    
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
    
    def update(self, number_of_selection):
        """
        CAUTION: global model's id is equal with transaction id.
        """
        selected_tx = self.select_transactions(number_of_selection)
        selected_models = [ m for m in self.model_table[selected_tx.model_id] ]
        new_model = self.updating.update(selected_models)
        # TODO: 여기서부터 new_model의 성능을 테스트하고 테스트 한 결과를 통해 괜찮으면 tx를 만들어 전파.
        
    def __str__(self):
        return (
            "\nnode id: " + self.nid + \
            "\nadjacent nodes: " + self.adjacent_list + \
            "\ncurrent model id: " + self.__model_id
        )
