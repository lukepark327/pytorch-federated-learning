from enum import Enum

from network.graph import TxGraph
from ml.flmodel import FLModel

class SelectionTypeEnum(Enum):
    HEAVY_REF_TX = 'heavy_referenced_transaction'
    HEAVY_REF_OWNER_LATEST = 'heavy_referenced_owner_latest'
    HIGH_EVAL_ACC_TX = 'high_evaluated_accuracy_transaction'
    LOW_EVAL_LOSS_TX = 'low_evaluated_loss_transaction'
    HIGH_EVAL_ACC_OWNER_LATEST = 'high_evaluated_accuracy_owner_latest'


# TODO: selection needs to handle various kinds of tasks.
class Selection:
    def __init__(self, selection_type, number_of_selection=1, evaluations=list()):
        self.type = selection_type
        self.number = number_of_selection
        self.eval_list = evaluations
    
    @property
    def number_of_selection(self):
        return self.number

    @number_of_selection.setter
    def number_of_selection(self, number: int):
        self.number = number

    def select(self, tx_graph: TxGraph):
        """
        select returns the list of transactions
        """
        if (self.type is SelectionTypeEnum.HEAVY_REF_TX
            or self.type is SelectionTypeEnum.HIGH_EVAL_ACC_TX
            or self.type is SelectionTypeEnum.LOW_EVAL_LOSS_TX):
            return [ tx_graph.get_transaction_by_id(e[0])\
                for e in self.eval_list[0:self.number] ]
        # For now, there are only selection rules using tx or owner
        else:
            return [ tx_graph.get_latest_transaction_by_owner(e[0]) \
                for e in self.eval_list[0:self.number] ]

    def update(self, tx_graph: TxGraph):
        if self.type is SelectionTypeEnum.HEAVY_REF_TX:
            self.eval_list = sorted(
                iter(tx_graph.tx_referenced_table.items()),
                key=lambda x: x[1],
                reverse=True
            )
        elif self.type is SelectionTypeEnum.HEAVY_REF_OWNER_LATEST:
            self.eval_list = sorted(
                iter(tx_graph.owner_referenced_table.items()),
                key=lambda x: x[1],
                reverse=True
            )
        elif self.type is SelectionTypeEnum.HIGH_EVAL_ACC_TX:
            self.eval_list = sorted(
                iter(tx_graph.model_evaluated_results.items()),
                key=lambda x: x[1][1],
                reverse=True
            )
        elif self.type is SelectionTypeEnum.LOW_EVAL_LOSS_TX:
            self.eval_list = sorted(
                iter(tx_graph.model_evaluated_results.items()),
                key=lambda x: x[1][0],
            )
        else:
            pass
