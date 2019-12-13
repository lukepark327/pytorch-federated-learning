from enum import Enum

from network.graph import TxGraph
from ml.flmodel import FLModel

class SelectionTypeEnum(Enum):
    HEAVY_REF_TX = 'heavy_referenced_transaction'
    HEAVY_REF_OWNER_LATEST = 'heavy_referenced_owner_latest'
    HIGH_EVAL_TX = 'high_evaluated_transaction'
    HIGH_EVAL_OWNER_LATEST = 'high_evaluated_owner_latest'


class Selection:
    def __init__(self, selection_type, evaluations=list(), x_eval, y_eval):
        self.type = selection_type
        self.eval_list = evaluations

    def select(self, tx_graph: TxGraph, number_of_selection: int):
        """
        select returns the list of transactions
        """
        if (self.type is SelectionTypeEnum.HEAVY_REF_TX
            or self.type is SelectionTypeEnum.HIGH_EVAL_TX):
            return tx_graph.get_transaction_list_by_ids(
                self.eval_list[0:number_of_selection]
            )
        # For now, there are only selection rules using tx or owner
        else:
            return [ tx_graph.get_latest_transaction_by_owner(owner_id) \
                for owner_id in self.eval_list[0:number_of_selection] ]

    def update(self, tx_graph: TxGraph):
        if self.type is SelectionTypeEnum.HEAVY_REF_TX:
            self.eval_list = sorted(
                tx_graph.tx_referenced_table.iteritems(),
                key=lambda x: x[1],
                reverse=True
            )
        elif self.type is SelectionTypeEnum.HEAVY_REF_OWNER_LATEST:
            self.eval_list = sorted(
                tx_graph.owner_referenced_table.iteritems(),
                key=lambda x: x[1],
                reverse=True
            )
        elif self.type is SelectionTypeEnum.HIGH_EVAL_TX:
        elif self.type is SelectionTypeEnum.HIGH_EVAL_OWNER_LATEST:
        else:
