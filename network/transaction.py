from enum import Enum
import numpy as np

from utils.hash import sha3_256_from_array, cal_weights_hash
from results.event import Event, EventType

class Reference:
    """
    Reference represents the referencing transaction's id and corresponding evaluation
    """
    def __init__(self, txid, evaluation):
        self.txid = txid
        self.evaluation = evaluation

    def __str__(self):
        return ("transaction id: " + self.txid + "\nevaluation: " + self.evaluation)


class TxTypeEnum(Enum):
    NONE = 'none'
    OPEN = 'open'
    SOLVE = 'solve'
    CLOSE = 'close'


class Transaction:
    def __init__(self, tx_type, task_id, owner, model_id, timestamp, refs):
        self.type = tx_type
        self.task_id = task_id
        self.owner = owner
        self.model_id = model_id
        self.timestamp = timestamp
        self.refs = refs

    @property
    def txid(self):
        return sha3_256_from_array(inputs=[
            self.type,
            self.task_id,
            self.owner,
            self.model_id,
            self.timestamp, 
            self.refs,
        ])
        
    @property
    def ref_ids(self):
        return [r.txid for r in self.refs]

    @property
    def meta(self):
        return {
            "ID": self.txid,
            "Type": self.type.value,
            "Owner": self.owner,
            "Model ID": self.model_id,
            "Timestamep": self.timestamp,
            "References": str(self.refs),
        }

    def __str__(self):
        return (
            "Transaction Object: " + \
            "\nid: " + self.txid + \
            "\ntype: " + self.type.name + \
            "\nowner: " + self.owner + \
            "\nmodel id: " + self.model_id + \
            "\ntimestamp: " + str(self.timestamp) + \
            "\nreferences: " + str(self.refs)
        )

def generate_genesis_tx():
    return Transaction(
        tx_type=TxTypeEnum.NONE,
        task_id='',
        owner='',
        model_id='',
        timestamp=0,
        refs=[]
    )