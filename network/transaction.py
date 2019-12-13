from enum import Enum
import numpy as np

from utils.hash import sha3_256_from_array, cal_weights_hash

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
    def __init__(self, tx_type, owner, timestamp, refs):
        self.type = tx_type
        self.owner = owner
        self.timestamp = timestamp
        self.refs = refs
    
    @property
    def txid(self):
        return sha3_256_from_array(inputs=[
            self.owner, 
            self.timestamp, 
            self.refs,
        ])

    @property
    def model_id(self):
        return self.txid
        
    @property
    def ref_ids(self):
        return [r.txid for r in self.refs]

    def __str__(self):
        return (
            "id: " + self.txid + \
            "\ntype: " + self.type.name + \
            "\nowner: " + self.owner + \
            "\ntimestamp: " + str(self.timestamp) + \
            "\nreferences: " + str(self.refs)
        )
