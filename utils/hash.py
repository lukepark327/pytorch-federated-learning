import hashlib
import numpy as np

class MetaFunc:
    def __init__(self, name=None, func=None):
        self.name = name
        self.func = func


def sha3_256_from_array(inputs=[]):
    SHA3 = hashlib.sha3_256()
    bytes_array = [np.array(inputs).tobytes()]
    for i in bytes_array:
        SHA3.update(i)
    return SHA3.hexdigest()


def cal_weights_hash(weights: list):
    weights_list = list()
    for weight in weights:
        b = weight.tobytes()
        weights_list.append(b)
    return sha3_256_from_array(weights_list)
