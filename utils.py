import hashlib


class MetaFunc:
    def __init__(self, name=None, func=None):
        self.name = name
        self.func = func


def sha3_256(inputs=[]):
    SHA3 = hashlib.sha3_256()
    for i in inputs:
        SHA3.update(i)
    return SHA3.hexdigest()


if __name__ == "__main__":
    hash_output = sha3_256([b'1', b'2', b'3'])
    print(hash_output, type(hash_output))
