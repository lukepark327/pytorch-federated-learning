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


# Input remains into last elem.
def split_dataset(dataset, num):
    unit = int(len(dataset) / num)
    res = list()
    for i in range(num):
        if i == num - 1:
            res.append(dataset[i * unit:])
        else:
            res.append(dataset[i * unit:(i + 1) * unit])
    return res


# TODO: treat np.ndarray
# Split dataset into roughly equal dataset-s
def split_dataset_equally(dataset, num):
    unit = int(len(dataset) / num)
    res = list()
    for i in range(num):
        res.append(dataset[i * unit:(i + 1) * unit])

        # remains
        if i == num - 1:
            remains = dataset[(i + 1) * unit:]
            for j, remain in enumerate(remains):
                res[j].append(remain)

    return res


if __name__ == "__main__":
    from pprint import pprint
    a = list(range(111))
    b = split_dataset(a, 12)
    pprint(b)

    hash_output = sha3_256([b'1', b'2', b'3'])
    print(hash_output, type(hash_output))
