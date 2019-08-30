import hashlib
import rlp
import numpy as np
import tensorflow as tf


class Header:
    def __init__(self, n: int, prev, w, t):
        self.blockNumber = n
        self.prevBlockHash = prev
        self.weightHash = w
        self.testsetHash = t


class Block:
    def __init__(self, blockNumber: int, prevBlockHash, weights: list, testset: tuple):
        self.header = Header(
            blockNumber,
            prevBlockHash,
            self.calWeightHash(weights),
            self.calTestsetHash(testset)
        )
        self.weights = weights
        self.testset = testset

    def calBlockHash(self):
        header_list = list()
        header_list.append(bytes(self.header.blockNumber))
        header_list.append(self.header.prevBlockHash.encode('utf-8'))
        header_list.append(self.header.weightHash.encode('utf-8'))
        header_list.append(self.header.testsetHash.encode('utf-8'))

        return self.__getHash(header_list)

    def calWeightHash(self, weights: list):
        weights_list = list()
        for weight in weights:
            # print("original:", weight, weight.shape, weight.dtype.name)
            # t = weight.dtype.name
            # s = weight.shape
            b = weight.tobytes()
            # f = np.frombuffer(b, dtype=t).reshape(s)
            # print("encoded :", f, f.shape, f.dtype.name)
            weights_list.append(b)
            # if type(weight) == np.ndarray:
            #     weight = weight.tolist()
            # weights_list.append(weight)

        return self.__getHash(weights_list)

    def calTestsetHash(self, testset: tuple):
        testset_list = list()
        for i in testset:
            # (x, y)
            if type(i) != np.ndarray:
                i = np.array(i)

            b = i.tobytes()
            testset_list.append(b)

        return self.__getHash(testset_list)

    def __getHash(self, inputs=[]):
        SHA3 = hashlib.sha3_256()
        for i in inputs:
            SHA3.update(i)
        return SHA3.hexdigest()


class Blockchain:
    def __init__(self, genesisBlock):
        self.blocks = [genesisBlock]

    def append(self, block):
        self.blocks.append(block)

    def getBlock(self, blockNumber):
        return self.blocks[blockNumber]


def printBlock(block: Block):
    print("{")
    print("    \"blockNumber\"    :", block.header.blockNumber, end=",\n")
    print("    \"prevBlockHash\"  :", block.header.prevBlockHash, end=",\n")
    print("    \"weightHash\"     :", block.header.weightHash, end=",\n")
    print("    \"testsetHash\"    :", block.header.testsetHash, end=",\n")
    print("    \"(+)testsetSize\" :", len(block.testset[0]), end=",\n")
    print("    \"(+)blockHash\"   :", block.calBlockHash())
    print("}")


if __name__ == "__main__":
    from model import FLModel
    import tensorflow as tf

    # load data
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    testset = (x_test, y_test)

    # set FL model
    mnist_model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    mnist_model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    flmodel = FLModel(mnist_model)

    # set Blockchain
    init_weights = flmodel.get_weights()
    genesis = Block(
        0,
        "0" * 64,
        init_weights,
        testset
    )
    flchain = Blockchain(genesis)  # set blockchain with genesis block

    flmodel.fit(x_train, y_train, epochs=1)  # training

    nextBlockNumber = 1
    modified_weight = flmodel.get_weights()
    new_block = Block(
        nextBlockNumber,
        flchain.getBlock(nextBlockNumber - 1).calBlockHash(),
        modified_weight,
        testset
    )
    flchain.append(new_block)

    flmodel.evaluate(x_test, y_test)
    print(flmodel.loss, flmodel.acc)
