import tensorflow as tf


class FLModel:
    def __init__(self, compiled_model):
        self.__model = compiled_model
        self.__weight = self.get_weights()
        self.loss = None
        self.acc = None

    def fit(self, x_train, y_train, epochs=5, verbose=0):
        self.__model.fit(x_train, y_train, epochs=epochs, verbose=verbose)

    def evaluate(self, x_test, y_test, verbose=0):
        res = self.__model.evaluate(x_test, y_test, verbose=verbose)
        self.loss = res[0]
        self.acc = res[1]

    def raw_evaluate(self, x_test, y_test, verbose=0):
        res = self.__model.evaluate(x_test, y_test, verbose=verbose)
        return res[0], res[1]  # loss, acc

    def get_weights(self):
        self.__weight = self.__model.get_weights()
        return self.__weight

    def set_weights(self, new_weights):
        self.__model.set_weights(new_weights)


if __name__ == "__main__":
    mnist = tf.keras.datasets.fashion_mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

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

    # weights = flmodel.get_weights()
    # print(weights)

    flmodel.fit(x_train, y_train, epochs=5, verbose=1)

    flmodel.evaluate(x_test, y_test, verbose=1)
    print(flmodel.loss, flmodel.acc)
