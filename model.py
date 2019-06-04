import tensorflow as tf


class Model:
    def __init__(self):
        self.model = self.build_model()
        self.weight = self.model.get_weights()
        self.loss = None
        self.acc = None

    def build_model(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(512, activation=tf.nn.relu),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation=tf.nn.softmax)
        ])
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
        return model
    
    def fit(self, x_train, y_train, epochs=5, verbose=0):
        self.model.fit(x_train, y_train, epochs=epochs, verbose=verbose)

    def evaluate(self, x_test, y_test, verbose=0):
        res = self.model.evaluate(x_test, y_test, verbose=verbose)
        self.loss = res[0]
        self.acc = res[1]

    def evaluate_by_other(self, x_test, y_test, verbose=0):
        res = self.model.evaluate(x_test, y_test, verbose=verbose)
        loss = res[0]
        acc = res[1]
        return loss, acc

    def set_weights(self, new_weights):
        self.model.set_weights(new_weights)
        self.weight = self.model.get_weights()


if __name__ == "__main__":
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = Model()
    model.fit(x_train, y_train, epochs=1)
    model.evaluate(x_test, y_test)
