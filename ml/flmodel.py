import tensorflow as tf
from enum import Enum


class Metric(Enum):
    LOSS = 'loss'
    ACC = 'accuracy'


class FLModel:
    def __init__(self, compiled_model, epochs):
        self.model = compiled_model
        self.__epochs = epochs

    def fit(self, x_train, y_train, callbacks=[], verbose=0):
        self.model.fit(
            x_train, y_train, epochs=self.__epochs, callbacks=callbacks, verbose=verbose
        )
    
    def evaluate(self, x_test, y_test, verbose=0):
        self.__evaluation = self.model.evaluate(x_test, y_test, verbose=verbose)
        return self.__evaluation

    @property
    def epochs(self):
        return self.__epochs

    @epochs.setter
    def epochs(self, epochs):
        self.__epochs = epochs

    @property
    def weights(self):
        return self.model.get_weights()

    @weights.setter
    def weights(self, new_weights):
        self.model.set_weights(new_weights)

    def predict(self, x_input):
        return self.model.predict(x_input)
