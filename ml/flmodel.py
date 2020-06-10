import tensorflow as tf
from enum import Enum

from utils.hash import cal_weights_hash


class Metric(Enum):
    LOSS = 'loss'
    ACC = 'accuracy'


class FLModel:
    def __init__(self, compiled_model, epochs, previous_history=list()):
        self.model = compiled_model
        self.__epochs = epochs
        self.__history_list = previous_history

    def fit(self, x_train, y_train, callbacks=[], verbose=0):
        self.model.fit(
            x_train, y_train, epochs=self.__epochs, callbacks=callbacks, verbose=verbose
        )
    
    def evaluate(self, x_test, y_test, verbose=0):
        self.__evaluation = self.model.evaluate(x_test, y_test, verbose=verbose)
        return self.__evaluation

    @property
    def model_id(self):
        return cal_weights_hash(self.weights)

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

    @property
    def history(self):
        return self.__history_list
    
    @history.setter
    def history(self, history_list):
        self.__history_list = history_list

    def add_history(self, new_history):
        self.__history_list.append(new_history)

    def __str__(self):
        s = "Model Object: \nModel ID: " + self.model_id
        s = s + "\nhistory list: "
        for history in self.__history_list:
            s += "\n\t" + str(history)
        
        return s

    @property
    def meta(self):
        return {
            "ID": self.model_id,
            "History": self.__history_list,
        }
