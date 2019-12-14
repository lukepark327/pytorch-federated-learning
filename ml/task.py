import tensorflow as tf

from ml.flmodel import FLModel


class Task:
    def __init__(self, task_id, flmodel: FLModel):
        self.__id = task_id
        self.__model = flmodel
    
    @property
    def task_id(self):
        return self.__id

    @task_id.setter
    def task_id(self, _id):
        self.__id = _id

    @property
    def task_model(self):
        return self.__model


def create_simple_sequential_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    return model

def compile_model(model, optimizer, loss, metrics, epochs):
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
    )
    return FLModel(compiled_model=model, epochs=epochs)
    