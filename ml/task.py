import tensorflow as tf

from ml.flmodel import FLModel


class Task:
    def __init__(
        self, 
        task_id, 
        create_base_model_function,
        optimizer, loss, metrics, epochs,
        model: FLModel
        ):
        self.__id = task_id
        self.__model = model
        self.__create_func = create_base_model_function
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.epochs = epochs
    
    @property
    def task_id(self):
        return self.__id

    @task_id.setter
    def task_id(self, _id):
        self.__id = _id

    @property
    def task_model(self):
        return self.__model

    def create_base_model(self):
        return self.__create_func(
            optimizer=self.optimizer,
            loss=self.loss,
            metrics=self.metrics,
            epochs=self.epochs,
        )
    

def create_simple_sequential_model(optimizer, loss, metrics, epochs):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    return compile_model(model, optimizer, loss, metrics, epochs)

def compile_model(model, optimizer, loss, metrics, epochs):
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
    )
    return FLModel(compiled_model=model, epochs=epochs)
    