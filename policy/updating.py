from enum import Enum
from typing import List
import numpy as np

from ml.task import Task, compile_model
from ml.flmodel import FLModel

class UpdatingTypeEnum(Enum):
    AVERAGE = 'average'
    W_SUM = 'weighted_sum'
    CONTINUAL = 'continual'


class Updating:
    def __init__(self, updating_type, weights=None, x_train=None, y_train=None):
        self.type = updating_type
        self.weights = weights
        self.x_train = x_train
        self.y_train = y_train

    def update(self, models: List["FLModel"], task: Task):
        if len(models) < 1 :
            raise ValueError
        
        if self.type is UpdatingTypeEnum.AVERAGE:
            w = list()
            for m in models:
                w.append(m.weights)
            w = np.array(w)
            model = task.create_base_model()
            model.weights = np.average(w, axis=0)
            return model

        elif self.type is UpdatingTypeEnum.W_SUM:
            w = list()
            for m in models:
                w.append(m.weights)
            w = np.array(w)
            model = task.create_base_model()
            model.weights = np.average(w, axis=0, weights=self.weights)
            return model

        elif self.type is UpdatingTypeEnum.CONTINUAL:
            referencing_model = task.create_base_model()
            referencing_model.weights = models[0].weights
            referencing_model.fit(self.x_train, self.y_train)
            return referencing_model

        else:
            raise TypeError