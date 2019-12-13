from enum import Enum
from typing import List
from copy import deepcopy
import numpy as np

from ml.flmodel import FLModel

class UpdatingTypeEnum(Enum):
    AVERAGE = 'average'
    W_SUM = 'weighted_sum'
    CONTINUAL = 'continual'


class Updating:
    def __init__(self, updating_type, weights, x_train=None, y_train=None):
        self.type = updating_type
        self.weights = weights
        self.x_train = x_train
        self.y_train = y_train

    def update(self, models: List["FLModel"]):
        if len(models) < 1 :
            raise ValueError
        
        if self.type is UpdatingTypeEnum.AVERAGE:
            w = list()
            for m in models:
                w.append(m.weights)
            w = np.array(w)
            model = deepcopy(models[0])
            model.weights = np.average(w, axis=0)
            return model

        elif self.type is UpdatingTypeEnum.W_SUM:
            w = list()
            for m in models:
                w.append(m.weights)
            w = np.array(w)
            model = deepcopy(models[0])
            model.weights = np.average(w, axis=0, weights=self.weights)
            return model

        elif self.type is UpdatingTypeEnum.CONTINUAL:
            referencing_model = deepcopy(models[0])
            referencing_model.fit(self.x_train, self.y_train)
            return referencing_model

        else:
            raise TypeError