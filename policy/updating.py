from enum import Enum
from typing import List
from copy import deepcopy
import numpy as np

from ml.task import Task, compile_model
from ml.flmodel import FLModel

class UpdatingTypeEnum(Enum):
    AVERAGE = 'average'
    W_SUM = 'weighted_sum'
    CONTINUAL = 'continual'


class Updating:
    def __init__(
        self, 
        updating_type: UpdatingTypeEnum, 
        owner, 
        weights=None, x_train=None, y_train=None
        ):
        self.type = updating_type
        self.owner = owner
        self.weights = weights
        self.x_train = x_train
        self.y_train = y_train

    def make_new_history(self, prev_models, new_model):
        return {
            "owner": self.owner,
            "type": self.type.name,
            "model_id": new_model.model_id,
            "ref_ids": [ p.model_id for p in prev_models ]
        }
    
    def add_history_to_new_model(self, prev_models, new_model):
        prev_histories = []
        for prev_model in prev_models:
            prev_histories.append(prev_model.history)
        new_model.history = prev_histories
        new_history = self.make_new_history(prev_models, new_model)
        new_model.add_history(new_history)


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
            self.add_history_to_new_model(models, model)
            return model

        elif self.type is UpdatingTypeEnum.W_SUM:
            w = list()
            for m in models:
                w.append(m.weights)
            w = np.array(w)
            model = task.create_base_model()
            model.weights = np.average(w, axis=0, weights=self.weights)
            self.add_history_to_new_model(models, model)
            return model

        elif self.type is UpdatingTypeEnum.CONTINUAL:
            referencing_model = task.create_base_model()
            referencing_model.weights = models[0].weights
            referencing_model.fit(self.x_train, self.y_train)
            self.add_history_to_new_model(models, referencing_model)
            return referencing_model

        else:
            raise TypeError