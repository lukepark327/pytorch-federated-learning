from enum import Enum


class EventType(Enum):
    EXPERIMENT_BEGIN = 'Experiment Begin'
    TASK_CREATED = 'Task Created'
    NODE_CREATED = 'Node Created'
    NODE_CONNECTED = 'Node Connected with Adjacent nodes'
    ROUND_START = 'Round Start'
    TX_CREATED = 'Transaction Created'
    TX_SENT = 'Transaction Sent'
    INIT_LOCAL_TRAIN = 'Node initiated local train'
    MODEL_UPDATED = 'Model Updated'
    MODEL_UPLOADED = 'Model Uploaded'
    MODEL_EVALUATED = 'Model Evaluated'
    MODEL_SELECTED = 'Model Selected'
    COMPARE_SATISFIED = 'Compare Satisfied'
    TX_RECEIVED = 'Transaction Received'
    ROUND_END = 'Round End'
    EXPERIMENT_END = 'Experiment End'
    NODE_RESULT = 'Node Result'
    

class Event:
    def __init__(self, event_type, meta: dict = dict()):
        self._type = event_type
        self._meta = meta

    def __str__(self):
        s = "Event Type: " + self._type.name
        for key, val in self._meta.items():
            s = s + "\n" + str(key) + ": " + str(val)
        return s + "\n"

