from copy import copy
import tensorflow as tf

from ml.task import Task, create_simple_sequential_model, compile_model

if __name__ == "__main__":
        
    # Task definition
    simple_model = create_simple_sequential_model()
    simple_fl_model = compile_model(
        model=simple_model,
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
        epochs=3
    )
    simple_task = Task('simple_task', simple_fl_model)

    a = copy(simple_task)

    print(a.task_id)
    a.task_id = 'changed'
    print(simple_task.task_id)
    print(a.task_id)