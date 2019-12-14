import numpy as np


def random_choices(data_length, number_of_nodes):
    choices = np.random.choice(data_length, number_of_nodes-1, replace=False)
    return sorted(choices)

def split_data_with_choices(data, choices: list()):
    return np.array_split(data, choices)

def split_data_uniform(data, number_of_nodes: int):
    return np.array_split(data, number_of_nodes)

# TODO: Non-iid settings