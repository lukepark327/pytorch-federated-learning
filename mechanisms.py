import numpy as np
import random as rand


def select(mecha, data):
    if mecha == "random":
        arr = random(data)
    elif mecha == "acute":
        arr = acute(data)
    else:
        arr = uniform(data)
    return arr


def custom(arr):
    return arr


def random(size):
    arr = np.array([rand.random() for _ in range(size)])
    arr /= sum(arr)
    return arr 


def uniform(size):
    arr = np.ones(size)
    arr *= (1 / size)
    return arr


def acute(size):
    target = rand.randrange(0, size)

    arr = np.ones(size)
    arr *= (0.1 / (size - 1))
    arr[target] = 0.9
    return arr


# def normal(size, mu=0., sd=1.):
# def pareto():
# def linear():
# et al.
